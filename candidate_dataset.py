import json
import math
import os
import re
from collections import Counter
from typing import Any, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import torch
import xgboost as xgb
from pandera.polars import Check, Column, DataFrameSchema
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from social_graph import SocialNetwork
import torch
import tqdm


# TODO: Remove the parsing logic and adapt the feature_metadata_refined.json file to use proper hardcoded ranges
def parse_range_string(range_str: str) -> Tuple[int, int]:
    """
    Parse range strings like "1-7" or "-8-100" where the second minus is the separator.
    Returns a tuple of (min_value, max_value) as integers.

    Examples:
        "1-7" -> (1, 7)
        "-8-100" -> (-8, 100)
        "5-15" -> (5, 15)
        "-10--5" -> (-10, -5)
    """
    # Use regex to find the pattern: optional minus, digits, minus, optional minus, digits
    match = re.match(r"^(-?\d+)-(-?\d+)$", range_str)

    if not match:
        raise ValueError(f"Invalid range format: {range_str}")

    try:
        min_val = int(match.group(1))
        max_val = int(match.group(2))
        return (min_val, max_val)
    except ValueError as e:
        raise ValueError(f"Could not parse range '{range_str}': {e}")


def generate_df_schema_from_metadata(
        feature_metadata: dict[str, dict[str, Any]],
) -> DataFrameSchema:
    # Start with user_hash column that is not in the feature metadata file
    columns = {"user_hash": Column(str, nullable=False)}

    for name, spec in feature_metadata.items():
        if spec["to_exclude_in_test"]:
            continue  # makeshift way to exclude columns that only exist in train set
        given_min, given_max = parse_range_string(spec["range"])
        nullable = spec["missing_values"]

        if spec["schema_type"] == "binary":
            col = Column(float, Check.isin([0.0, 1.0]), nullable=nullable)
        elif spec["schema_type"] == "float":
            col = Column(float, Check.between(given_min, given_max), nullable=nullable)
        elif spec["schema_type"] == "int":
            col = Column(int, Check.between(given_min, given_max), nullable=nullable)

        columns[name] = col

    return DataFrameSchema(columns)





class CandidateDataFrame:
    def __init__(self, source_train: str, source_test: str, source_feature_data: str):
        assert os.path.isfile(source_train)
        assert os.path.isfile(source_test)
        assert os.path.isfile(source_feature_data)

        self.source_train: str = source_train
        self.source_test: str = source_test
        self.source_feature_map = source_feature_data

        self.candidate_ids: set[int] = set()
        self.df_train: pl.DataFrame = pl.DataFrame()
        self.df_test: pl.DataFrame = pl.DataFrame()
        self.df_validation: pl.DataFrame = pl.DataFrame()
        with open(self.source_feature_map) as f:
            self.feature_metadata: dict[str, dict[str, Any]] = json.load(f)
        self.import_schema: DataFrameSchema = generate_df_schema_from_metadata(self.feature_metadata)

        print(
            "Successfully instanciated Candidate Dataset and imported feature metadata."
        )

    def display_feature_data(self):
        with open(self.source_feature_map, "r") as json_data:
            feature_data = json.load(json_data)

        for feature_name in self.df_train.columns:
            print(f"=== {feature_name} ===")
            feature_datum = feature_data.get(feature_name)
            detected_max_val = self.df_train[feature_name].max()
            detected_min_val = self.df_train[feature_name].min()
            nb_unique_values = self.df_train[feature_name].unique().count()
            print(f"  Detected range: [{detected_min_val}, {detected_max_val}]")

            if feature_datum is None:
                print("  No feature metadata provided")

            if feature_datum is not None:
                given_type = feature_datum["type"]
                given_range = feature_datum["range"]
                given_min_val, given_max_val = parse_range_string(given_range)
                nb_range_values = given_max_val - given_min_val + 1

                print(f"  Type is given: {given_type}")

                if feature_datum["type"] == "numeric":
                    print(f"  Range is given: [{given_min_val}, {given_max_val}]")

                if given_min_val >= 0 and given_max_val <= 255:
                    print("  Could be made pl.Int8 because positive <= 255")
                print(
                    f"There are {nb_unique_values} unique values in the train dataset"
                )
                if nb_unique_values <= nb_range_values and detected_max_val <= 10:
                    print(
                        "  Could be made categorical as (Unique values) <= (Range discrete values)"
                    )

    def _import_and_validate_dataset(
            self, csv_source_path, limit: Optional[int]
    ):
        dataframe: pl.DataFrame = pl.read_csv(
            csv_source_path,
            n_rows=limit,
            n_threads=4,
            has_header=True,
            row_index_name=None,
        )

        self.import_schema.validate(dataframe)

        return dataframe

    def _transform_user_hash_to_id(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        dataframe = dataframe.with_columns(
            pl.col("user_hash").map_elements(
                lambda x: int(x, 16), return_dtype=pl.UInt64
            )
        ).rename({"user_hash": "ID"})
        return dataframe

    def get_columns_of_target_type(
            self, dataframe: pl.DataFrame, dtype: str
    ) -> set[str]:
        return set([
            feat_name
            for feat_name in dataframe.columns
            if feat_name in self.feature_metadata
               and self.feature_metadata[feat_name]["target_type"] == dtype
        ])

    def _cast_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Intersect available columns of the dataframe from all known types
        boolean_cols = self.get_columns_of_target_type(df, "boolean")
        categorical_cols = self.get_columns_of_target_type(df, "categorical")
        integer_cols = self.get_columns_of_target_type(df, "integer")
        float_cols = self.get_columns_of_target_type(df, "float")

        # batch recast columns with proper data type handling
        df = df.with_columns(
            pl.col(boolean_cols).cast(pl.Boolean),
            pl.col(categorical_cols).cast(pl.Int16).cast(pl.Utf8),  # pl.Categorical?
            pl.col(integer_cols).cast(pl.Int16),
            pl.col(float_cols).cast(pl.Float32),
        )
        return df

    def _handle_missing_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        # handle is_cheating case based on the high_conf_clean column ONLY IN TRAINING SET
        if "is_cheating" in df.columns:
            df = df.with_columns(
                pl.col("is_cheating")
                .fill_null(
                    pl.when(pl.col("high_conf_clean"))
                    .then(pl.lit(False))
                    # .otherwise(pl.lit(pl.Null, dtype=pl.Boolean))
                )
            )

        # handle the special case of high_conf_clean_clean being False if absent ONLY IN TRAINING SET
        if "high_conf_clean" in df.columns:
            df = df.with_columns(
                pl.col("high_conf_clean")
                .fill_null(pl.lit(False))
            )
        return df

    def _guess_missing_features(self, df: pl.DataFrame, social_graph: Optional[nx.Graph]) -> pl.DataFrame:
        # fill the missing data with social relationships
        boolean_cols = (self.get_columns_of_target_type(df, "boolean")
                        .difference({"high_conf_clean", "is_cheating"})) # remove the formerly already handled
        categorical_cols = self.get_columns_of_target_type(df, "categorical")
        integer_cols = self.get_columns_of_target_type(df, "integer")
        float_cols = self.get_columns_of_target_type(df, "float")

        # Some mechanism to guess the missing values from the graph
        # for every point of data of id given by golumn "ID" in the dataframe df,
        # # check if it has missing values in all columns except "is_cheating" and "is_missing" if applicable,
        # then fetch its neighbors in the social graph
        # and fill the missing values of the datapoint
        #   with the average of its neighbors' values if it is numerical
        #   or by voting if it is boolean or categorical
        # keep track of the completions and display a message summarizing the number of successful completions performed

        completions = 0
        if social_graph is None:
            print(f"No completion entries with graph-based imputation were made.")
            return df
        else:
            print(f"Starting completion process with provided graph of {len(social_graph)} nodes...")

        passed_df_schema = df.schema # save for later reuse when saving the attributions
        df_dict = df.to_dict(as_series=False)
        id_list = df_dict.get("ID", [])
        id_to_row: dict[int, int] = {node_id: idx for idx, node_id in enumerate(id_list)}
        excluded_cols = {"ID", "is_cheating", "high_conf_clean"}
        if "is_missing" in df.columns:
            excluded_cols.add("is_missing")
        cols_to_guess = [col for col in df.columns if col not in excluded_cols]

        def is_missing(value: Any) -> bool:
            return value is None or (isinstance(value, float) and math.isnan(value))

        for row_idx, node_id in tqdm.tqdm(enumerate(id_list)):
            if node_id not in social_graph:
                continue

            missing_cols: list[str] = [
                col for col in cols_to_guess if is_missing(df_dict[col][row_idx])
            ]
            if not missing_cols:
                continue

            neighbor_indices: list[int] = [
                id_to_row[neighbor]
                for neighbor in social_graph.neighbors(node_id)
                if neighbor in id_to_row
            ]
            if not neighbor_indices:
                continue

            filled_any = False
            for col in missing_cols:
                neighbor_values = [
                    df_dict[col][n_idx]
                    for n_idx in neighbor_indices
                    if not is_missing(df_dict[col][n_idx])
                ]
                if not neighbor_values:
                    continue

                if col in boolean_cols:
                    vote = Counter(neighbor_values).most_common(1)[0][0]
                    df_dict[col][row_idx] = bool(vote)
                    filled_any = True
                elif col in categorical_cols:
                    vote = Counter(neighbor_values).most_common(1)[0][0]
                    df_dict[col][row_idx] = str(vote)
                    filled_any = True
                elif col in integer_cols:
                    df_dict[col][row_idx] = int(round(float(np.mean(neighbor_values))))
                    filled_any = True
                elif col in float_cols:
                    df_dict[col][row_idx] = float(np.mean(neighbor_values))
                    filled_any = True

            if filled_any:
                completions += 1

        df = pl.DataFrame(df_dict, schema=passed_df_schema)
        print(f"Detected {completions} entries with graph-based imputation.")
        return df

    def _fill_missing_features(self, df: pl.DataFrame) -> pl.DataFrame:
        boolean_cols = (self.get_columns_of_target_type(df, "boolean")
                        .difference({"high_conf_clean", "is_cheating"})) # remove the formerly already handled
        categorical_cols = self.get_columns_of_target_type(df, "categorical")
        integer_cols = self.get_columns_of_target_type(df, "integer")
        float_cols = self.get_columns_of_target_type(df, "float")

        # fill the remaining missing data with heuristics or dummy min, mean or missing thereafter
        missing_indicator_exprs = [
            # categorical columns are handled thereafter in a special way
            pl.col(col).is_null().alias(f"{col}:missing")
            for col in boolean_cols.union(integer_cols).union(float_cols) if
            self.feature_metadata[col]["missing_values"]
        ]

        if missing_indicator_exprs:
            df = df.with_columns(missing_indicator_exprs)

        df = df.with_columns(
            pl.col(boolean_cols).fill_null(False),
            pl.col(categorical_cols).fill_null("missing"),
            pl.col(integer_cols).fill_null(strategy="mean"),
            pl.col(float_cols).fill_null(strategy="mean"),
        )
        return df

    def _encode_dummies(self, df: pl.DataFrame, discard_missing: bool = False) -> pl.DataFrame:
        categorical_cols = self.get_columns_of_target_type(df, "categorical")
        if not categorical_cols:
            return df

        dummies = (
            df.select(categorical_cols)
            .to_dummies(separator=":")
            .cast(pl.Boolean)
        )
        if discard_missing:
            dummies = dummies.select(pl.all().exclude("^.*:missing$"))

        df = df.drop(categorical_cols).hstack(dummies)
        return df

    def _compute_mean_std(self, df) -> dict[str, tuple[float, float]]:
        integer_cols = self.get_columns_of_target_type(df, "integer")
        float_cols = self.get_columns_of_target_type(df, "float")

        numerical_cols = integer_cols.union(float_cols)
        return {col: (df[col].mean(), df[col].std()) for col in numerical_cols}

    def _normalize_columns(self, df: pl.DataFrame, stats_dict: dict[str, tuple[float, float]]):
        def signed_log_transform(expr: pl.Expr, eps) -> pl.Expr:
            return (
                pl.when(expr == 0)
                .then(0.0)
                .otherwise(expr.sign() * (expr.abs() + eps).log10())
            )

        cols = ["feature_015", "feature_016"]
        df = df.with_columns(
            [signed_log_transform(pl.col(c), .5).alias(c) for c in cols]
        )

        df = df.with_columns(
            [
                ((pl.col(col).cast(pl.Float32) - mean) / std).alias(col)
                for (col, (mean, std)) in stats_dict.items()
            ]
        )
        return df

    def _enrich_features(self, network: SocialNetwork):
        # TODO: Enrich features using the graph with other indicators
        pass

    def build_pipeline(self, limit: Optional[int] = None, fill_missing: bool = False, encode_categorical: bool = False,
                       normalize_cols: bool = False, guess_missing: bool = False, social_network: Optional[nx.Graph] = None):

        # Import and validate datasets
        df_train = self._import_and_validate_dataset(
            self.source_train, limit=limit
        )
        df_test = self._import_and_validate_dataset(
            self.source_test, limit=limit
        )
        print(f"Loaded datasets (train: {df_train.height} rows, test: {df_test.height} rows).")

        # Transform user_hash to integer ID
        df_train = self._transform_user_hash_to_id(df_train)
        df_test = self._transform_user_hash_to_id(df_test)
        print("Transformed user_hash to integer IDs.")

        # Cast columns to proper types and encode categorical features as dummies
        df_train = self._cast_columns(df_train)
        df_test = self._cast_columns(df_test)
        print("Casted columns to target dtypes.")

        # Handle missing labels
        df_train = self._handle_missing_labels(df_train)
        print("Handled missing labels in training set.")

        # Guess missing features based on the graph data
        if guess_missing:
            # TODO: Later merge the dataframes to perform that step in order to avoid second-guessing test features
            df_train = self._guess_missing_features(df_train, social_network)
            df_test = self._guess_missing_features(df_test, social_network)
            print("Guessed missing features using graph context.")

        # Fill missing features
        if fill_missing:
            df_train = self._fill_missing_features(df_train)
            df_test = self._fill_missing_features(df_test)
            print("Filled remaining missing features with defaults.")

        # Encode categorical features as dummies
        if encode_categorical:
            df_train = self._encode_dummies(df_train)
            df_test = self._encode_dummies(df_test)
            print("Encoded categorical features.")

        if normalize_cols:
            train_stats_dict = self._compute_mean_std(df_train)
            df_train = self._normalize_columns(df_train, train_stats_dict)
            df_test = self._normalize_columns(df_test, train_stats_dict)
            print("Normalized numerical columns using train statistics.")

        self.df_train = df_train
        self.df_test = df_test

        print(
            f"Imported train dataset with {self.df_train.height} entries ({self.df_train.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )
        print(
            f"Imported test dataset with {self.df_test.height} entries ({self.df_test.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )

    def build_candidate_id_set(self):
        self.candidate_ids.clear()
        self.candidate_ids.update(self.df_train["ID"].unique().to_list())
        self.candidate_ids.update(self.df_test["ID"].unique().to_list())

    def compute_isolate_candidates_from_network(self, network: SocialNetwork):
        diff1 = self.candidate_ids.difference(network.node_ids)
        print(
            f"There are {len(diff1)} elements in the dataset which are not present in the network."
        )
        diff2 = network.node_ids.difference(self.candidate_ids)
        print(
            f"There are {len(diff2)} elements in the network which are not present in the dataset."
        )

    def split(
            self,
            test_fraction: float = 0.2,
            seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Shuffle a dataframe deterministically and split it into train/test subsets.
        """

        data_train, data_val = train_test_split(
            self.df_train,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=self.df_train["is_cheating"]
        )

        print(
            f"Split dataframe into {data_train.height} training and {data_val.height} validation entries."
        )

        return data_train, data_val

    def fit_xgboost_model(self, data_train: pl.DataFrame, params: dict) -> xgb.XGBModel:
        """
        Train an XGBoost booster on the curated dataframe.
        Drops the same non-feature columns used during evaluation to keep shapes aligned.
        """
        drop_cols = [c for c in ["is_cheating", "high_conf_clean", "ID"] if c in data_train.columns]
        features = data_train.drop(drop_cols)
        labels = data_train["is_cheating"]

        dtrain = xgb.DMatrix(features, label=labels)

        # Set a small sane default if the caller did not pass anything useful
        train_params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 6,
        }

        model = xgb.train(train_params, dtrain)
        print("XGBoost model trained successfully!")
        return model

    def test_xgboost_model(self, model, data_val: pl.DataFrame) -> dict[str, float]:
        """
        Evaluate a trained XGBoost booster on the provided dataframe using log loss and accuracy.
        Uses direct metric computation instead of parsing xgboost's string output.
        """
        drop_cols = [c for c in ["is_cheating", "high_conf_clean", "ID"] if c in data_val.columns]
        features = data_val.drop(drop_cols)
        labels = data_val["is_cheating"]

        deval = xgb.DMatrix(features, label=labels)
        preds = model.predict(deval)
        labels_np = labels.to_numpy()

        eval_logloss = float(log_loss(labels_np, preds))
        eval_accuracy = float(accuracy_score(labels_np, preds >= 0.5))

        print(f"Evaluation metrics -> LogLoss: {eval_logloss:.5f}, Accuracy: {eval_accuracy:.4f}")

        return {"logloss": eval_logloss, "accuracy": eval_accuracy}

    def sample_from_train(self, is_cheating: Optional[bool], n_samples: int) -> torch.Tensor:
        return (self.df_train
         .filter(pl.col("is_cheating") == is_cheating)
         .select(pl.exclude(["ID", "is_cheating", "high_conf_clean"]))
         .sample(n=n_samples, with_replacement=False)
         .to_torch())
