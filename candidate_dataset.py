import json
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import xgboost as xgb
from pandera.polars import Check, Column, DataFrameSchema
from sklearn.model_selection import train_test_split
from social_graph import SocialGraph


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





class CandidateDataset:
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
        self.schema: DataFrameSchema = generate_df_schema_from_metadata(self.feature_metadata)

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

        self.schema.validate(dataframe)

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

    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
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

        # TODO: Add some mechanism to guess the missing values from the graph for the other columns

        boolean_cols = (self.get_columns_of_target_type(df, "boolean")
                        .difference({"high_conf_clean", "is_cheating"})) # remove the formerly already handled
        categorical_cols = self.get_columns_of_target_type(df, "categorical")
        integer_cols = self.get_columns_of_target_type(df, "integer")
        float_cols = self.get_columns_of_target_type(df, "float")

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
            pl.col(integer_cols).fill_null(strategy="min"),
            pl.col(float_cols).fill_null(strategy="min"),
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

        df = df.with_columns(
            [
                ((pl.col(col).cast(pl.Float32) - mean) / std).alias(col)
                for (col, (mean, std)) in stats_dict.items()
            ]
        )
        return df

    def build_pipeline(self, limit: Optional[int] = None, fill_missing: bool = False, encode_categorical: bool = False,
                       normalize_cols: bool = False):

        # Import and validate datasets
        df_train = self._import_and_validate_dataset(
            self.source_train, limit=limit
        )
        df_test = self._import_and_validate_dataset(
            self.source_test, limit=limit
        )

        # Transform user_hash to integer ID
        df_train = self._transform_user_hash_to_id(df_train)
        df_test = self._transform_user_hash_to_id(df_test)

        # Cast columns to proper types and encode categorical features as dummies
        df_train = self._cast_columns(df_train)
        df_test = self._cast_columns(df_test)

        # Handle missing values
        if fill_missing:
            df_train = self._handle_missing_values(df_train)
            df_test = self._handle_missing_values(df_test)

        # Encode categorical features as dummies
        if encode_categorical:
            df_train = self._encode_dummies(df_train)
            df_test = self._encode_dummies(df_test)

        if normalize_cols:
            train_stats_dict = self._compute_mean_std(df_train)
            df_train = self._normalize_columns(df_train, train_stats_dict)
            df_test = self._normalize_columns(df_test, train_stats_dict)

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

    def compute_isolate_candidates_from_network(self, network: SocialGraph):
        diff1 = self.candidate_ids.difference(network.candidate_ids)
        print(
            f"There are {len(diff1)} elements in the dataset which are not present in the network."
        )
        diff2 = network.candidate_ids.difference(self.candidate_ids)
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
        shuffled = self.df_train.sample(fraction=1.0, with_replacement=False, seed=seed)

        df_train, df_test = train_test_split(
            shuffled,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
        )

        print(
            f"Split dataframe into {df_train.height} training and {df_test.height} validation entries."
        )

        return df_train, df_test

    def fit_xgboost_model(self, df_train: pl.DataFrame, params: Optional[dict]) -> xgb.XGBModel:
        train_data = df_train.drop("is_cheating")
        target_data = df_train["is_cheating"]

        dtrain = xgb.DMatrix(train_data, label=target_data)

        model_params = params if params is not None else {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }

        bst = xgb.train(params, dtrain)
        print("XGBoost model trained successfully!")
        return bst

    def evaluate_xgboost_model(self, model, df_eval: pl.DataFrame) -> dict[str, float]:
        """
        Evaluate a trained XGBoost booster on the provided dataframe using log loss and accuracy.
        """
        features = df_eval.drop("is_cheating")
        labels = df_eval["is_cheating"]

        deval = xgb.DMatrix(features, label=labels)
        eval_result = model.eval(deval, name="eval")

        metric_values: dict[str, float] = {}
        for entry in eval_result.strip().split("\t"):
            entry = entry.strip()
            if not entry or entry.startswith("[") or ":" not in entry:
                continue
            metric, value = entry.split(":", 1)
            try:
                metric_values[metric] = float(value)
            except ValueError:
                continue

        preds = model.predict(deval)
        labels_np = labels.to_numpy()
        accuracy = float(((preds >= 0.5) == labels_np).mean())

        logloss = metric_values.get("eval-logloss")
        if logloss is not None:
            print(
                f"Evaluation metrics -> LogLoss: {logloss:.5f}, Accuracy: {accuracy:.4f}"
            )
        else:
            print(f"Evaluation metrics -> {eval_result}, Accuracy: {accuracy:.4f}")

        return {"logloss": logloss, "accuracy": accuracy, "raw_eval": eval_result}
