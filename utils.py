import json
import re
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs


class SocialGraph:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.graph: nx.Graph = nx.Graph()
        self.individuals: set[str] = set()

    def build(self, limit: int = 0):
        df_polars = pl.read_csv(
            self.source_file, n_rows=limit, n_threads=4, use_pyarrow=True
        )

        df_pandas = df_polars.to_pandas(use_pyarrow_extension_array=True)

        self.graph = nx.from_pandas_edgelist(
            df_pandas, source="user_a", target="user_b", create_using=nx.Graph()
        )

        self.individuals.update(self.graph.nodes())

        print(f"Successfully built social graph with {len(self.graph.nodes)} nodes.")


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


FEATURE_TYPING = {
    "feature_001": "categorical",
    "feature_002": "integer",
    "feature_003": "integer",
    "feature_004": "integer",
    "feature_005": "integer",
    "feature_006": "integer",
    "feature_007": "boolean",
    "feature_008": "integer",
    "feature_009": "integer",
    "feature_010": "float",
    "feature_011": "boolean",
    "feature_012": "categorical",
    "feature_013": "boolean",
    "feature_014": "boolean",
    "feature_015": "float",
    "feature_016": "float",
    "feature_017": "float",
    "feature_018": "float",
    "high_conf_clean": "boolean",
    "is_cheating": "boolean",
}


def get_columns_of_type(dataframe: pl.DataFrame, dtype: str):
    return list(
        set(dataframe.columns).intersection(
            set(
                col_name
                for col_name, col_typ in FEATURE_TYPING.items()
                if col_typ == dtype
            )
        )
    )


def safe_dummy_encode(df: pl.DataFrame) -> pl.DataFrame:
    categorical_cols = df.select(cs.categorical()).columns

    if not categorical_cols:
        return df

    dummies_df = (
        df.select(categorical_cols)
        .cast(pl.Utf8)
        .to_dummies(separator=":")
        .cast(pl.Boolean)
    )
    dummies_df = dummies_df.select(pl.all().exclude("^.*:missing$"))
    result_df = df.drop(categorical_cols).hstack(dummies_df)

    return result_df


def cast_dataframe_columns(dataframe: pl.DataFrame) -> pl.DataFrame:
    # Intersect available columns of the dataframe from all known types
    boolean_cols = get_columns_of_type(dataframe, "boolean")
    categorical_cols = get_columns_of_type(dataframe, "categorical")
    integer_cols = get_columns_of_type(dataframe, "integer")
    float_cols = get_columns_of_type(dataframe, "float")

    # batch recast columns with proper data type handling
    dataframe = dataframe.with_columns(
        pl.col(boolean_cols).cast(pl.Boolean).fill_null(False),
        pl.col(categorical_cols)
        .cast(pl.Int8)
        .cast(pl.Utf8)
        .fill_null(
            "missing"
        )  # to make it sort before 0 in order for drop_first to work as intended
        .cast(pl.Categorical),
        pl.col(integer_cols).cast(pl.Int8).fill_null(strategy="min"),
        pl.col(float_cols).cast(pl.Float32).fill_null(strategy="mean"),
    )
    return dataframe


class CandidateDataset:
    def __init__(self, source_train: str, source_test: str, source_feature_data: str):
        self.source_train: str = source_train
        self.source_test: str = source_test
        self.source_feature_map = source_feature_data

        self.candidates: set[str] = set()
        self.df_train: pl.DataFrame = pl.DataFrame()
        self.df_test: pl.DataFrame = pl.DataFrame()

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

    def build_datasets(self, limit: Optional[int] = None):
        def import_helper(csv_source_path):
            return (
                pl.read_csv(
                    csv_source_path,
                    n_rows=limit,
                    n_threads=4,
                    has_header=True,
                    row_index_name=None,
                )
                .with_columns(
                    pl.col("user_hash").map_elements(
                        lambda x: int(x, 16), return_dtype=pl.UInt64
                    )
                )
                .rename({"user_hash": "ID"})
            )

        self.df_train = import_helper(self.source_train)
        self.df_test = import_helper(self.source_test)

        print(
            f"Imported train dataset with {self.df_train.height} entries ({self.df_train.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )
        print(
            f"Imported test dataset with {self.df_test.height} entries ({self.df_test.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )

        self.candidates.clear()
        self.candidates.update(self.df_train["ID"].unique().to_list())
        self.candidates.update(self.df_test["ID"].unique().to_list())

    def apply_feature_typing(self):
        print("Applying feature types to datasets...")
        self.df_train = cast_dataframe_columns(self.df_train)
        self.df_test = cast_dataframe_columns(self.df_test)

    def dummy_encode_categorical(self):
        self.df_train = safe_dummy_encode(self.df_train)
        self.df_test = safe_dummy_encode(self.df_test)
