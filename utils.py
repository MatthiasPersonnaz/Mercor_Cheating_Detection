import json
import os
import re
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
import polars.selectors as cs
from pandera.polars import Check, Column, DataFrameSchema


class SocialGraph:
    def __init__(self, source_file: str):
        assert os.path.isfile(source_file)
        self.source_file = source_file
        self.graph: nx.Graph = nx.Graph()
        self.candidate_ids: set[int] = set()
        self.df_edges: pl.DataFrame = pl.DataFrame()

    def build(self, limit: int = 0):
        df_edges = pl.read_csv(
            self.source_file, n_rows=limit, n_threads=4, use_pyarrow=True
        )

        schema = DataFrameSchema(
            {
                "user_a": Column(str, nullable=False),
                "user_b": Column(str, nullable=False),
            }
        )
        schema.validate(df_edges)


        df_edges = df_edges.with_columns(
            pl.col(["user_a", "user_b"]).map_elements(
                lambda x: int(x, 16), return_dtype=pl.UInt64
            )
        ).rename({"user_a": "ID_A", "user_b": "ID_B"})

        self.graph = nx.from_pandas_edgelist(
            df_edges, source="ID_A", target="ID_B", create_using=nx.Graph()
        )
        
        self.df_edges = df_edges
        self.candidate_ids.update(self.graph.nodes())

        print(
            f"Successfully built & validated social graph with {len(self.graph.nodes)} nodes."
        )


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


def generate_schema_from_metadata(json_path: str) -> DataFrameSchema:
    with open(json_path) as f:
        meta = json.load(f)

    columns = {"user_hash": Column(str, nullable=False)}

    for name, spec in meta.items():
        given_min, given_max = parse_range_string(spec["range"])
        nullable = spec["missing_%"] > 0.0

        if spec["type"] == "binary":
            col = Column(float, Check.isin([0.0, 1.0]), nullable=nullable)
        elif spec["type"] == "float":
            col = Column(float, Check.between(given_min, given_max), nullable=nullable)
        elif spec["type"] == "int":
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

        print("Successfully instanciated Candidate Dataset")

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

    def import_and_validate_dataset(
        self, csv_source_path, limit: int, schema: DataFrameSchema
    ):
        dataframe: pl.DataFrame = pl.read_csv(
            csv_source_path,
            n_rows=limit,
            n_threads=4,
            has_header=True,
            row_index_name=None,
        )
        schema.validate(dataframe)

        dataframe = dataframe.with_columns(
            pl.col("user_hash").map_elements(
                lambda x: int(x, 16), return_dtype=pl.UInt64
            )
        ).rename({"user_hash": "ID"})

        return dataframe

    def build_datasets(self, limit: Optional[int] = None):
        schema = generate_schema_from_metadata(self.source_feature_map)

        self.df_train = self.import_and_validate_dataset(
            self.source_train, limit=limit, schema=schema
        )
        self.df_test = self.import_and_validate_dataset(
            self.source_test, limit=limit, schema=schema
        )

        print(
            f"Imported train dataset with {self.df_train.height} entries ({self.df_train.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )
        print(
            f"Imported test dataset with {self.df_test.height} entries ({self.df_test.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )

        self.candidate_ids.clear()
        self.candidate_ids.update(self.df_train["ID"].unique().to_list())
        self.candidate_ids.update(self.df_test["ID"].unique().to_list())

    def apply_feature_typing(self):
        print("Applying feature types to datasets...")
        self.df_train = cast_dataframe_columns(self.df_train)
        self.df_test = cast_dataframe_columns(self.df_test)

    def dummy_encode_categorical(self):
        self.df_train = safe_dummy_encode(self.df_train)
        self.df_test = safe_dummy_encode(self.df_test)

    def compute_isolate_candidates_from_network(self, network: SocialGraph):
        difference = self.candidate_ids.difference(network.candidate_ids)
        print(f"There are {len(difference)} elements in the dataset which are not present in the network.")
