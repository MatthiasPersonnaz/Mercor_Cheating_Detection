import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Tuple
import json
import re


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
    "feature_001": pl.Enum,
    "feature_002": pl.Int8,
    "feature_003": pl.Int8,
    "feature_004": pl.Int8,
    "feature_005": pl.Int8,
    "feature_006": pl.Int8,
    "feature_007": pl.Boolean,
    "feature_008": pl.Int8,
    "feature_009": pl.Int8,
    "feature_010": pl.Float32,
    "feature_011": pl.Boolean,
    "feature_012": pl.Enum,
    "feature_013": pl.Boolean,
    "feature_014": pl.Boolean,
    "feature_015": pl.Float32,
    "feature_016": pl.Float32,
    "feature_017": pl.Float32,
    "feature_018": pl.Float32,
    "high_conf_clean": pl.Boolean,
    "is_cheating": pl.Boolean
}

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
                given_type = feature_datum['type']
                given_range = feature_datum["range"]
                given_min_val, given_max_val = parse_range_string(given_range)
                nb_range_values = given_max_val - given_min_val + 1

                print(f"  Type is given: {given_type}")

                if feature_datum["type"] == "numeric":
                    print(f"  Range is given: [{given_min_val}, {given_max_val}]")


                if given_min_val >= 0 and given_max_val <= 255:
                    print("  Could be made pl.Int8 because positive <= 255")
                print(f"There are {nb_unique_values} unique values in the train dataset")
                if nb_unique_values <= nb_range_values and detected_max_val <= 10:
                    print("  Could be made categorical as (Unique values) <= (Range discrete values)")


    def build_datasets(self, limit: Optional[int] = None):
        self.df_train = pl.read_csv(
            self.source_train, n_rows=limit, n_threads=4, has_header=True
        )
        self.df_test = pl.read_csv(
            self.source_test, n_rows=limit, n_threads=4, has_header=True
        )
        print(
            f"Imported train dataset with {self.df_train.height} entries ({self.df_train.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )
        print(
            f"Imported test dataset with {self.df_test.height} entries ({self.df_test.select(pl.any_horizontal(pl.all().is_null())).sum().item()} of which have missing features)"
        )

        self.candidates.clear()
        self.candidates.update(self.df_train["user_hash"].unique().to_list())
        self.candidates.update(self.df_test["user_hash"].unique().to_list())

    def apply_feature_typing(self):
        print("Applying feature types to datasets...")

        def cast_expr(col_name, target_type, df):
            """Create casting expression with optional enum category detection"""
            if target_type == pl.Categorical:
                # For categorical columns: numeric → string → categorical
                return pl.col(col_name).cast(pl.Utf8).cast(pl.Categorical).alias(col_name)
            elif target_type == pl.Enum:
                # For enum columns: detect categories from unique values, then create enum
                categories = df[col_name].unique().sort().to_list()
                enum_dtype = pl.Enum(categories)
                return pl.col(col_name).cast(pl.Utf8).cast(enum_dtype).alias(col_name)
            else:
                return pl.col(col_name).cast(target_type).alias(col_name)

        # Apply to train dataset - only columns that exist
        train_exprs = [
            cast_expr(col, typ, self.df_train) 
            for col, typ in FEATURE_TYPING.items() 
            if col in self.df_train.columns
        ]
        if train_exprs:
            self.df_train = self.df_train.with_columns(train_exprs)
            print(f"Recast {len(train_exprs)} columns in train dataset")

        # Apply to test dataset - only columns that exist (safely handles missing target columns)
        test_exprs = [
            cast_expr(col, typ, self.df_test) 
            for col, typ in FEATURE_TYPING.items() 
            if col in self.df_test.columns
        ]
        if test_exprs:
            self.df_test = self.df_test.with_columns(test_exprs)
            print(f"Recast {len(test_exprs)} columns in test dataset")
        
        # Report any columns that were skipped
        train_columns = set(self.df_train.columns)
        test_columns = set(self.df_test.columns)
        feature_columns = set(FEATURE_TYPING.keys())
        
        missing_in_train = feature_columns - train_columns
        missing_in_test = feature_columns - test_columns
        
        if missing_in_train:
            print(f"Note: Columns not found in train dataset: {missing_in_train}")
        if missing_in_test:
            print(f"Note: Columns not found in test dataset: {missing_in_test}")
