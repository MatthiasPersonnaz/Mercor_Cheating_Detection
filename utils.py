import networkx as nx
import numpy as np
import pandas as pd
import polars as pl


class SocialGraph:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.graph: nx.Graph = nx.Graph()
        self.individuals: set[str] = set()

    def build(self, limit: int = 0):
        # Read CSV efficiently with polars (multi-threaded)
        df_polars = pl.read_csv(self.source_file, n_rows=limit, n_threads=4)

        # Convert to pandas DataFrame for use with networkx.from_pandas_edgelist
        df_pandas = df_polars.to_pandas(use_pyarrow_extension_array=True)

        # Use networkx.from_pandas_edgelist to create graph directly
        self.graph = nx.from_pandas_edgelist(
            df_pandas, source="user_a", target="user_b", create_using=nx.Graph()
        )

        # Update individuals set with all nodes from the graph
        self.individuals.update(self.graph.nodes())


class CandidateDataset:
    def __init__(self, source_train: str, source_test: str):
        self.source_train: str = source_train
        self.source_test: str = source_test

        self.candidates: set[str] = {}
        self.df_train: pl.DataFrame = pl.DataFrame()
        self.df_test: pl.DataFrame = pl.DataFrame()

    def build(self, limit: int = 0):
        self.df_train = pl.read_csv(self.source_train, n_rows=limit, n_threads=4)
        self.df_test = pl.read_csv(self.source_test, n_rows=limit, n_threads=4)
