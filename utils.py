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
        # Read CSV with pandas for use with networkx.from_pandas_edgelist
        df = pd.read_csv(self.source_file)
        
        if limit > 0:
            df = df.head(limit)
        
        # Use networkx.from_pandas_edgelist to create graph directly
        self.graph = nx.from_pandas_edgelist(
            df, 
            source='user_a', 
            target='user_b', 
            create_using=nx.Graph()
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
        self.df_train = pl.read_csv(self.source_train)
        self.df_test = pl.read_csv(self.source_test)
