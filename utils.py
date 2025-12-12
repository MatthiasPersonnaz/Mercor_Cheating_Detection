import networkx as nx
import numpy as np
import polars as pl
import csv


class SocialGraph:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.graph: nx.Graph = nx.Graph()
        self.individuals: set[str] = set()
    
    def build(self, limit: int = 0):
        if not hasattr(self, 'individuals') or self.individuals is None:
            self.individuals = set()
        
        with open(self.source_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            
            for counter,row in enumerate(reader):
                user_a, user_b = row[0], row[1]
                
                if user_a not in self.graph:
                    self.graph.add_node(user_a)
                    self.individuals.add(user_a)
                
                if user_b not in self.graph:
                    self.graph.add_node(user_b)
                    self.individuals.add(user_b)
                
                self.graph.add_edge(user_a, user_b)

                if counter > limit:
                    break




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
