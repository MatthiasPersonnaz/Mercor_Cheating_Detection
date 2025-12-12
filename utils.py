import networkx as nx
import numpy as np
import polars as pl



class SocialGraph():
    def __init__(self, file_path: str):
        self.graph = nx.Graph()
        self.source_file = file_path
        self.individuals: set[str]
    
    def build(self):
        df = pl.read_csv(self.source_file)
        
        if not hasattr(self, 'individuals') or self.individuals is None:
            self.individuals = set()
        
        for row in df.iter_rows():
            user_a, user_b = row[0], row[1]
            
            if user_a not in self.graph:
                self.graph.add_node(user_a)
                self.individuals.add(user_a)
            
            if user_b not in self.graph:
                self.graph.add_node(user_b)
                self.individuals.add(user_b)
            
            # Add edge between the users
            self.graph.add_edge(user_a, user_b)