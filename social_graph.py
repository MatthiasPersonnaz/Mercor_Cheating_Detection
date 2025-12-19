import json
import os
import re
from typing import Any, Optional, Tuple

import networkx as nx
import numpy as np
import polars as pl
from pandera.polars import Check, Column, DataFrameSchema
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from polars import DataFrame
import scipy
from sklearn.manifold import SpectralEmbedding
import tqdm
from itertools import combinations

class SocialGraph:
    def __init__(self, source_file: str):
        self.communities: list[int] = []
        assert os.path.isfile(source_file)
        self.source_file = source_file
        self.graph: nx.Graph = nx.Graph()
        self.candidate_ids: set[int] = set()
        self.df_edges: pl.DataFrame = pl.DataFrame()

    def build(
            self, limit: Optional[int] = None, database_nodes_subset: Optional[set[int]] = None
    ):
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

        graph = nx.from_pandas_edgelist(
            df_edges,
            source="ID_A",
            target="ID_B",
            create_using=nx.Graph(),
        )
        print("Graph loaded successfully")
        print("Number of nodes:", len(graph.nodes()))
        print("Number of edges:", len(graph.edges()))

        # contract here
        if database_nodes_subset is not None:
            print("Beginning pruning...")
            nodes_to_keep = set(graph.nodes()).intersection(database_nodes_subset)

            contracted_graph = nx.Graph()
            contracted_graph.add_nodes_from(nodes_to_keep)
            connected_components = nx.connected_components(graph)
            for idx, component in enumerate(connected_components):
                comp_nodes = nodes_to_keep.intersection(component)
                if len(comp_nodes) <= 2:
                    continue

                subgraph = graph.subgraph(component)

                # multi-source BFS from each kept node, restricted to component
                # visited = set()
                #
                lengths_generator = nx.all_pairs_shortest_path_length(subgraph, backend="cugraph")

                keep = nodes_to_keep  # already a set
                add_edge = contracted_graph.add_edge

                for u, dist_dict in tqdm.tqdm(lengths_generator):
                    if u not in keep:
                        continue

                    # local bindings (critical for speed)
                    u_local = u
                    for v, d in dist_dict.items():
                        if v in keep and v > u_local:
                            add_edge(u_local, v, weight=d)
                # print(f"There are {len(edges)} edges in component {idx} of size {len(comp_nodes)}")

                contracted_graph.add_edges_from()
                graph = contracted_graph

        self.graph = graph
        self.df_edges = df_edges
        self.candidate_ids.update(self.graph.nodes())
        self.communities = [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

        print(
            f"Successfully built & validated social graph with {len(self.graph.nodes)} nodes."
        )

    def get_subgraph_view(self, comm_idx: int):
        return nx.induced_subgraph(self.graph, self.communities[comm_idx])

    def plot_subgraph_view_(self, comm_idx: int, candidate_dataset: DataFrame):
        subgraph = self.get_subgraph_view(comm_idx)
        pos = graphviz_layout(subgraph, prog="sfdp")  # dot, neato, fdp, sfdp, twopi
        nx.draw(subgraph, pos, with_labels=False, node_size=20, font_size=10)
        plt.show()

    def get_attributes_relationship_matrices(
            self, subgraph: nx.Graph, dataset: DataFrame
    ) -> Tuple[Any, Any]:
        # Implement a method that computes attribute-based relationship matrices
        return None, None

    def get_graph_relationship_matrices(self, subgraph: nx.Graph) -> Tuple[Any, Any]:
        adj_mat = nx.to_scipy_sparse_array(
            subgraph, nodelist=subgraph.nodes, dtype=float
        ).tocsr()
        lapl_mat = scipy.sparse.csgraph.laplacian(adj_mat, normed=True)

        return adj_mat, lapl_mat

    def get_spectral_embeddings(self, subgraph: nx.Graph, n_components: int):
        adj_mat, _ = self.get_graph_relationship_matrices(subgraph)
        spectral_embedder = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",  # to use our similarity_matrix
            random_state=42,  # for reproducibility
            n_jobs=-1,  # uses all available processors
        )

        embeddings = spectral_embedder.fit_transform(adj_mat.toarray())
        return embeddings

    def plot_subgraph_view_by_datasets(
            self, community_idx: int, dataset_train: DataFrame, dataset_test: DataFrame
    ):
        subgraph = self.get_subgraph_view(community_idx)
        pos = graphviz_layout(subgraph, prog="sfdp")  # dot, neato, fdp, sfdp, twopi

        # Create a dictionary to store node colors
        node_colors = {}

        # Check each node in the subgrapsh against the candidate dataset
        for node in subgraph.nodes():
            # Look for the node ID in both train and test dataframes
            found_in_train = dataset_train.filter(pl.col("ID") == node)
            found_in_test = dataset_test.filter(pl.col("ID") == node)

            if not found_in_train.is_empty():
                # Node found in candidate dataset
                is_cheating = False

                # Check train data
                if not found_in_train.is_empty():
                    is_cheating = found_in_train["is_cheating"][0]

                # Set color based on cheating status
                node_colors[node] = "red" if is_cheating else "green"
            elif not found_in_test.is_empty():
                node_colors[node] = "yellow"
            else:
                # Node not found in candidate dataset
                node_colors[node] = "blue"

        # Draw the graph with colored nodes
        nx.draw(
            subgraph,
            pos,
            with_labels=False,
            node_size=20,
            font_size=10,
            node_color=list(node_colors.values()),
        )
        plt.show()

    def plot_subgraph_view_by_network(self, community_idx: int):
        # TODO:
        subgraph = self.get_subgraph_view(community_idx)
        if subgraph.number_of_nodes() == 0:
            return

        pos = graphviz_layout(subgraph, prog="sfdp")
        embeddings = np.asarray(
            self.get_spectral_embeddings(subgraph=subgraph, n_components=3)
        )
        nodes = list(subgraph.nodes())

        # Normalize each embedding dimension so it can be interpreted as an RGB channel.
        min_vals = embeddings.min(axis=0)
        ranges = embeddings.max(axis=0) - min_vals
        ranges[ranges == 0] = 1  # avoid divide-by-zero when component is constant
        normalized_embeddings = (embeddings - min_vals) / ranges
        node_colors = [tuple(color) for color in normalized_embeddings.tolist()]

        nx.draw(
            subgraph,
            pos,
            with_labels=False,
            node_size=20,
            font_size=10,
            nodelist=nodes,
            node_color=node_colors,
        )
        plt.show()
