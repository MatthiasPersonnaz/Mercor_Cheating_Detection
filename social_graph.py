import json
import os
import re
from typing import Any, Optional, Tuple

import networkx as nx
import numpy as np
import polars as pl
from pandera.polars import Check, Column, DataFrameSchema
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from polars import DataFrame
import scipy
from sklearn.manifold import SpectralEmbedding
import tqdm
from itertools import combinations


class SocialGraph:
    def __init__(self, source_file: str):
        self.communities: list[int] = []
        assert os.path.isfile(source_file)
        self.source_file: str = source_file
        self.graph: nx.Graph = nx.Graph()
        self.candidate_ids: set[int] = set()
        self.df_edges: pl.DataFrame = pl.DataFrame()

    def build_from_edges(self, limit: Optional[int] = None):
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

        complete_social_graph = nx.from_pandas_edgelist(
            df_edges,
            source="ID_A",
            target="ID_B",
            create_using=nx.Graph(),
        )

        self.graph = complete_social_graph
        self.df_edges = df_edges
        self.candidate_ids.update(self.graph.nodes())
        self.communities = [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

        print("Built graph successfully")
        print("Number of nodes:", len(self.graph.nodes()))
        print("Number of edges:", len(self.graph.edges()))

    def prune_from_datasets(self, dataset_train: DataFrame, dataset_test: DataFrame):
        train_nodes = set(dataset_train["ID"].unique().to_list())
        test_nodes = set(dataset_test["ID"].unique().to_list())
        dataset_nodes = train_nodes | test_nodes
        # complete_graph_nodes = set(self.graph.nodes())

        print("Beginning pruning...")

        for component in tqdm.tqdm(self.communities):
            nodes_deg_1 = {u for u in component if self.graph.degree(u) == 1}
            uninformative_nodes = nodes_deg_1.difference(dataset_nodes) # nodes that do not appear in the dataset, to discard
            self.graph.remove_nodes_from(uninformative_nodes)

        # update communities
        self.communities = [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]
        print("Pruned graph successfully")
        print("Number of nodes:", len(self.graph.nodes()))
        print("Number of edges:", len(self.graph.edges()))

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

    def plot_subgraph_dataset_labels(
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

    def export_subgraph_dataset_labels_dot(
            self,
            community_idx: int,
            dataset_train: DataFrame,
            dataset_test: DataFrame,
            output_path: str,
            show_node_ids: bool = False,
            node_diameter_inches: float = 0.15,
    ):
        subgraph = self.get_subgraph_view(community_idx)
        if subgraph.number_of_nodes() == 0:
            raise ValueError(f"Community index {community_idx} has no nodes to export.")

        node_fill_colors = {}

        for node in subgraph.nodes():
            found_in_train = dataset_train.filter(pl.col("ID") == node)
            found_in_test = dataset_test.filter(pl.col("ID") == node)

            if not found_in_train.is_empty():
                is_cheating = False
                if not found_in_train.is_empty():
                    is_cheating = bool(found_in_train["is_cheating"][0])
                node_fill_colors[node] = "red" if is_cheating else "green"
            elif not found_in_test.is_empty():
                node_fill_colors[node] = "yellow"
            else:
                node_fill_colors[node] = "blue"

        # Avoid mutating the cached subgraph.
        export_graph = nx.Graph(subgraph)
        node_attributes = {}
        for node, color in node_fill_colors.items():
            node_attributes[node] = {
                "fillcolor": color,
                "style": "filled",
                "label": str(node) if show_node_ids else "",
                "shape": "circle",
                "fixedsize": "true",
                "width": node_diameter_inches,
                "height": node_diameter_inches,
            }

        nx.set_node_attributes(export_graph, node_attributes)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        write_dot(export_graph, output_path)

    def plot_subgraph_spectral_structure(self, community_idx: int):
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
