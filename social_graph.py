import json
import os
import re
from typing import Any, Iterable, Optional, Tuple, Union

import networkx as nx
import numpy as np
import polars as pl
from pandera.polars import Check, Column, DataFrameSchema
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from polars import DataFrame
from sklearn.manifold import SpectralEmbedding
import tqdm
import scipy
from scipy.sparse import csr_matrix

import cupy as cp
import cupyx
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg as cu_linalg


def to_int32_sparse(mat):
    mat = mat.tocsr()
    mat.indices = mat.indices.astype(np.int32)
    mat.indptr = mat.indptr.astype(np.int32)
    return mat


class SocialGraph:
    def __init__(self, source_file: str):
        self.communities: list[int] = []
        assert os.path.isfile(source_file)
        self.source_file: str = source_file
        self.graph: nx.Graph = nx.Graph()
        self.node_ids: set[int] = set()
        self.df_edges: pl.DataFrame = pl.DataFrame()
        self.node_embeddings: Optional[np.ndarray] = None
        self.node_id_to_embedding_row: dict[int, int] = {}

    def _reset_embedding_cache(self) -> None:
        self.node_embeddings = None
        self.node_id_to_embedding_row = {}

    def build_from_links(self, limit: Optional[int] = None):
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
        self.node_ids.clear()
        self.node_ids.update(self.graph.nodes())
        self.communities = [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

        print("Built graph successfully")
        print("Number of nodes:", len(self.graph.nodes()))
        print("Number of edges:", len(self.graph.edges()))
        print(f"There are {len(self.communities)} communities")
        self._reset_embedding_cache()

    def prune_from_datasets(self, dataset_train: DataFrame, dataset_test: DataFrame, iter_cutoff: int = None) -> None:
        train_nodes = set(dataset_train["ID"].unique().to_list())
        test_nodes = set(dataset_test["ID"].unique().to_list())
        dataset_nodes = train_nodes | test_nodes
        # complete_graph_nodes = set(self.graph.nodes())

        print("Beginning pruning...")

        # Continuously remove leaves that are not present in any dataset.
        # As nodes are removed, new leaves appear; iterate until no more deletions.
        iter = 0
        candidate_nodes = set(self.graph.nodes()).difference(dataset_nodes)
        while candidate_nodes:
            # Degree view is computed lazily for the given node subset.
            degree_view = self.graph.degree(candidate_nodes)
            leaves_to_remove = [
                node for node, degree in degree_view if degree <= 1
            ]
            if not leaves_to_remove:
                break
            self.graph.remove_nodes_from(leaves_to_remove)
            candidate_nodes.difference_update(leaves_to_remove)
            iter += 1
            if iter_cutoff and iter >= iter_cutoff:
                break

        # update communities
        self.communities = [c for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]
        self.node_ids.clear()
        self.node_ids.update(self.graph.nodes())
        print("Pruned graph successfully")
        print("Number of nodes:", len(self.graph.nodes()))
        print("Number of edges:", len(self.graph.edges()))
        self._reset_embedding_cache()

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

    def get_graph_adj_lapl_matrices(self, subgraph: nx.Graph) -> Tuple[
        csr_matrix, csr_matrix
    ]:
        adj_mat = nx.to_scipy_sparse_array(
            subgraph, nodelist=subgraph.nodes, dtype=float
        ).tocsr()
        lapl_mat = scipy.sparse.csgraph.laplacian(adj_mat, normed=True)

        return adj_mat, lapl_mat

    def get_spectral_embeddings(self, subgraph: nx.Graph, n_components: int, mode: str = "neighbor",
                                diffuse: bool = False) -> np.ndarray:

        if len(subgraph) <= 5:
            return np.zeros((len(subgraph), n_components), dtype=np.float32)

        adj_mat, lapl_mat = self.get_graph_adj_lapl_matrices(subgraph)

        n_components = min(n_components, lapl_mat.shape[0] - 1)

        embeddings: np.ndarray

        if mode == "neighbor":
            spectral_embedder = SpectralEmbedding(
                n_components=n_components,
                affinity="precomputed",
                random_state=42,
                n_jobs=-1,  # uses all available processors
            )
            if diffuse:
                affinity = scipy.linalg.expm(adj_mat.toarray())
                embeddings = spectral_embedder.fit_transform(affinity)
            else:
                adj_mat = to_int32_sparse(adj_mat)
                embeddings = spectral_embedder.fit_transform(adj_mat)
        if mode == "laplacian":
            if len(subgraph) >= 50000:
                lapl_gpu = cupyx.scipy.sparse.csr_matrix(lapl_mat)
                vals, vecs = cu_linalg.eigsh(lapl_gpu, k=n_components + 1, which="SA", tol=1e-6, maxiter=1000)
                vals = cp.asnumpy(vals)
                vecs = cp.asnumpy(vecs)
            else:
                vals, vecs = scipy.sparse.linalg.eigsh(lapl_mat, k=n_components + 1, which='SM', tol=1e-6, maxiter=1000)

            # Drop the trivial eigenvector
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            embeddings = vecs[:, 1:n_components + 1]

        return embeddings

    def compute_node_embeddings(self, n_components: int) -> None:
        print("Building graph node embeddings...")
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        node_order = list(self.graph.nodes())
        if not node_order:
            raise ValueError("Graph contains no nodes.")

        embedding_matrix = np.zeros(
            (len(node_order), n_components), dtype=np.float32
        )
        node_id_to_row = {node_id: idx for idx, node_id in enumerate(node_order)}

        for community in tqdm.tqdm(self.communities):
            component_size = len(community)
            if component_size <= 3:
                continue
            embed_dims = min(n_components, component_size)

            subgraph = self.graph.subgraph(community)
            component_node_order = list(subgraph.nodes())

            if component_size == 1:
                component_embeddings = np.zeros(
                    (1, n_components), dtype=np.float32
                )
            else:

                component_partial = self.get_spectral_embeddings(
                    subgraph=subgraph, n_components=embed_dims, mode="laplacian"
                )
                component_embeddings = np.zeros(
                    (component_size, n_components), dtype=np.float32
                )
                component_embeddings[:, :embed_dims] = component_partial

            for node_id, embedding in zip(
                    component_node_order, component_embeddings
            ):
                row_idx = node_id_to_row[node_id]
                embedding_matrix[row_idx] = embedding

        self.node_embeddings = embedding_matrix
        self.node_id_to_embedding_row = node_id_to_row

    def get_node_embeddings(
            self, node_ids: Union[int, Iterable[int]]
    ) -> np.ndarray:
        if self.node_embeddings is None:
            raise ValueError("Call build_graph_node_embeddings before querying.")

        if isinstance(node_ids, int):
            requested_ids = [node_ids]
            return_single = True
        else:
            requested_ids = list(node_ids)
            return_single = False

        feature_dim = self.node_embeddings.shape[1]
        result = np.zeros((len(requested_ids), feature_dim), dtype=self.node_embeddings.dtype)

        for idx, node_id in enumerate(requested_ids):
            row_idx = self.node_id_to_embedding_row.get(node_id)
            if row_idx is None:
                continue
            result[idx] = self.node_embeddings[row_idx]

        if return_single:
            return result[0]
        return result

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

    def plot_subgraph_spectral_structure(self, community_idx: int, mode="laplacian", diffuse: bool = False):
        subgraph = self.get_subgraph_view(community_idx)
        if subgraph.number_of_nodes() == 0:
            return

        pos = graphviz_layout(subgraph, prog="sfdp")
        embeddings = np.asarray(
            self.get_spectral_embeddings(subgraph=subgraph, n_components=3, mode=mode, diffuse=diffuse)
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

    def plot_community_sizes(self):
        connected_components_sizes = [len(c) for c in self.communities]
        sizes = np.array(connected_components_sizes)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: histogram
        axes[0].hist(np.log10(sizes), bins=30, log=True)
        axes[0].set_title("Connected Components Sizes")
        axes[0].set_xlabel("Size [log10]")
        axes[0].set_ylabel("Frequency")

        # Right: log–log plot
        axes[1].loglog(sizes)
        axes[1].set_title("Connected Components Sizes [log]")
        axes[1].set_xlabel("Component size")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
