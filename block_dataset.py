from concurrent.futures import ProcessPoolExecutor

from loguru import logger
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from block_graph import create_street_block_graph, extract_street_blocks, get_roads_from_block_data
from classification import compute_features

TQDM_DISABLE = not __import__("sys").stderr.isatty()


def _log(message: str) -> None:
    logger.bind(tag="[street-pattern]").debug(message)


def _prepare_subgraph_item(item):
    cell_id, block_data = item
    roads = get_roads_from_block_data(block_data)
    polygon = block_data["polygon"]
    crs = roads.crs if roads is not None else block_data["graph"].graph["crs"]

    gdf_blocks = extract_street_blocks(block_data)
    if gdf_blocks.empty:
        return None

    block_graph, _ = create_street_block_graph(gdf_blocks, crs)
    if block_graph.number_of_nodes() == 0:
        return None

    block_graph_with_features = compute_features(block_graph, block_data)
    return {
        "cell_id": cell_id,
        "block_data": block_data,
        "graph_with_features": block_graph_with_features,
        "polygon": polygon,
    }


class BlockDataset(Dataset):
    def __init__(self, block_graphs_dict, workers: int = 1):
        self.blocks = []
        self.cell_ids = []
        self.all_block_graphs_with_features = []
        _log("Computing features for all subgraphs...")
        items = list(block_graphs_dict.items())
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(_prepare_subgraph_item, items)
                for result in tqdm(
                    results,
                    total=len(items),
                    desc="Processing subgraphs",
                    disable=TQDM_DISABLE,
                    leave=False,
                    ascii=True,
                    dynamic_ncols=True,
                    mininterval=0.5,
                ):
                    if result is None:
                        continue
                    self.all_block_graphs_with_features.append(
                        {
                            "cell_id": result["cell_id"],
                            "graph_with_features": result["graph_with_features"],
                            "polygon": result["polygon"],
                        }
                    )
                    self.blocks.append(result["block_data"])
                    self.cell_ids.append(result["cell_id"])
        else:
            for item in tqdm(
                items,
                desc="Processing subgraphs",
                disable=TQDM_DISABLE,
                leave=False,
                ascii=True,
                dynamic_ncols=True,
                mininterval=0.5,
            ):
                cell_id, _ = item
                try:
                    result = _prepare_subgraph_item(item)
                    if result is None:
                        continue
                    self.all_block_graphs_with_features.append(
                        {
                            "cell_id": result["cell_id"],
                            "graph_with_features": result["graph_with_features"],
                            "polygon": result["polygon"],
                        }
                    )
                    self.blocks.append(result["block_data"])
                    self.cell_ids.append(result["cell_id"])
                except Exception as exc:
                    logger.bind(tag="[street-pattern]").warning(f"Skipping subgraph {cell_id}: {exc}")
                    continue

        _log("Normalization...")
        self._fit_normalization()
        _log(f"Prepared {len(self.blocks)} blocks with valid features")

    def _fit_normalization(self):
        all_node_features = []

        for item in self.all_block_graphs_with_features:
            graph_with_features = item["graph_with_features"]
            for node in graph_with_features.nodes():
                node_data = graph_with_features.nodes[node]
                all_node_features.append(self._extract_node_features_from_data(node_data))

        if not all_node_features:
            self.scaler_min_max = MinMaxScaler()
            self.scaler_standard = StandardScaler()
            self._has_scaler_fit = False
            return

        all_node_features_np = np.array(all_node_features, dtype=np.float32)
        _log(f"Number of nodes: {len(all_node_features_np)}")
        _log(f"Number of features: {all_node_features_np.shape[1]}")

        self.scaler_min_max = MinMaxScaler()
        self.scaler_standard = StandardScaler()
        all_node_features_normalized = self.scaler_min_max.fit_transform(all_node_features_np)
        self.scaler_standard.fit(all_node_features_normalized)
        self._has_scaler_fit = True

    def _extract_node_features_from_data(self, node_data):
        return [
            node_data.get("number_of_linestrings", 0),
            node_data.get("area", 0),
            node_data.get("circuity", 0),
            node_data.get("concavity", 0),
            node_data.get("rectanglarity", 0),
            node_data.get("degree", 0),
            node_data.get("formfacter", 0),
            node_data.get("elongation", 0),
        ]

    def _normalize_node_features(self, node_features_np):
        if len(node_features_np) == 0 or not getattr(self, "_has_scaler_fit", False):
            return node_features_np

        node_features_np = self.scaler_min_max.transform(node_features_np)
        node_features_np = self.scaler_standard.transform(node_features_np)
        return node_features_np

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        item = self.all_block_graphs_with_features[idx]
        graph_with_features = item["graph_with_features"]
        gnn0_data = self._graph_to_pyg_data(graph_with_features)
        return {"gnn0": gnn0_data, "label": torch.tensor(idx % 6, dtype=torch.long)}

    def _graph_to_pyg_data(self, graph):
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        node_features = []
        for node in graph.nodes():
            node_features.append(self._extract_node_features_from_data(graph.nodes[node]))

        node_features_np = np.array(node_features, dtype=np.float32)
        if len(node_features_np) > 0:
            node_features_np = self._normalize_node_features(node_features_np)

        x = torch.tensor(node_features_np, dtype=torch.float32)

        edge_index = []
        edge_attr = []
        graph_undirected = graph.to_undirected()
        for u, v, key in graph_undirected.edges(keys=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])

            edge_data = graph_undirected.get_edge_data(u, v, key) or {}
            arrangement = edge_data.get("arrangement_degree", 0.5)
            similarity = edge_data.get("shape_similarity", 0.5)
            edge_attr.extend([[arrangement, similarity], [arrangement, similarity]])

        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 2), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
