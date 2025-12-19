import torch
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
import random
import networkx as nx
from block_graph import extract_street_blocks, create_street_block_graph
from classification import compute_features

class BlockDataset(Dataset):

    def __init__(self, block_graphs_dict):
        self.blocks = []
        self.cell_ids = []
        self.all_block_graphs_with_features = []
        print("Computing features for all subgraphs...")

        for cell_id, block_data in tqdm(block_graphs_dict.items(), desc="Processing subgraphs..."):
            G = block_data['graph']
            polygon = block_data['polygon']
            gdf_blocks = extract_street_blocks(block_data)
            block_graph, gdf_blocks = create_street_block_graph(gdf_blocks, G)
            block_graph_with_features = compute_features(block_graph, G)
            self.all_block_graphs_with_features.append({
                'cell_id': cell_id,
                'graph_with_features': block_graph_with_features,
                'original_graph': G,
                'polygon': polygon
            })
            self.blocks.append(block_data)
            self.cell_ids.append(cell_id)

        print("Normalization...")
        self._fit_normalization()

        print(f"Подготовлено {len(self.blocks)} блоков с корректными признаками")

    def _fit_normalization(self):
        all_node_features = []

        for item in self.all_block_graphs_with_features:
            graph_with_features = item['graph_with_features']
            for node in graph_with_features.nodes():
                node_data = graph_with_features.nodes[node]
                features = self._extract_node_features_from_data(node_data)
                all_node_features.append(features)

        all_node_features_np = np.array(all_node_features, dtype=np.float32)
        print(f"Number of nodes: {len(all_node_features_np)}")
        print(f"Number of features: {all_node_features_np.shape[1]}")
        self.scaler_min_max = MinMaxScaler()
        self.scaler_standard = StandardScaler()

        all_node_features_normalized = self.scaler_min_max.fit_transform(all_node_features_np)
        self.scaler_standard.fit(all_node_features_normalized)

    def _extract_node_features_from_data(self, node_data):
        """Извлекает узловые признаки из данных узла"""
        features = [
            node_data.get('number_of_linestrings', 0),
            node_data.get('area', 0),
            node_data.get('circuity', 0),
            node_data.get('concavity', 0),
            node_data.get('rectanglarity', 0),
            node_data.get('degree', 0),
            node_data.get('formfacter', 0),
            node_data.get('elongation', 0)
        ]
        return features

    def _normalize_node_features(self, node_features_np):
        """Применяет нормализацию к узловым признакам"""
        if len(node_features_np) == 0:
            return node_features_np

        node_features_np = self.scaler_min_max.transform(node_features_np)
        node_features_np = self.scaler_standard.transform(node_features_np)

        return node_features_np

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        item = self.all_block_graphs_with_features[idx]
        graph_with_features = item['graph_with_features']

        gnn0_data = self._graph_to_pyg_data(graph_with_features)

        data = {
            'gnn0': gnn0_data,
            # 'label': torch.tensor(random.randint(0, 6), dtype=torch.long)
            'label': torch.tensor(idx % 6, dtype=torch.long)
        }

        return data


    def _graph_to_pyg_data(self, G):
        node_mapping = {node: i for i, node in enumerate(G.nodes())}

        node_features = []
        for node in G.nodes():
            features = []
            node_data = G.nodes[node]
            features = self._extract_node_features_from_data(node_data)
            node_features.append(features)

        node_features_np = np.array(node_features, dtype=np.float32)
        if len(node_features_np) > 0:
            node_features_np = self._normalize_node_features(node_features_np)


        x = torch.tensor(node_features_np, dtype=torch.float32)

        edge_index = []
        edge_attr = []

        G_undirected = G.to_undirected()
        for u, v in G_undirected.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])

            edge_data = G.get_edge_data(u, v)
            if edge_data:
                arrangement = edge_data.get('arrangement_degree', 0.5)
                similarity = edge_data.get('shape_similarity', 0.5)
                edge_attr.extend([[arrangement, similarity], [arrangement, similarity]])
            else:
                edge_attr.extend([[0.5, 0.5], [0.5, 0.5]])


        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)