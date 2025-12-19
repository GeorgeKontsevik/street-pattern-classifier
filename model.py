import torch
import torch.nn as nn
from torchvision import models
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import (
    TransformerConv, 
    TopKPooling, 
    ChebConv, 
    global_mean_pool as gap, 
    global_max_pool as gmp
)

all_configs = []

class_names = ["Loops & Lollipops", "Irregular Grid", "Regular Grid", "Warped Parallel", "Sparse", "Broken Grid"]


# all possible model combinations
possible_models = {
    'cnn': {'type': 'image', 'columns': ['images'], 'features': 512},
    'gnn0': {
        'type': 'graph0',
        'columns': ['nx_list'],
        'input_feature_size': 8,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 2
        }
    },
    'gnn1': {
        'type': 'graph1',
        'columns': ['dual_graph_nx_list'],
        'input_feature_size': 4,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 1
        }
    },
    'gnn2': {
        'type': 'graph2',
        'columns': ['primal_graph_nx_list'],
        'input_feature_size': 5,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 1
        }
    },
    'global0': {'type': 'global', 'columns': ['global_handcrafted_features'], 'input_features': 23, 'features': 256},
    'label': {'type': 'label', 'columns': ['label0']}
}


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

combined_features_storage = []

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()


class BaseGNN(nn.Module):
    def __init__(self):
        super(BaseGNN, self).__init__()


class BaseGlobalModel(nn.Module):
    def __init__(self):
        super(BaseGlobalModel, self).__init__()


from torchvision import models


class ModifiedResNet34(nn.Module):
    def __init__(self, num_ftrs):
        super(ModifiedResNet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Identity()  # Use Identity to remove the final fully connected layer
        self.adjust_features = nn.Linear(512, num_ftrs)  # Adjust features to the desired size

    def forward(self, x):
        features = self.resnet34(x)
        adjusted_features = self.adjust_features(features)
        return adjusted_features


class CustomGNN(nn.Module):
    def __init__(self, input_feature_size, model_params):
        super(CustomGNN, self).__init__()
        # Initialize model parameters and layers as provided
        self.initialize_layers(input_feature_size, model_params)

    def initialize_layers(self, input_feature_size, model_params):
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        self.conv1 = TransformerConv(input_feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.conv2 = ChebConv(input_feature_size,
                              embedding_size,
                              3
                              )
        self.transf1 = nn.Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(ChebConv(embedding_size, embedding_size, 3))
            self.transf_layers.append(nn.Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.linear1 = nn.Linear(embedding_size * 2, dense_neurons)
        self.linear2 = nn.Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = nn.Linear(int(dense_neurons / 2), 6)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)
        global_representation = []
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            if i % self.top_k_every_n == 0 or i == self.n_layers - 1:  # Ensure it works for the last layer
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        x = sum(global_representation)
        # print(x.shape)
        # exit()
        return x


class EnhancedMLP(nn.Module):
    def __init__(self, input_features, output_features=256):
        super(EnhancedMLP, self).__init__()
        self.layer1 = nn.Linear(input_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, output_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(output_features, output_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(FusionLayer, self).__init__()
        # Calculate total combined feature size
        self.total_feature_size = sum(input_dims)
        # Layers to process combined features
        self.fusion = nn.Sequential(
            nn.Linear(self.total_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, inputs):
        # Concatenate all input features
        combined_features = torch.cat(inputs, dim=1)

        # Store the combined features for analysis
        global combined_features_storage  # Declare as global if used outside the class
        # combined_features_storage = combined_features.detach().cpu().numpy()
        # print("Appending features with shape:", combined_features.shape)

        combined_features_storage.append(combined_features.detach().cpu().numpy())

        # Map combined features to class predictions
        output = self.fusion(combined_features)

        # _, preds = output.max(1)
        # # print(preds)
        # # print(output)
        # # all_preds.extend(preds.cpu().numpy())
        # pred = []
        # pred.append(preds.cpu().numpy())
        # print(pred)
        # print(output.detach().cpu().numpy())
        # combined_features_storage.append(pred)
        return output
        # Map combined features to class predictions
        # return self.fusion(combined_features)


class DynamicModel(nn.Module):

    def __init__(self, config, num_classes):
        super(DynamicModel, self).__init__()
        self.models = nn.ModuleDict()
        self.feature_sizes = []

        if 'cnn' in config:
            self.models['cnn'] = ModifiedResNet34(num_ftrs=config['cnn']['features'])
            self.feature_sizes.append(config['cnn']['features'])

        if 'gnn0' in config:
            # print(gnn_key)
            self.models['gnn0'] = CustomGNN(input_feature_size=config['gnn0']['input_feature_size'],
                                            model_params=config['gnn0']['params'])
            self.feature_sizes.append(config['gnn0']['output_feature_size'])

        if 'gnn1' in config:
            # print(gnn_key)
            self.models['gnn1'] = CustomGNN(input_feature_size=config['gnn1']['input_feature_size'],
                                            model_params=config['gnn1']['params'])
            self.feature_sizes.append(config['gnn1']['output_feature_size'])

        if 'gnn2' in config:
            # print(gnn_key)
            self.models['gnn2'] = CustomGNN(input_feature_size=config['gnn2']['input_feature_size'],
                                            model_params=config['gnn2']['params'])
            self.feature_sizes.append(config['gnn2']['output_feature_size'])

        if 'global' in config:
            self.models['global'] = EnhancedMLP(input_features=config['global']['input_features'],
                                                output_features=config['global']['features'])
            self.feature_sizes.append(config['global']['features'])

        if 'global0' in config:
            self.models['global0'] = EnhancedMLP(input_features=config['global0']['input_features'],
                                                 output_features=config['global0']['features'])
            self.feature_sizes.append(config['global0']['features'])

        # print(self.feature_sizes)
        self.fusion_layer = FusionLayer(self.feature_sizes, num_classes)

    def get_cnn_parameters(self):
        # Retrieve parameters from all CNN components
        for name, module in self.models.items():
            if 'cnn' in name:
                for param in module.parameters():
                    yield param

    def get_gnn0_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn0' in name:
                for param in module.parameters():
                    yield param

    def get_gnn1_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn1' in name:
                for param in module.parameters():
                    yield param

    def get_gnn2_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn2' in name:
                for param in module.parameters():
                    yield param

    def forward(self, inputs):
        # print(inputs)
        outputs = []
        for key, model in self.models.items():
            base_key = key.split('_')[0]
            for input_key in inputs:
                # print(input_key)
                if base_key in input_key:  # Match base key with input keys
                    input_data = inputs[input_key]
                    if 'gnn0' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)
                    elif 'gnn1' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)

                    elif 'gnn2' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)
                    else:
                        # For CNN and MLP, pass the data directly
                        output = model(input_data)
                    outputs.append(output)
                    # print(f"Matched {key} with {input_key}")
                    break

        # return output
        return self.fusion_layer(outputs)