import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import networkx as nx
from shapely.geometry import LineString
from scipy import stats

def plot_subgraphs_polygons(subgraphs_dict, size=(12, 10)):
    
    num_subgraphs = len(subgraphs_dict)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_subgraphs))
    fig, ax = plt.subplots(1, 1, figsize=size)
    
    for idx, (cell_id, graph_item) in enumerate(subgraphs_dict.items()):
        if isinstance(graph_item, dict) and 'graph' in graph_item:
            G = graph_item['graph']
            polygon = graph_item.get('polygon', None)
        else:
            G = graph_item
            polygon = None
        
        color = colors[idx]
        
        try:
            if polygon is not None:
                if hasattr(polygon, 'exterior'):
                    coords = list(polygon.exterior.coords)
                elif isinstance(polygon, (list, np.ndarray)):
                    coords = polygon
                else:
                    coords = []
                
                if len(coords) > 2:
                    polygon_patch = Polygon(coords, 
                                           alpha=0.4, 
                                           facecolor=color,
                                           edgecolor=color,
                                           linewidth=1.5)
                    ax.add_patch(polygon_patch)
            
            for u, v, data in G.edges(data=True):
                if 'geometry' in data:
                    geometry = data['geometry']
                    if hasattr(geometry, 'coords'):
                        coords = list(geometry.coords)
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        ax.plot(x_coords, y_coords, color=color,
                               linewidth=1.5, alpha=0.7)
                else:
                    if u in G.nodes() and v in G.nodes():
                        if ('x' in G.nodes[u] and 'y' in G.nodes[u] and
                            'x' in G.nodes[v] and 'y' in G.nodes[v]):
                            u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                            v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
                            ax.plot([u_x, v_x], [u_y, v_y], color=color,
                                   linewidth=1.5, alpha=0.7)   
        except Exception as e:
            print(f"Error processing cell {cell_id}: {e}")
            continue
    
    ax.set_title(f'Подграфы (всего: {num_subgraphs})', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def plot_all_subgraphs(subgraphs_dict, predictions, class_names, size):
    colors = plt.cm.Paired(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(1, 1, figsize=size)
    for cell_id, graph_item in subgraphs_dict.items():
        if isinstance(graph_item, dict) and 'graph' in graph_item:
            G = graph_item['graph']
        else:
            G = graph_item

        if cell_id in predictions:
            pred_class = predictions[cell_id]
            color = colors[pred_class]

            try:
                for u, v, data in G.edges(data=True):
                    if 'geometry' in data:
                        geometry = data['geometry']
                        if isinstance(geometry, LineString):
                            x_coords, y_coords = geometry.coords.xy
                            ax.plot(x_coords, y_coords, color=color,
                                   linewidth=1, alpha=0.6)
                    else:
                        if u in G.nodes() and v in G.nodes():
                            if ('x' in G.nodes[u] and 'y' in G.nodes[u] and
                                'x' in G.nodes[v] and 'y' in G.nodes[v]):
                                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                                v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
                                ax.plot([u_x, v_x], [u_y, v_y], color=color,
                                       linewidth=1, alpha=0.6)

                for node in G.nodes():
                    if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
                        x, y = G.nodes[node]['x'], G.nodes[node]['y']
                        ax.plot(x, y, 'o', color=color, markersize=2, alpha=0.8)
            except Exception as e:
                print(f"Error processing cell {cell_id}: {e}")
                continue

    legend_elements = []
    for class_id, class_name in enumerate(class_names):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors[class_id], markersize=10, label=class_name)
        )

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title('Все подграфы с цветовой кодировкой классов', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_features_by_class(dataset, predictions, class_names, feature_names=None):
    
    if feature_names is None:
        feature_names = [
            'number_of_linestrings', 'area', 'circuity', 'concavity',
            'rectanglarity', 'degree', 'formfacter', 'elongation'
        ]
    
    features_by_class = {class_id: [] for class_id in range(len(class_names))}
    
    for idx, item in enumerate(dataset.all_block_graphs_with_features):
        cell_id = item['cell_id']
        
        if cell_id not in predictions:
            continue
            
        pred_class = predictions[cell_id]
        graph_with_features = item['graph_with_features']
        
        for node in graph_with_features.nodes():
            node_data = graph_with_features.nodes[node]
            features = dataset._extract_node_features_from_data(node_data)
            features_by_class[pred_class].append(features)
    
    for class_id in features_by_class:
        if features_by_class[class_id]:
            features_by_class[class_id] = np.array(features_by_class[class_id])
    
    n_features = len(feature_names)
    n_classes = len(class_names)
    
    fig_width = max(4 * n_classes, 12)
    fig_height = max(3 * n_features, 10)
    
    fig, axes = plt.subplots(n_features, n_classes, 
                             figsize=(fig_width, fig_height),
                             constrained_layout=True) 
    
    if n_features == 1 and n_classes == 1:
        axes = np.array([[axes]])
    elif n_features == 1:
        axes = axes.reshape(1, -1)
    elif n_classes == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for feat_idx, feat_name in enumerate(feature_names):
        for class_idx, class_name in enumerate(class_names):
            ax = axes[feat_idx, class_idx]
            
            if (class_idx in features_by_class and 
                features_by_class[class_idx] is not None and 
                len(features_by_class[class_idx]) > 0):
                
                if feat_idx < features_by_class[class_idx].shape[1]:
                    feat_values = features_by_class[class_idx][:, feat_idx]
                    
                    if len(feat_values) > 0:
                        ax.hist(feat_values, bins=30, alpha=0.7, 
                               color=colors[class_idx], edgecolor='black', density=True)
                        
                        try:
                            from scipy import stats
                            kde = stats.gaussian_kde(feat_values)
                            x_range = np.linspace(min(feat_values), max(feat_values), 100)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.8)
                        except:
                            pass
                        
                        mean_val = np.mean(feat_values)
                        ax.axvline(mean_val, color='blue', linestyle='--', 
                                  linewidth=1.5, alpha=0.8)
                        
                        stats_text = f'Mean: {mean_val:.3f}\nStd: {np.std(feat_values):.3f}\nN: {len(feat_values)}'
                        
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        
                        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.02  
                        y_pos = ylim[1] * 0.98  
                        
                        ax.text(x_pos, y_pos, stats_text, 
                               verticalalignment='top', 
                               fontsize=7,  
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.3))
                        
                        ax.grid(True, alpha=0.2, linestyle='--')
                    else:
                        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', 
                               verticalalignment='center', transform=ax.transAxes,
                               fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Invalid feature\nindex', horizontalalignment='center', 
                           verticalalignment='center', transform=ax.transAxes,
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', 
                       verticalalignment='center', transform=ax.transAxes,
                       fontsize=10)
            
            if feat_idx == 0:
                ax.set_title(f'{class_name}', fontsize=11, fontweight='bold', pad=15)
            
            if class_idx == 0:
                if len(feat_name) > 20:
                    words = feat_name.split('_')
                    if len(words) > 1:
                        lines = []
                        current_line = []
                        current_len = 0
                        
                        for word in words:
                            if current_len + len(word) + 1 <= 20: 
                                current_line.append(word)
                                current_len += len(word) + 1
                            else:
                                lines.append('_'.join(current_line))
                                current_line = [word]
                                current_len = len(word)
                        
                        if current_line:
                            lines.append('_'.join(current_line))
                        
                        ylabel = '\n'.join(lines)
                    else:
                        ylabel = '\n'.join([feat_name[i:i+20] for i in range(0, len(feat_name), 20)])
                else:
                    ylabel = feat_name
                
                ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', labelpad=10)
            
            if feat_idx == n_features - 1:
                ax.set_xlabel('Value', fontsize=9, labelpad=10)
            
            ax.margins(x=0.1, y=0.1)
    
    plt.suptitle('Distribution of Features by Class', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=3.0)
    plt.show()
    
    return features_by_class
