import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
import osmnx as ox
import shapely
from shapely.geometry import Point, LineString
from shapely.affinity import translate
from tqdm import tqdm
from model import DynamicModel, possible_models
# from block_dataset import BlockDataset



def compute_features(block_graph, graph):

    for node in block_graph.nodes:
        block_graph.nodes[node]['number_of_linestrings'] = 0
        polygon = block_graph.nodes[node]['geometry'] # к какому полигону (кварталу) относится этот узел
        area = polygon.area
        length = polygon.length

        points_coords = []
        lines_edges = []
        for node_street_graph in graph.nodes:
            point = Point(graph.nodes[node_street_graph]['x'], graph.nodes[node_street_graph]['y'])
            if polygon.intersects(point) and graph.degree[node_street_graph] == 1:
                block_graph.nodes[node]['number_of_linestrings'] += 1
            else:
                points_coords += [node_street_graph]

        for i1, p1 in enumerate(points_coords):
            for i2, p2 in enumerate(points_coords[i1+1:]):
                if graph.has_edge(p1, p2, 0):
                    g = graph.edges[p1, p2, 0]['geometry']
                    lines_edges.append(g)

        len_edges = len(lines_edges)

        circuity = 0
        for line in lines_edges:
            length = line.length
            line_coo = line.coords
            s_p = Point(line_coo[0])
            e_p = Point(line_coo[-1])
            straight_line_length = LineString([s_p, e_p]).length
            circuity += (length / straight_line_length) / len_edges if len_edges > 0 and straight_line_length > 0 else 0

        circle = shapely.minimum_bounding_circle(polygon)
        b_box_minx, b_box_miny, b_box_maxx, b_box_maxy = polygon.bounds
        b_box_width = b_box_maxx - b_box_minx
        b_box_height = b_box_maxy - b_box_miny
        convex_hull = polygon.convex_hull
        convex_hull_area = convex_hull.area
        min_rot_rec = polygon.minimum_rotated_rectangle
        min_rot_rec_area = min_rot_rec.area

        block_graph.nodes[node]['area'] = area / 10000
        block_graph.nodes[node]['circuity'] = circuity
        block_graph.nodes[node]['formfacter'] = area / circle.area if circle.area > 0 else 0
        block_graph.nodes[node]['elongation'] = b_box_height / b_box_width if b_box_width > 0 else 0
        block_graph.nodes[node]['degree'] = block_graph.degree[node]

        block_graph.nodes[node]['concavity'] = area / convex_hull_area if convex_hull_area > 0 else 0

        block_graph.nodes[node]['rectanglarity'] = area / min_rot_rec_area if min_rot_rec_area > 0 else 0

    for edge in block_graph.edges:
        node_a = edge[0]
        node_b = edge[1]
        block_a = block_graph.nodes[node_a]['geometry']
        block_b = block_graph.nodes[node_b]['geometry']
        centroid_a = block_a.centroid
        centroid_b = block_b.centroid

        dx = centroid_b.x - centroid_a.x
        dy = centroid_b.y - centroid_a.y

        block_a_translated = translate(block_a, xoff=dx, yoff=dy)
        area_a = block_a_translated.area
        area_b = block_b.area

        symmetric_diff = block_a_translated.symmetric_difference(block_b)
        symmetric_diff_area = symmetric_diff.area
        r_sim = 1 - (symmetric_diff_area / (area_a + area_b))
        r_sim = max(0.0, min(1.0, r_sim))
        block_graph.edges[edge]['shape_similarity'] = r_sim

        def main_dir(polygon):
            if polygon.geom_type == 'MultiPolygon':
              coords = list(list(polygon.geoms)[0].exterior.coords)
            else:
              coords = list(polygon.exterior.coords)

            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]

            centroid = np.mean(coords, axis=0)
            centered_coords = coords - centroid
            cov_matrix = np.cov(centered_coords, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            main_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

            angle_rad = np.arctan2(main_eigenvector[1], main_eigenvector[0])
            angle_deg = np.degrees(angle_rad) % 180
            return angle_deg

        def acute_angle(angle1, angle2):
            diff = abs(angle1 - angle2) % 180
            return min(diff, 180 - diff)

        para_a = main_dir(block_a)
        para_b = main_dir(block_b)
        arrangement_degree = acute_angle(para_a, para_b)
        block_graph.edges[edge]['arrangement_degree'] = min(arrangement_degree, 90 - arrangement_degree)

    return block_graph


def simple_collate_fn(batch):
    batched_data = {}

    graph_data_list = [item['gnn0'] for item in batch]
    batched_data['gnn0'] = Batch.from_data_list(graph_data_list)

    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    batched_data['label'] = labels

    return batched_data

def classify_blocks(dataset, model_path='best_model.pth', device='cuda'):

    # dataset = BlockDataset(block_graphs_dict)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)

    config = {
        'gnn0': possible_models['gnn0'],
        'label': possible_models['label']
    }

    device = torch.device(device)
    model = DynamicModel(config, num_classes=6).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print("Модель загружена")
    except:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        print("Модель загружена частично")

    model.eval()

    predictions = {}
    probabilities = {}

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), desc="Processing batches", total=len(loader)):
            data['gnn0'].x = data['gnn0'].x.to(device)
            data['gnn0'].edge_index = data['gnn0'].edge_index.to(device)
            data['gnn0'].edge_attr = data['gnn0'].edge_attr.to(device)
            data['gnn0'].batch = data['gnn0'].batch.to(device)

            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = output.max(1)

            batch_size = len(preds)
            for i in range(batch_size):
                global_idx = batch_idx * 32 + i
                if global_idx < len(dataset):
                    cell_id = dataset.cell_ids[global_idx]
                    predictions[cell_id] = preds[i].item()
                    probabilities[cell_id] = probs[i].cpu().numpy()

    return predictions, probabilities