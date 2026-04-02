from loguru import logger
import numpy as np
import shapely
import torch
import warnings
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from block_graph import get_roads_from_block_data
from model import DynamicModel, possible_models

TQDM_DISABLE = not __import__("sys").stderr.isatty()


def _log(message: str) -> None:
    logger.info(f"[street-pattern] {message}")


def _build_road_context(block_data):
    roads = get_roads_from_block_data(block_data)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy().reset_index(drop=True)
    if roads.empty:
        return roads, {}, None

    node_degree = {}
    endpoint_rows = []
    for _, row in roads.iterrows():
        line = row.geometry
        start = tuple(line.coords[0])
        end = tuple(line.coords[-1])
        node_degree[start] = node_degree.get(start, 0) + 1
        node_degree[end] = node_degree.get(end, 0) + 1
        endpoint_rows.append({"point": Point(start), "coord": start})
        endpoint_rows.append({"point": Point(end), "coord": end})

    endpoints_gdf = None
    if endpoint_rows:
        import geopandas as gpd

        endpoints_gdf = gpd.GeoDataFrame(endpoint_rows, geometry="point", crs=roads.crs)

    return roads, node_degree, endpoints_gdf


def _main_dir(polygon):
    if polygon.geom_type == "MultiPolygon":
        coords = list(list(polygon.geoms)[0].exterior.coords)
    else:
        coords = list(polygon.exterior.coords)

    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    coords_np = np.asarray(coords, dtype=float)
    centroid = np.mean(coords_np, axis=0)
    centered_coords = coords_np - centroid
    if len(coords_np) < 3:
        return 0.0

    cov_matrix = np.cov(centered_coords, rowvar=False)
    if np.ndim(cov_matrix) < 2 or np.isnan(cov_matrix).any():
        return 0.0
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    angle_rad = np.arctan2(main_eigenvector[1], main_eigenvector[0])
    return np.degrees(angle_rad) % 180


def _acute_angle(angle1, angle2):
    diff = abs(angle1 - angle2) % 180
    return min(diff, 180 - diff)


def _safe_polygon_metrics(polygon):
    if polygon is None or polygon.is_empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    area = float(polygon.area)
    b_box_minx, b_box_miny, b_box_maxx, b_box_maxy = polygon.bounds
    b_box_width = b_box_maxx - b_box_minx
    b_box_height = b_box_maxy - b_box_miny
    convex_hull_area = float(polygon.convex_hull.area)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        circle_area = float(shapely.minimum_bounding_circle(polygon).area)
        min_rot_rec_area = float(polygon.minimum_rotated_rectangle.area)

    if not np.isfinite(circle_area):
        circle_area = 0.0
    if not np.isfinite(min_rot_rec_area):
        min_rot_rec_area = 0.0

    return area, b_box_width, b_box_height, convex_hull_area, circle_area, min_rot_rec_area


def compute_features(block_graph, block_data):
    roads, node_degree, endpoints_gdf = _build_road_context(block_data)
    road_sindex = roads.sindex if not roads.empty else None

    for node in block_graph.nodes:
        polygon = block_graph.nodes[node]["geometry"]
        area, b_box_width, b_box_height, convex_hull_area, circle_area, min_rot_rec_area = (
            _safe_polygon_metrics(polygon)
        )

        number_of_linestrings = 0
        if endpoints_gdf is not None and len(endpoints_gdf) > 0:
            endpoint_idx = list(endpoints_gdf.sindex.intersection(polygon.bounds))
            if endpoint_idx:
                endpoint_matches = endpoints_gdf.iloc[endpoint_idx]
                for _, endpoint_row in endpoint_matches.iterrows():
                    if polygon.intersects(endpoint_row["point"]):
                        if node_degree.get(endpoint_row["coord"], 0) == 1:
                            number_of_linestrings += 1

        circuity_values = []
        if road_sindex is not None:
            road_idx = list(road_sindex.intersection(polygon.bounds))
            if road_idx:
                road_matches = roads.iloc[road_idx]
                for _, road_row in road_matches.iterrows():
                    line = road_row.geometry
                    if not line.intersects(polygon):
                        continue
                    start_point = Point(line.coords[0])
                    end_point = Point(line.coords[-1])
                    straight_line_length = LineString([start_point, end_point]).length
                    if straight_line_length > 0:
                        circuity_values.append(line.length / straight_line_length)

        block_graph.nodes[node]["number_of_linestrings"] = number_of_linestrings
        block_graph.nodes[node]["area"] = area / 10000
        block_graph.nodes[node]["circuity"] = (
            float(np.mean(circuity_values)) if circuity_values else 0.0
        )
        block_graph.nodes[node]["formfacter"] = area / circle_area if circle_area > 0 else 0
        block_graph.nodes[node]["elongation"] = b_box_height / b_box_width if b_box_width > 0 else 0
        block_graph.nodes[node]["degree"] = block_graph.degree[node]
        block_graph.nodes[node]["concavity"] = area / convex_hull_area if convex_hull_area > 0 else 0
        block_graph.nodes[node]["rectanglarity"] = (
            area / min_rot_rec_area if min_rot_rec_area > 0 else 0
        )

    for edge in block_graph.edges:
        node_a = edge[0]
        node_b = edge[1]
        block_a = block_graph.nodes[node_a]["geometry"]
        block_b = block_graph.nodes[node_b]["geometry"]
        centroid_a = block_a.centroid
        centroid_b = block_b.centroid

        dx = centroid_b.x - centroid_a.x
        dy = centroid_b.y - centroid_a.y

        block_a_translated = translate(block_a, xoff=dx, yoff=dy)
        area_a = block_a_translated.area
        area_b = block_b.area

        denominator = area_a + area_b
        if denominator > 0:
            symmetric_diff_area = block_a_translated.symmetric_difference(block_b).area
            r_sim = 1 - (symmetric_diff_area / denominator)
        else:
            r_sim = 0.0
        block_graph.edges[edge]["shape_similarity"] = max(0.0, min(1.0, r_sim))

        para_a = _main_dir(block_a)
        para_b = _main_dir(block_b)
        arrangement_degree = _acute_angle(para_a, para_b)
        block_graph.edges[edge]["arrangement_degree"] = min(
            arrangement_degree, 90 - arrangement_degree
        )

    return block_graph


def simple_collate_fn(batch):
    batched_data = {}

    graph_data_list = [item["gnn0"] for item in batch]
    batched_data["gnn0"] = Batch.from_data_list(graph_data_list)
    batched_data["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return batched_data


def classify_blocks(dataset, model_path="best_model.pth", device="cuda"):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=simple_collate_fn)

    config = {"gnn0": possible_models["gnn0"], "label": possible_models["label"]}

    device = torch.device(device)
    model = DynamicModel(config, num_classes=6).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        _log("Model loaded")
    except Exception:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        _log("Model loaded partially")

    model.eval()
    predictions = {}
    probabilities = {}

    with torch.no_grad():
        for batch_idx, data in tqdm(
            enumerate(loader),
            desc="Processing batches",
            total=len(loader),
            disable=TQDM_DISABLE,
            leave=False,
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            data["gnn0"].x = data["gnn0"].x.to(device)
            data["gnn0"].edge_index = data["gnn0"].edge_index.to(device)
            data["gnn0"].edge_attr = data["gnn0"].edge_attr.to(device)
            data["gnn0"].batch = data["gnn0"].batch.to(device)

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
