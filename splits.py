import random
import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
from tqdm import tqdm
from rtree import index
from shapely.geometry import LineString, Polygon, Point
from shapely import box
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shapely

def split_graph_by_comm_detec(G, resolution):
    def get_edge_graph(G):
        edge_graph = nx.Graph()
        edge_nodes = {}

        for u, v, key, data in G.edges(keys=True, data=True):
            edge_id = (u, v, key)
            edge_nodes[edge_id] = data.copy()
            edge_nodes[edge_id]['original_nodes'] = (u, v)
            edge_nodes[edge_id]['original_edge_key'] = key

            if 'geometry' not in data:
                if 'x' in G.nodes[u] and 'y' in G.nodes[u] and 'x' in G.nodes[v] and 'y' in G.nodes[v]:
                    line = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), 
                                      (G.nodes[v]['x'], G.nodes[v]['y'])])
                    edge_nodes[edge_id]['geometry'] = line

            edge_graph.add_node(edge_id, **edge_nodes[edge_id])
        
        vertex_to_edges = {}

        for edge_id in edge_nodes:
            u, v, key = edge_id
            if u not in vertex_to_edges:
                vertex_to_edges[u] = []
            if v not in vertex_to_edges:
                vertex_to_edges[v] = []
            vertex_to_edges[u].append(edge_id)
            vertex_to_edges[v].append(edge_id)

        for vertex, edges in vertex_to_edges.items():
            if len(edges) >= 2:
                for edge1, edge2 in combinations(edges, 2):
                    edge_graph.add_edge(edge1, edge2, connecting_vertex=vertex)
        return edge_graph

    print("Creating edge graph...")
    edge_graph = get_edge_graph(G)
    print("Creating SLA's (small local areas)...")
    comms = nx.community.louvain_communities(edge_graph, seed=42, resolution=resolution)
    orig_subgraphs = {}

    for i, community in tqdm(enumerate(comms), total=len(comms), desc="Assigning edge subgraphs to original graph..."):
        subgraph = nx.MultiDiGraph()
        subgraph.graph["crs"] = G.graph["crs"]
        lines_for_hull = []
        for edge_id in community:
            u, v, key = edge_id
            original_data = edge_graph.nodes[edge_id]

            if not subgraph.has_node(u):
                if u in G.nodes:
                    subgraph.add_node(u, **G.nodes[u].copy())
                else:
                    subgraph.add_node(u)

            if not subgraph.has_node(v):
                if v in G.nodes:
                    subgraph.add_node(v, **G.nodes[v].copy())
                else:
                    subgraph.add_node(v)

            edge_data = {k: v for k, v in original_data.items() 
                        if k not in ['original_nodes', 'original_edge_key']}
            subgraph.add_edge(u, v, key=key, **edge_data)

            lines_for_hull.append(original_data['geometry'])
        
        hull_geometry = None
        if lines_for_hull:
            all_geoms = shapely.union_all(lines_for_hull)
        # hull_geometry = all_geoms.convex_hull
        bbox_geometry = shapely.box(*all_geoms.bounds)

        orig_subgraphs[i] = {
             'graph': subgraph,
            # 'polygon': hull_geometry
            'polygon': bbox_geometry
        }
    return orig_subgraphs

def split_graph_by_grid(G, grid_step=1000):

    nodes, edges = ox.graph_to_gdfs(G)
    bounds = nodes.total_bounds

    minx, miny, maxx, maxy = bounds

    x_coords = np.arange(minx, maxx, grid_step)
    y_coords = np.arange(miny, maxy, grid_step)

    if len(x_coords) == 0 or x_coords[-1] < maxx:
        x_coords = np.append(x_coords, maxx)
    if len(y_coords) == 0 or y_coords[-1] < maxy:
        y_coords = np.append(y_coords, maxy)

    subgraphs = {}

    node_cell_idx = index.Index()
    node_cell_bounds = {}

    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            cell_polygon = Polygon([
                (x_coords[i], y_coords[j]),
                (x_coords[i+1], y_coords[j]),
                (x_coords[i+1], y_coords[j+1]),
                (x_coords[i], y_coords[j+1])
            ])

            subgraph = nx.MultiDiGraph()
            subgraph.graph['crs'] = G.graph['crs']

            subgraphs[(i, j)] = {
                'graph': subgraph,
                'polygon': cell_polygon
            }

    for i, (key, cell_data) in enumerate(subgraphs.items()):
        bounds = cell_data['polygon'].bounds
        node_cell_idx.insert(i, bounds)
        node_cell_bounds[i] = (key, cell_data)

    node_to_cell = {}

    for node, data in tqdm(G.nodes(data=True), desc="Assigning nodes to cells"):
        point = Point(data['x'], data['y'])

        potential_cells = list(node_cell_idx.intersection((data['x'], data['y'], data['x'], data['y'])))

        assigned_to_cell = False

        for cell_idx in potential_cells:
            key, cell_data = node_cell_bounds[cell_idx]
            if cell_data['polygon'].contains(point):
                cell_data['graph'].add_node(node, **data)
                node_to_cell[node] = key
                assigned_to_cell = True
                break

        if not assigned_to_cell:
            for cell_idx in potential_cells:
                key, cell_data = node_cell_bounds[cell_idx]
                if cell_data['polygon'].intersects(point):
                    cell_data['graph'].add_node(node, **data)
                    node_to_cell[node] = key
                    break

    edge_cell_idx = index.Index()
    edge_cell_data_by_idx = {}

    for i, (key, cell_data) in enumerate(subgraphs.items()):
        bounds = cell_data['polygon'].bounds
        edge_cell_idx.insert(i, bounds)
        edge_cell_data_by_idx[i] = (key, cell_data)

    edges_with_geometry = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            line = data['geometry']
        else:
            u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
            v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
            line = LineString([(u_x, u_y), (v_x, v_y)])
        edges_with_geometry.append((u, v, key, data, line))

    for u, v, key, data, line in tqdm(edges_with_geometry, desc="Assigning edges to cells"):
        u_cell = node_to_cell.get(u)
        v_cell = node_to_cell.get(v)

        line_bounds = line.bounds
        potential_cell_indices = list(edge_cell_idx.intersection(line_bounds))

        for cell_index in potential_cell_indices:
            cell_key, cell_data = edge_cell_data_by_idx[cell_index]
            cell_polygon = cell_data['polygon']

            cell_bounds = cell_polygon.bounds
            if (line_bounds[0] > cell_bounds[2] or line_bounds[2] < cell_bounds[0] or
                line_bounds[1] > cell_bounds[3] or line_bounds[3] < cell_bounds[1]):
                continue

            if not line.intersects(cell_polygon):
                continue

            subgraph = cell_data['graph']
            intersection = line.intersection(cell_polygon)

            if intersection.is_empty:
                continue

            if intersection.geom_type == 'LineString':
                segments = [intersection]
            elif intersection.geom_type == 'MultiLineString':
                segments = list(intersection.geoms)
            else:
                continue

            for segment in segments:
                start_coord = segment.coords[0]
                end_coord = segment.coords[-1]

                start_node_exists = u in subgraph.nodes() if cell_key == u_cell else False
                end_node_exists = v in subgraph.nodes() if cell_key == v_cell else False

                if start_node_exists and end_node_exists:
                    start_node_id = u
                    end_node_id = v
                elif start_node_exists:
                    start_node_id = u
                    end_node_id = f"end_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    subgraph.add_node(end_node_id, x=end_coord[0], y=end_coord[1])
                elif end_node_exists:
                    start_node_id = f"start_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    end_node_id = v
                    subgraph.add_node(start_node_id, x=start_coord[0], y=start_coord[1])
                else:
                    start_node_id = f"start_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    end_node_id = f"end_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    subgraph.add_node(start_node_id, x=start_coord[0], y=start_coord[1])
                    subgraph.add_node(end_node_id, x=end_coord[0], y=end_coord[1])

                edge_data = data.copy()
                edge_data['geometry'] = segment
                edge_data['original_nodes'] = (u, v)
                edge_data['segment_id'] = f"{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                edge_data['original_edge_key'] = key

                subgraph.add_edge(start_node_id, end_node_id, **edge_data)

    result = {}
    for key, cell_data in subgraphs.items():
        if len(cell_data['graph'].nodes) > 0:
            result[key] = {
                'graph': cell_data['graph'],
                'polygon': cell_data['polygon']
            }

    return result

def split_graph(G, grid_step=None, resolution=None):
    if grid_step:
        subgraphs = split_graph_by_grid(G, grid_step=grid_step)
    else:
        subgraphs = split_graph_by_comm_detec(G, resolution)
    return subgraphs
