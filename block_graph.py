import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString

def extract_street_blocks(graph_dict):
    graph = graph_dict['graph']
    polygon = graph_dict['polygon']
    nodes, lines = ox.graph_to_gdfs(graph)
    all_lines = lines.union_all()
    lines_buffered = all_lines.buffer(0.0001)
    result_geometry = polygon.difference(lines_buffered)

    if result_geometry.geom_type == 'MultiPolygon':
        polygons = list(result_geometry.geoms)
    else:
        polygons = [result_geometry]

    blocks_gdf = gpd.GeoDataFrame({
        'block_id': range(len(polygons)),
        'geometry': [poly.buffer(0.0001) for poly in polygons]
    }, crs=graph.graph['crs'])

    return blocks_gdf

def create_street_block_graph(gdf_blocks, graph, buffer: float = 0.0001):
    gdf_blocks = gdf_blocks.copy()
    gdf_blocks['centroid'] = gdf_blocks.geometry.centroid

    block_graph = nx.MultiDiGraph()

    block_graph.graph['crs'] = graph.graph['crs']

    for idx, row in gdf_blocks.iterrows():
        block_graph.add_node(
            row['block_id'],
            geometry=row['geometry'],
            centroid=row['centroid'],
            x=row['centroid'].x,
            y=row['centroid'].y
        )

    spatial_index = gdf_blocks.sindex
    gdf_blocks_buffered = gdf_blocks.copy()
    gdf_blocks_buffered['geometry_buffered'] = gdf_blocks_buffered.geometry.buffer(buffer)

    for i, block_i in gdf_blocks.iterrows():
        possible_matches_index = list(spatial_index.intersection(block_i.geometry.bounds))
        possible_matches = gdf_blocks.iloc[possible_matches_index]

        for j, block_j in possible_matches.iterrows():
            if i >= j:
                continue

            geom_i_buffered = gdf_blocks_buffered.loc[i, 'geometry_buffered']
            geom_j_buffered = gdf_blocks_buffered.loc[j, 'geometry_buffered']

            if geom_i_buffered.intersects(geom_j_buffered):
                line_geom = LineString([block_i['centroid'], block_j['centroid']])

                block_graph.add_edge(
                    block_i['block_id'],
                    block_j['block_id'],
                    geometry=line_geom,
                    length=line_geom.length
                )

    return block_graph, gdf_blocks

