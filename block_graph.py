import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

MIN_BLOCK_AREA = 1e-6


def _clean_block_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return None

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    if geometry.is_empty:
        return None

    if geometry.geom_type == "MultiPolygon":
        polygons = [poly for poly in geometry.geoms if not poly.is_empty and poly.area > MIN_BLOCK_AREA]
        if not polygons:
            return None
        geometry = max(polygons, key=lambda poly: poly.area)

    if geometry.geom_type != "Polygon" or geometry.area <= MIN_BLOCK_AREA:
        return None

    return geometry


def get_roads_from_block_data(block_data):
    roads = block_data.get("roads")
    if roads is not None:
        return roads

    graph = block_data.get("graph")
    if graph is None:
        raise ValueError("block_data must contain either 'roads' or 'graph'.")

    _, roads = ox.graph_to_gdfs(graph)
    return roads.reset_index(drop=True)


def extract_street_blocks(block_data, road_buffer: float = 0.0001):
    roads = get_roads_from_block_data(block_data)
    polygon = block_data["polygon"]
    crs = roads.crs if roads is not None else block_data["graph"].graph["crs"]

    if roads.empty:
        return gpd.GeoDataFrame({"block_id": [], "geometry": []}, geometry="geometry", crs=crs)

    lines = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    if lines.empty:
        return gpd.GeoDataFrame({"block_id": [], "geometry": []}, geometry="geometry", crs=crs)

    all_lines = lines.geometry.union_all()
    lines_buffered = all_lines.buffer(road_buffer)
    result_geometry = polygon.difference(lines_buffered)

    if result_geometry.is_empty:
        polygons = []
    elif result_geometry.geom_type == "MultiPolygon":
        polygons = [poly for poly in result_geometry.geoms if not poly.is_empty]
    else:
        polygons = [result_geometry]

    polygons = [
        cleaned
        for cleaned in (_clean_block_polygon(poly.buffer(road_buffer)) for poly in polygons if poly.area > 0)
        if cleaned is not None
    ]
    return gpd.GeoDataFrame(
        {"block_id": range(len(polygons)), "geometry": polygons},
        geometry="geometry",
        crs=crs,
    )


def create_street_block_graph(gdf_blocks, crs, buffer: float = 0.0001):
    gdf_blocks = gdf_blocks[gdf_blocks.geometry.notna() & ~gdf_blocks.geometry.is_empty].copy()
    if not gdf_blocks.empty:
        gdf_blocks["geometry"] = gdf_blocks.geometry.map(_clean_block_polygon)
        gdf_blocks = gdf_blocks[gdf_blocks.geometry.notna()].copy()
    if gdf_blocks.empty:
        block_graph = nx.MultiDiGraph()
        block_graph.graph["crs"] = crs
        return block_graph, gdf_blocks

    gdf_blocks["centroid"] = gdf_blocks.geometry.centroid
    block_graph = nx.MultiDiGraph()
    block_graph.graph["crs"] = crs

    for _, row in gdf_blocks.iterrows():
        block_graph.add_node(
            row["block_id"],
            geometry=row["geometry"],
            centroid=row["centroid"],
            x=row["centroid"].x,
            y=row["centroid"].y,
        )

    spatial_index = gdf_blocks.sindex
    gdf_blocks_buffered = gdf_blocks.copy()
    gdf_blocks_buffered["geometry_buffered"] = gdf_blocks_buffered.geometry.buffer(buffer)

    for i, block_i in gdf_blocks.iterrows():
        possible_matches_index = list(
            spatial_index.intersection(gdf_blocks_buffered.loc[i, "geometry_buffered"].bounds)
        )
        possible_matches = gdf_blocks.iloc[possible_matches_index]

        geom_i_buffered = gdf_blocks_buffered.loc[i, "geometry_buffered"]
        for j, block_j in possible_matches.iterrows():
            if i >= j:
                continue

            geom_j_buffered = gdf_blocks_buffered.loc[j, "geometry_buffered"]
            if not geom_i_buffered.intersects(geom_j_buffered):
                continue

            line_geom = LineString([block_i["centroid"], block_j["centroid"]])
            block_graph.add_edge(
                block_i["block_id"],
                block_j["block_id"],
                geometry=line_geom,
                length=line_geom.length,
            )

    return block_graph, gdf_blocks
