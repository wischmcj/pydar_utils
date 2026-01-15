"""Geometry utilities for point cloud and mesh processing."""

__version__ = "0.1.0"

from .mesh_processing import (
    get_ball_mesh,
    edges_to_lineset,
    check_properties,
    subdivide_mesh,
    cluster_and_remove_triangles,
    get_surface_clusters,
    map_density,
)
from .point_cloud_processing import (
    voxelize_and_trace,
    recover_from_trace,
    join_pcd_files,
    create_one_or_many_pcds,
    normalize_to_origin,
    clean_cloud,
    crop,
    crop_by_percentile,
    crop_and_highlight,
    cluster_plus,
    cluster_and_get_largest,
)
from .point_cloud_filtering import (
    zoom_pcd,
    zoom,
    filter_list_to_region,
    filter_to_region_pcds,
    filter_pcd_list,
    bounding_box,
    filter_by_bb,
    filter_by_norm,
)
from .reconstruction import (
    expand_features_to_orig,
    overlap_voxel_grid,
    get_neighbors_kdtree,
    get_nbrs_voxel_grid,
)
from .surf_recon import (
    deform_mesh,
    pytmesh_to_mesh,
    meshfix,
    pivot_ball_mesh,
    get_mesh,
    radius_search,
    knn_search,
)

__all__ = [
    # mesh_processing
    "get_ball_mesh",
    "edges_to_lineset",
    "check_properties",
    "subdivide_mesh",
    "cluster_and_remove_triangles",
    "get_surface_clusters",
    "map_density",
    # point_cloud_processing
    "voxelize_and_trace",
    "recover_from_trace",
    "join_pcd_files",
    "create_one_or_many_pcds",
    "normalize_to_origin",
    "clean_cloud",
    "crop",
    "crop_by_percentile",
    "crop_and_highlight",
    "cluster_plus",
    "cluster_and_get_largest",
    # point_cloud_filtering
    "zoom_pcd",
    "zoom",
    "filter_list_to_region",
    "filter_to_region_pcds",
    "filter_pcd_list",
    "bounding_box",
    "filter_by_bb",
    "filter_by_norm",
    # reconstruction
    "expand_features_to_orig",
    "overlap_voxel_grid",
    "get_neighbors_kdtree",
    "get_nbrs_voxel_grid",
    # surf_recon
    "deform_mesh",
    "pytmesh_to_mesh",
    "meshfix",
    "pivot_ball_mesh",
    "get_mesh",
    "radius_search",
    "knn_search",
]
