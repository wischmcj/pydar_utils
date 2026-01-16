"""Visualization utilities for point clouds, meshes, and plotting."""

__version__ = "0.1.0"

from .color import (
    homog_colors,
    remove_color_pts,
    get_green_surfaces,
    mute_colors,
    bin_colors,
    color_compare,
    saturate_colors,
    segment_hues,
    get_color_by_hue,
    isolate_color,
    color_distribution,
    split_on_percentile,
)
from .plotting import plot_dist_dist, plot_3d, histogram, plot_neighbor_distribution
from .ray_casting import (
    get_points_inside_mesh,
    project_pcd,
    sparse_cast_w_intersections,
    birdseye,
    project_to_image,
    mri,
    cast_rays,
    raycast_to_pcd,
    get_intersected,
)
from .viz_utils import (
    draw_view,
    iter_draw,
    draw,
    color_continuous_map,
    rotating_compare_gif,
)

__all__ = [
    # color
    "homog_colors",
    "remove_color_pts",
    "get_green_surfaces",
    "mute_colors",
    "bin_colors",
    "color_compare",
    "saturate_colors",
    "segment_hues",
    "get_color_by_hue",
    "isolate_color",
    "color_distribution",
    "split_on_percentile",
    # plotting
    "plot_dist_dist",
    "plot_3d",
    "histogram",
    "plot_neighbor_distribution",
    # ray_casting
    "get_points_inside_mesh",
    "project_pcd",
    "sparse_cast_w_intersections",
    "birdseye",
    "project_to_image",
    "mri",
    "cast_rays",
    "raycast_to_pcd",
    "get_intersected",
    # viz_utils
    "draw_view",
    "iter_draw",
    "draw",
    "color_continuous_map",
    "rotating_compare_gif",
]
