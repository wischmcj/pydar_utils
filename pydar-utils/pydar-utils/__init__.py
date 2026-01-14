"""Python Utilities for TLS LiDAR Scans of Trees.

A comprehensive toolkit for processing, analyzing, and visualizing terrestrial laser scanning (TLS)
data of trees, including point cloud processing, mesh reconstruction, skeletonization, and QSM generation.
"""

__version__ = "0.1.0"

# Import submodules to make them available as pydar_utils.geometry, etc.
from . import geometry
from . import math_utils
from . import utils
from . import viz

# Import commonly used functions at package level for convenience
from .geometry import center_and_rotate, zoom_pcd, clean_cloud, extract_skeleton
from .math_utils import z_align_and_fit, get_center, rotation_matrix_from_arr
from .utils import save, load, convert_las, to_o3d
from .viz import draw, plot_3d, cluster_color

__all__ = [
    # Submodules
    "geometry",
    "math_utils",
    "utils",
    "viz",
    # Common functions
    "center_and_rotate",
    "zoom_pcd",
    "clean_cloud",
    "extract_skeleton",
    "z_align_and_fit",
    "get_center",
    "rotation_matrix_from_arr",
    "save",
    "load",
    "convert_las",
    "to_o3d",
    "draw",
    "plot_3d",
    "cluster_color",
]
