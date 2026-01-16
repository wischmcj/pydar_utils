"""Python Utilities for TLS LiDAR Scans of Trees.

A comprehensive toolkit for processing, analyzing, and visualizing terrestrial laser scanning (TLS)
data of trees, including point cloud processing, mesh reconstruction, skeletonization, and QSM generation.
"""

__version__ = "0.1.0"
import os 
from .set_config import load_config
# Load unconfigured logger 
import logging
log = logging.getLogger('initialize')

# Get dir name for troubleshooting
cwd = os.getcwd()
print(f"Current working directory: {cwd}")
# Read in environment variables, set defaults if not present
package_location = os.path.dirname(__file__)
print(f"Package Location: {package_location}")

# get config file locations
config_file = os.environ.get("PDAR_CONFIG", f"{package_location}/package_config.toml")
log_config_file = os.environ.get("PDAR_LOG_CONFIG", f"{package_location}/log.yml")

package_location = os.path.dirname(__file__)
log_config_file = os.environ.get("PDAR_LOG_CONFIG", f"{package_location}/log.yml")

# load log config
log_config = load_config(log_config_file, load_to_env=False)

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    log.error(f"Error loading log config {log_config_file}: {e}")
    log.error(f"Default values will be used")

# load package config
load_config(log_config_file, load_to_env=False)


# Import submodules to make them available as pydar_utils.geometry, etc.
from . import geometry
from . import math_utils
from . import utils
from . import viz

# Import commonly used functions at package level for convenience
from .geometry import zoom_pcd
from .math_utils import z_align_and_fit, get_center, rotation_matrix_from_arr
from .utils import save, load, convert_las, to_o3d
from .viz import draw, plot_3d
from .processing import clean_cloud, crop_and_highlight

__all__ = [
    # Submodules
    "geometry",
    "math_utils",
    "utils",
    "viz",
    # Common functions
    "zoom_pcd",
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
    "clean_cloud",
    "crop_and_highlight",
]
