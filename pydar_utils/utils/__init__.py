"""General utilities for I/O, algorithms, and library integration."""

__version__ = "0.0.2"

from .algo import smooth_feature
from .general import list_if, poprow
from .io import (
    save_line_set,
    load_line_set,
    update,
    save,
    load,
    get_attrs_las,
    convert_las,
    np_to_o3d,
    to_o3d,
    create_table,
)
from .lib_integration import (
    pts_to_cloud,
    get_pairs,
    get_neighbors_in_tree,
)
from .log_utils import ConsoleHandler

__all__ = [
    # algo
    "smooth_feature",
    # general
    "list_if",
    "poprow",
    # io
    "save_line_set",
    "load_line_set",
    "update",
    "save",
    "load",
    "get_attrs_las",
    "convert_las",
    "np_to_o3d",
    "to_o3d",
    "create_table",
    # lib_integration
    "pts_to_cloud",
    "get_pairs",
    "get_neighbors_in_tree",
    # log_utils
    "ConsoleHandler",
]
