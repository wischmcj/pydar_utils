from itertools import chain
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import scipy.spatial as sps
from matplotlib import pyplot as plt

import logging
log = logging.getLogger()
from math_utils.general import get_center, get_radius, get_percentile
from viz.viz_utils import draw

## Numpy

def pts_to_cloud(points:np.ndarray, colors = None):
    '''
    Convert a numpy array to an open3d point cloud. Just for convenience to avoid converting it every single time.
    Assigns blue color uniformly to the point cloud.

    :param points: Nx3 array with xyz location of points
    :return: a blue open3d.geometry.PointCloud()
    '''
    if not colors:
        colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

### KDTrees
def get_pairs(query_pts=None, kd_tree=None, radius=.2, return_line_set=False):
    """
        Gets a graph cosisting of pairs of points in kdtree that are 
            at most 'raduis' distance away from each other
    """
    if kd_tree is None:
        if query_pts is None:
            raise ValueError('must pass either query_pts or kd_tree')
    kd_tree = sps.KDTree(query_pts)
    pairs = kd_tree.query_pairs(r=radius, output_type='ndarray')
    line_set=None
    if return_line_set:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(query_pts)
        line_set.lines = o3d.utility.Vector2iVector(pairs)
        # draw(line_set)
    degrees= [0]*len(query_pts)
    # for u,v in arr(line_set.lines): 
        # degrees= [0]*len(query_pts)
    for u,v in arr(pairs): 
        degrees[u]+=1
        degrees[v]+=1
    udegrees, cnts = np.unique(degrees,return_counts=True)
    return degrees,cnts,line_set

def get_neighbors_in_tree(sub_pcd_pts, full_tree, radius):
    '''Returns indicies n_idx of points in full_tree.data such that 
        distance(full_tree.data[n_idx],q_pt)<=radius for q_pt in sub_pcd_pts'''
    trunk_tree = sps.KDTree(sub_pcd_pts)
    pairs = full_tree.query_ball_tree(trunk_tree, r=radius)
    neighbors = np.array(list(set(chain.from_iterable(pairs))))
    return neighbors
