from itertools import chain
import open3d as o3d
import numpy as np
from numpy import asarray as arr
import scipy.spatial as sps
from matplotlib import pyplot as plt, patches

from set_config import config, log
from math_utils.general import get_center, get_radius, get_percentile
from viz.viz_utils import iter_draw, draw

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

def find_neighbors_in_ball(
    base_pts, points_to_search, 
    points_idxs, radius=None, center=None,
    use_top = None, #(85,100)
    draw_results = False
):
    """
        Essentially a KNN radius search but with one sphere 
            used rather than one for each pt
       Defines a centroid for a set of base points,
        finds all points within the search set falling
         within a sphere of a given radius centered on the centroid
    """
    if use_top:
        top_pt_idxs,_ = get_percentile(base_pts,use_top[0],use_top[1])
        top_pts = base_pts[top_pt_idxs]
        if not center:
            center = get_center(top_pts)
            center = [center[0], center[1], max(base_pts[:, 2])] 
        if not radius:
            radius = get_radius(top_pts) * config['sphere']["radius_multiplier"]
    else:
        if not center:
            center = get_center(base_pts)
        if not radius:
            radius = get_radius(base_pts) * config['sphere']["radius_multiplier"]

    if radius < config['sphere']["min_radius"]:
        radius = config['sphere']["min_radius"]
    if radius > config['sphere']["max_radius"]:
        radius = config['sphere']["max_radius"]
    log.info(f" Finding nbrs in ball w/ {radius=}, {center=}")

    full_tree = sps.KDTree(points_to_search)
    neighbors = full_tree.query_ball_point(center, r=radius)
    res = []
    if draw_results:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(base_pts[:,0], base_pts[:,1], base_pts[:,2], 'r')
    for results in neighbors:
        new_points = np.setdiff1d(results, points_idxs)
        res.extend(new_points)
        if draw_results:
            nearby_points = points_to_search[new_points]
            ax.plot(nearby_points[:,0], nearby_points[:,1], nearby_points[:,2], 'o')
    sphere =  o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    if draw_results:
        test = o3d.geometry.PointCloud()
        test.points = o3d.utility.Vector3dVector(base_pts)
        draw([sphere, test])
    return sphere, neighbors, center, radius
