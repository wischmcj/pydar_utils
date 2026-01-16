import numpy as np
import os 
from logging import getLogger

from math_utils import get_percentile, get_center, get_radius
import open3d as o3d
from viz import draw
import matplotlib.pyplot as plt

log = getLogger(__name__)

PDAR_TRUNK_LOWER_PCTILE = float(os.environ.get("PDAR_TRUNK_LOWER_PCTILE", 3))
PDAR_TRUNK_UPPER_PCTILE = float(os.environ.get("PDAR_TRUNK_UPPER_PCTILE", 10))
PDAR_INITIAL_CLEAN_VOXEL_SIZE = float(os.environ.get("PDAR_INITIAL_CLEAN_VOXEL_SIZE", 0.04))
PDAR_INITIAL_CLEAN_NEIGHBORS = int(os.environ.get("PDAR_INITIAL_CLEAN_NEIGHBORS", 2))
PDAR_INITIAL_CLEAN_RATIO = float(os.environ.get("PDAR_INITIAL_CLEAN_RATIO", 4))
PDAR_INITIAL_CLEAN_ITERS = int(os.environ.get("PDAR_INITIAL_CLEAN_ITERS", 3))

def crop_by_percentile(pcd, 
                  start = PDAR_TRUNK_LOWER_PCTILE,
                  end = PDAR_TRUNK_UPPER_PCTILE,
                  axis = 2,
                  invert = False):
    algo_source_pcd = pcd  
    algo_pcd_pts = np.asarray(algo_source_pcd.points)
    log.info(f"Getting points between the {start} and {end} percentiles")
    not_too_low_idxs, _ = get_percentile(algo_pcd_pts,start,end, axis,invert)
    low_cloud = algo_source_pcd.select_by_index(not_too_low_idxs)
    return low_cloud, not_too_low_idxs


def crop_and_highlight(pcd,lower,upper,axis):
    cropped_pcd,cropped_idxs = crop_by_percentile(pcd,lower,upper,axis)
    print(f'selecting from branch_grp')
    removed = pcd.select_by_index(cropped_idxs,invert=True)
    removed.paint_uniform_color([1,0,0])
    print(f'drawing removed')
    return cropped_pcd, cropped_idxs


def clean_cloud(pcd,
                voxels=PDAR_INITIAL_CLEAN_VOXEL_SIZE,
                neighbors=PDAR_INITIAL_CLEAN_NEIGHBORS,
                ratio=PDAR_INITIAL_CLEAN_RATIO,
                iters = PDAR_INITIAL_CLEAN_ITERS
                ):
    """Reduces the number of points in the point cloud via
    voxel downsampling. Reducing noise via statistical outlier removal.
    """
    run_voxels = voxels
    run_stat = all([neighbors, ratio, iters])
    voxel_down_pcd = pcd

    if run_voxels:
        log.info("Downsample the point cloud with voxels")
        log.info(f"orig {pcd}")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxels)
        log.info(f"downed {voxel_down_pcd}")
    if run_stat:
        log.info("Statistical oulier removal")
        for i in range(iters):
            _, ind = voxel_down_pcd.remove_statistical_outlier(    nb_neighbors=int(neighbors), std_ratio=ratio)
            voxel_down_pcd = voxel_down_pcd.select_by_index(ind)
            neighbors = neighbors * 2
            ratio = ratio / 1.5
        final = voxel_down_pcd
    else:
        final = pcd
    if not run_voxels and not run_stat:
        log.warning("No cleaning steps were run")        
    return final


def find_neighbors_in_ball(
    base_pts, points_to_search, 
    points_idxs, radius=None, center=None,
    use_top = None, #(85,100)
    draw_results = False,
    radius_multiplier = 1,
    min_radius = 0.01,
    max_radius = 1.5,
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
            radius = get_radius(top_pts) * radius_multiplier
    else:
        if not center:
            center = get_center(base_pts)
        if not radius:
            radius = get_radius(base_pts) * radius_multiplier

    if radius < min_radius:
        radius = min_radius
    if radius > max_radius:
        radius = max_radius
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
