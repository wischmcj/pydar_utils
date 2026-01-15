import copy
import numpy as np
import open3d as o3d
import scipy.cluster as spc
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
import pyransac3d as pyrsc
from matplotlib import pyplot as plt

from set_config import config, log
from .general import (get_radius, 
                    get_center, 
                    rotation_matrix_from_arr,
                    unit_vector)

def z_align_and_fit(pcd, axis_guess, **kwargs):
    """
        Attempts to fit a cylinder to a point cloud.
        Reduces dimensionality (3d->2d) by estimating the 
            axis of said cylinder, rotating the point cloud and
            fitting its 2D projection with a circle.
    """
    R_to_z = rotation_matrix_from_arr(unit_vector(axis_guess), [0, 0, 1])
    R_from_z = rotation_matrix_from_arr([0, 0, 1], unit_vector(axis_guess))
    # align with z-axis
    pcd_r = copy.deepcopy(pcd)
    pcd_r.rotate(R_to_z)
    # approx via circle
    mesh, _, inliers, fit_radius, _ = fit_shape_RANSAC(pcd=pcd_r, **kwargs)
    if mesh is None:
        log.warning('No mesh found')
        return mesh, _, inliers, fit_radius, _
    mesh_pts = mesh.sample_points_uniformly(1000)
    mesh_pts.paint_uniform_color([0, 1.0, 0])
    mesh_pts.rotate(R_from_z)
    return mesh, _, inliers, fit_radius, _

def cluster_2d(pcd,axis,eps,min_points):
    import copy
    pcd_pts = np.array( pcd.points)
    new_pcd_pts = copy.deepcopy(pcd_pts) 
    new_pcd_pts[:,axis] = np.zeros_like(pcd_pts[:,axis])
    pcd.points = o3d.utility.Vector3dVector(new_pcd_pts)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
    return labels

def orientation_from_norms(norms, samples=10, max_iter=100):
    """Attempts to find the orientation of a cylindrical point cloud
    given the normals of the points. Attempts to find <samples> number
    of vectors that are orthogonal to the normals and then averages
    the third ortogonal vector (the cylinder axis) to estimate orientation.
    """
    sum_of_vectors = [0, 0, 0]
    found = 0
    iter_num = 0
    while found < samples and iter_num < max_iter and len(norms) > 1:
        iter_num += 1
        rand_id = np.random.randint(len(norms) - 1)
        norms, vect = poprow(norms, rand_id)
        dot_products = abs(np.dot(norms, vect))
        most_normal_val = min(dot_products)
        if most_normal_val <= 0.001:
            idx_of_normal = np.where(dot_products == most_normal_val)[0][0]
            most_normal = norms[idx_of_normal]
            approx_axis = np.cross(unit_vector(vect), unit_vector(most_normal))
            sum_of_vectors += approx_axis
            found += 1
    log.info(f"found {found} in {iter_num} iterations")
    axis_guess = np.asarray(sum_of_vectors) / found
    return axis_guess

def evaluate_orientation(pcd):
    """
    Determine the apporoximate orientation
       of a bounding cylinder for a point cloud
    """
    # Get normals and align to Z axis
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20)
    )
    pcd.orient_normals_consistent_tangent_plane(100)
    pcd.normalize_normals()
    norms = np.array(pcd.normals)
    axis_guess = orientation_from_norms(norms, samples=100, max_iter=1000)
    return axis_guess

def kmeans_feature(smoothed_feature, pcd= None):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(smoothed_feature[:,np.newaxis])
    unique_vals, counts = np.unique(kmeans.labels_, return_counts=True)
    log.info(f'{unique_vals=} {counts=}')
    cluster_idxs = [np.where(kmeans.labels_==val)[0] for val in unique_vals]
    cluster_features = [smoothed_feature[idxs] for idxs in cluster_idxs]
    return cluster_idxs, cluster_features

def fit_shape_RANSAC(
    pcd=None,
    pts=None,
    threshold=0.1,
    lower_bound=None,
    max_radius=None,
    align_to_z=False,
    shape="circle",
    **kwargs,
):
    if pts is None:
        pts = np.asarray(pcd.points)
    if lower_bound:
        for pt in pts:
            if pt[2] < lower_bound:
                pt[2] = lower_bound

    lowest, heighest = min(pts[:, 2]), max(pts[:, 2])

    radius = get_radius(pts)
    fit_pts = np.asarray(pts.copy())
    if shape == "circle":
        for pt in fit_pts:
            pt[2] = 0
        shape_cls = pyrsc.Circle()
    if shape == "cylinder":
        shape_cls = pyrsc.Cylinder()

    center, axis, fit_radius, inliers = shape_cls.fit(
        pts=np.asarray(fit_pts), thresh=threshold
    )
    log.info(f"fit_cyl = center: {center}, axis: {axis}, radius: {fit_radius}")

    if max_radius is not None:
        if fit_radius> max_radius:
            log.info(f'{shape} had radius {fit_radius} but max_radius is {max_radius}')
            return None, None, None, None, None

    if len(center) == 0:
        log.info(f"no no fit {shape} found")
        return None, None, None, None, None

    in_pts = pts[inliers]
    lowest, heighest = min(in_pts[:, 2]), max(in_pts[:, 2])
    height = heighest - lowest
    center_height = (height / 2) + lowest
    test_center = [center[0], center[1], center_height]

    # test_center = center
    if height <= 0:
        breakpoint()
        return None, None, None, None, None

    if align_to_z:
        if ((axis[0] == 0 and axis[1] == 0 and axis[2] == 1) or
            (axis[0] == 0 and axis[1] == 0 and axis[2] == -1)):
            log.info(f'No Rotation Needed')
            cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,
                                center=tuple(test_center),
                                radius=fit_radius*1.2,
                                height=height)
        else:
            log.info(f'Rotation Needed')
            cyl_mesh = get_shape(pts, shape='cylinder', as_pts=False,
                                center=tuple(test_center),
                                radius=fit_radius*1.2,
                                height=height,
                                axis=axis)
        
    shape_radius = fit_radius * 1.05
    cyl_mesh = get_shape(
        pts,
        shape="cylinder",
        as_pts=False,
        center=tuple(test_center),
        radius=shape_radius,
        height=height,
        axis=axis,
        **kwargs,
    )
    in_pcd = None
    if pcd is not None:
        in_pcd = pcd.select_by_index(inliers)
        pcd.paint_uniform_color([1.0, 0, 0])
        in_pcd.paint_uniform_color([0, 1.0, 0])

    return cyl_mesh, in_pcd, inliers, fit_radius, axis

def get_shape(pts, shape="sphere", as_pts=True, rotate="axis", **kwargs):
    """
    Generate a geometric shape (sphere or cylinder) that fits the given points,
    with options for extracting it as a mesh or as uniformly sampled points.

    Args:
        pts (array-like): An array of points to determine the location and size of the shape.
        shape (str): The type of shape to create ("sphere" or "cylinder").
        as_pts (bool): If True, returns the shape as a point cloud. If False, returns a mesh.
        rotate (str): Rotation strategy, either "axis" for axis-angle or other for different rotation.
        **kwargs: Additional parameters:
            - center: Center of the shape (automatically computed if not provided).
            - radius: Radius of the shape (automatically computed if not provided).
            - height: Height for cylinder (required for 'cylinder' shape).
            - axis: The axis for rotation (optional).

    Returns:
        open3d.geometry.PointCloud or open3d.geometry.TriangleMesh:
            The constructed shape as a point cloud or mesh object.
    """
    if not kwargs.get("center"):
        kwargs["center"] = get_center(pts)
    if not kwargs.get("radius"):

        kwargs["radius"] = get_radius(pts)

    if shape == "sphere":
        shape = o3d.geometry.TriangleMesh.create_sphere(radius=kwargs["radius"])
    elif shape == "cylinder":
        try:
            shape = o3d.geometry.TriangleMesh.create_cylinder(
                radius=kwargs["radius"], height=kwargs["height"]
            )
        except Exception as e:
            breakpoint()
            log.info(f"error getting cylinder {e}")

    # log.info(f'Starting Translation/Rotation')

    if as_pts:
        shape = shape.sample_points_uniformly()
        shape.paint_uniform_color([0, 1.0, 0])

    shape.translate(kwargs["center"])
    arr = kwargs.get("axis")
    if arr is not None:
        vector = unit_vector(arr)
        log.info(f"rotate vector {arr}")
        if rotate == "axis":
            R = shape.get_rotation_matrix_from_axis_angle(kwargs["axis"])
        else:
            R = rotation_matrix_from_arr([0, 0, 1], vector)
        shape.rotate(R, center=kwargs["center"])
    elif rotate == "axis":
        log.info("no axis given for rotation, not rotating")
        return shape

    return shape