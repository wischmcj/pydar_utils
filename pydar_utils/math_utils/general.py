import numpy as np

from logging import getLogger
log = getLogger(__name__)

rot_90_y = np.array([[0,1,0],[1,0,0],[0,0,-1]]) # rotates dead on
rot_90_z = np.array([[0,-1,0],[1,0,0],[0,0,1]]) 
rot_90_x = np.array([[1,0,0],[0,0,-1],[0,1,0]])
rot_n90_x = np.array([[1,0,0],[0,-1,0],[0,0,1]]) 

def get_percentile(pts, low, high, axis=2, invert = False):
    """
    Returns the indices of the points that fall within the
    low and high percentiles of the z values of the points
    """
    if isinstance(axis,list):
        vals = sum([pts[:, axum] for axum in axis])
        if invert:
           vals =  pts[:, axis[0]] -  pts[:, axis[1]]
    else:
        vals = pts[:, axis]
    lower = np.percentile(vals, low)
    upper = np.percentile(vals, high)
    all_idxs = np.where(vals)
    too_low_idxs = np.where(vals <= lower)
    not_too_low_idxs = np.setdiff1d(all_idxs, too_low_idxs)
    if high<100:
        too_high_idxs = np.where(vals >= upper)
        select_idxs = np.setdiff1d(not_too_low_idxs, too_high_idxs)
    else:
        select_idxs = not_too_low_idxs
    vals = vals[select_idxs]
    # Similar but by scalar max
    # zmin = np.min(pts[:,2])
    # min_mask = np.where(pts[:, 2] <= (zmin+.4))[0]
    # pcd_minus_ground = pcd.select_by_index(min_mask, invert=True)
    return select_idxs, vals

def rotation_matrix_from_arr(a, b: np.array):
    """
    Returns matrix R such that a*R = b.
    """
    if np.linalg.norm(b) == 0 or np.allclose(a, b):
        return np.eye(3)
    if np.linalg.norm(b) < 0.99 or np.linalg.norm(b) > 1.01:
        raise ValueError("b must be a unit vector")
    # Algorithm from https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    # The skew-symmetric cross product matrix of v
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) * -1
    # Rotation matrix as per Rodregues formula
    R = np.eye(3) - vx + np.dot(-vx, -vx) * ((1 - c) / (s**2))
    return R


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm>0:
        return vector/norm
    return np.zeros_like(vector)


def angle_from_xy_plane(v1):
    """
        angle w/ xy plane = angle with projection on xy plane 
        projection of [u,v,w] on xy plane is [u,v,0]
        unit vectors have same angle as source vectors

        angle = arccos(dot product/(norm1*norm2))
        norm1=norm2=1 (as weve reduced to unit vectors)p
        angle = arccos(dot product)
        
    """
    # get projection on xy plane 
    v2 = [v1[0], v1[1], 0]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u),-1.0,1.0))


def get_angles(v1, radians=False, reference = 'XY'):
    """Gets the angle of a vector with the specified axis"""
    if reference == 'XY':
        v2 = [v1[0], v1[1], 0]
    if reference == 'XZ':
        v2 = [v1[0], 0, v1[2]]
    if reference == 'YZ':
        v2 = [0, v1[1], v1[2]]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u),-1.0,1.0))
    if radians:
        return angle_radians
    else:
        return np.degrees(angle_radians)

def get_center(points, center_type="centroid"):
    """Attempts to find a representitiver 'center' given a
    set of 3D points.
    """
    if len(points[0]) != 3:
        breakpoint()
        print("not 3 points")
    if center_type == "centroid":
        # Average of each coordinate value
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    elif center_type == "top":
        # X and y values a straight average 
        # Z values average of the values in 90th %ile
        idxs, vals = get_percentile(points,90, 100)
        top_ten_pts = points[idxs]
        x = top_ten_pts[:, 0]
        y = top_ten_pts[:, 1]
        z = top_ten_pts[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid
    elif center_type == "bottom":
        # X and y values a straight average 
        # Z values average of the values below the 10th %ile
        idxs, vals = get_percentile(points,0, 10)
        top_ten_pts = points[idxs]
        x = top_ten_pts[:, 0]
        y = top_ten_pts[:, 1]
        z = top_ten_pts[:, 2]
        centroid = np.average(x), np.average(y), np.average(z)
        return centroid

def get_radius(points, center_type="centroid"):
    """
    Given a set of 3D points, returns the average distance
    from a theoretical center (defined above)
    """
    center = get_center(points, center_type)
    xy_pts = points[:, :2]
    xy_center = center[:2]
    r = np.average([np.sqrt(np.sum((xy_pt - xy_center) ** 2)) for xy_pt in xy_pts])
    return r

def filter_by_angle(vectors, angle_thresh=10, rev = False):
    angles = np.apply_along_axis(get_angles, 1, vectors)
    angles = np.degrees(angles)
    log.info(f"{angle_thresh=}")
    if rev:
        in_idxs = np.where((angles < -angle_thresh) | (angles > angle_thresh))[0]
    else:
        in_idxs = np.where((angles > -angle_thresh) & (angles < angle_thresh))[0]
    return in_idxs