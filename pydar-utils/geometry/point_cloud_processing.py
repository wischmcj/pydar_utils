from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
from numpy import array as arr
from glob import glob

from open3d.visualization import draw_geometries

from logging import getLogger
log = getLogger(__name__)

def draw(pcds, **kwargs):
    if isinstance(pcds, list):
        draw_geometries(pcds, **kwargs)
    else:
        draw_geometries([pcds], **kwargs)

# subsample data with voxel_size
def voxelize_and_trace(data, voxel_size):
    """
    Voxelizes np.array data and preserves the non-point and color data if present
    """
    points = data[:, :3]
    points = np.round(points, 2)
    if data.shape[1] >= 4:
        other = data[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    bound = np.max(np.abs(points)) + 100
    min_bound, max_bound = np.array([-bound, -bound, -bound]), np.array([bound, bound, bound])
    downpcd, _, idx = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)

    if data.shape[1] >= 4:
        idx_keep = [item[0] for item in idx]
        other = other[idx_keep]
        data = np.hstack((np.asarray(downpcd.points), other))
    else:
        data = np.asarray(downpcd.points)
    
    return data, idx

def recover_from_trace(orig_data, map_idxs, filtered_idxs):
    """
    Recover elements from orig_data whose ids in map_idxs map to ids in filtered_idxs.

    Args:
        orig_data (np.ndarray): The original data array.
        map_idxs (list or np.ndarray): Each element is a list or iterable, where the first element is the target id and others are indices from orig_data that map to that id.
        filtered_idxs (list or np.ndarray): List of ids to select in the first element of each map_idxs group.

    Returns:
        np.ndarray: Filtered rows from orig_data whose ids map to those in filtered_idxs.
        np.ndarray: The corresponding indices (from orig_data) used for selection.
    """
    # Create a set for fast lookup
    filtered_set = set(filtered_idxs)
    map_idxs = np.array(map_idxs)

    filtered_mapee_lists = map_idxs[np.isin(map_idxs[:,0], filtered_set)]
    filtered_mapees = np.array(filtered_mapee_lists).flatten()
    
    selected_indices = np.array(selected_indices)
    recovered = orig_data[selected_indices]
    return recovered, selected_indices

    
def join_pcd_files(files_path, pattern = '*', 
                    voxel_size = None,
                    write_to_file = True):
    detail_files = glob(pattern,root_dir=files_path)
    pcds=[]
    joined = None
    for file in detail_files:
        pcd = o3d.io.read_point_cloud(f'{files_path}/{file}')
        print(f'{file} has {len(pcd.points)} points')
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)
            print(f'{file} has {len(pcd.points)} points after voxel downsampling')
        if joined is None:
            joined = pcd
        else:
            joined += pcd
    o3d.io.write_point_cloud(f'{files_path}/joined.pcd', joined[0])
    return joined

def create_one_or_many_pcds( pts,
                        colors = None,
                        labels = None,
                        single_pcd = False):    
    log.info('creating pcds from points')
    pcds = []
    tree_pts= []
    tree_color=[]
    if not isinstance(pts[0],list) and not isinstance(pts[0],np.ndarray):
        pts = [pts]
    if not labels:
        labels = np.asarray([idx for idx,_ in enumerate(pts)])
    if (not colors):
        labels = arr(labels)
        max_label = labels.max()
        try:
            label_colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        except Exception as e:
            log.info('err')

    # for pcd, color in zip(tree_pcds, colors): pcd.paint_uniform_color(color[:3])
    for pts, color in zip(pts, colors or label_colors): 
        if not colors: 
            # if true, then colors were generated from labels
            color = [color[:3]]*len(pts)
        if single_pcd:
            log.info(f'adding {len(pts)} points to final set')
            tree_pts.extend(pts)
            tree_color.extend(color)
        else:
            cols = [color[:3]]*len(pts)
            log.info(f'creating pcd with {len(pts)} points')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcds.append(pcd)
    if single_pcd:
        log.info(f'combining {len(tree_pts)} points into final pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tree_pts)
        pcd.colors = o3d.utility.Vector3dVector([x for x in tree_color])
        pcds.append(pcd)
    return pcds

def normalize_to_origin(pcd):
    print(f'original: {pcd.get_max_bound()=}, {pcd.get_min_bound()=}')
    center =pcd.get_min_bound()+((pcd.get_max_bound() -pcd.get_min_bound())/2)
    pcd.translate(-center)
    print(f'translated: {pcd.get_max_bound()=}, {pcd.get_min_bound()=}')
    return pcd


def crop(pts, minx=None, maxx=None, miny=None, maxy=None, minz=None, maxz=None):
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    z_vals = pts[:, 2]
    to_remove = []
    all_idxs = [idx for idx, _ in enumerate(pts)]
    for min_val, max_val, pt_vals in [
        (minx, maxx, x_vals),
        (miny, maxy, y_vals),
        (minz, maxz, z_vals),
    ]:
        if min_val:
            to_remove.append(np.where(pt_vals <= min_val))
        if max_val:
            to_remove.append(np.where(pt_vals >= max_val))

    select_idxs = np.setdiff1d(all_idxs, to_remove)
    return select_idxs

def cluster_plus(pcd,
                    eps=0.08,
                    min_points=10,
                    draw_result = True,
                    from_points=True,
                    return_pcds=True,
                    ransac=False):
    if from_points:
        pts=pcd
        pcd = o3d.geometry.PointCloud()
        breakpoint()
        pcd.points = o3d.utility.Vector3dVector(arr(pts))
    
    if ransac:
        plane_model, inliers = pcd.segment_plane(distance_threshold=eps, ransac_n=10, num_iterations=1000)
    else:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))

 
    if draw_result: 
        draw(pcd)

    unique_lbs, counts = np.unique(labels, return_counts=True)
    print(f"point cloud has {counts} clusters")
    # num_clusters = len(unique_vals)
    # if not top: 
    #     top = num_clusters
    label_to_cluster = {ulabel: np.where(labels == ulabel)[0] for ulabel in unique_lbs}
    if return_pcds:
        ret = [pcd.select_by_index(idx_list) for idx_list in label_to_cluster.values()]
    else:
        ret = label_to_cluster
        
    return ret

def cluster_and_get_largest(pcd,
                                eps=.08,
                                min_points=10,
                                draw_clusters = False):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    if draw_clusters: draw(pcd)
    unique_vals, counts = np.unique(labels, return_counts=True)
    largest = unique_vals[np.argmax(counts)]
    max_cluster_idxs = np.where(labels == largest)[0]
    max_cluster = pcd.select_by_index(max_cluster_idxs)
    return max_cluster


