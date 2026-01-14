

import glob
import open3d as o3d
import scipy.spatial as sps
import numpy as np
from numpy import asarray as arr
import itertools
import tqdm





from collections import defaultdict

import open3d as o3d
import numpy as np
from numpy import asarray as arr

from set_config import log

def expand_features_to_orig(nbr_pcd, orig_pcd, nbr_data):
    # # get neighbors of comp_pcd in the extracted feat pcd
    dists, nbrs = get_neighbors_kdtree(nbr_pcd, orig_pcd, return_pcd=False)

    full_detail_feats = defaultdict(list)
    full_detail_feats['points'] = orig_pcd.points
    # For each list of neighbors, get the average value of each feature in all_data and add it to full_detail_feats
    feat_names = [k for k in nbr_data.keys() if k not in ['points','colors', 'labels']]
    final_data =[]
    nbrs = [np.array([x for x in nbr_list  if x< len(orig_pcd.points)]) for nbr_list in nbrs]
    nbrs = [nbr_list if len(nbr_list) > 0 else np.array([0]) for nbr_list in nbrs]
    for nbr_list in tqdm(nbrs):
        nbr_vals = np.array([np.mean(nbr_data[feat_name][nbr_list]) for feat_name in feat_names])
        final_data.append(nbr_vals)
    final_data = np.array(final_data)
    full_detail_feats['features'] = final_data
    return full_detail_feats


def overlap_voxel_grid(src_pts, comp_voxel_grid = None, source_pcd = None, invert = False):
    if comp_voxel_grid is None:
        comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source_pcd,voxel_size=0.2)

    log.info('querying voxel grid')
    queries = src_pts
    in_occupied_voxel_mask= comp_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    num_in_occupied_voxel = np.sum(in_occupied_voxel_mask)
    log.info(f'{num_in_occupied_voxel} points in occupied voxels')
    if num_in_occupied_voxel == 0:
        return []
    if invert:
        not_in_occupied_voxel_mask = np.ones_like(in_occupied_voxel_mask, dtype=bool)
        not_in_occupied_voxel_mask[in_occupied_voxel_mask] = False
        uniques = np.where(not_in_occupied_voxel_mask)[0]
    # else:
    uniques = np.where(in_occupied_voxel_mask)[0]

    return uniques

def get_neighbors_kdtree(src_pcd, query_pcd=None,query_pts=None, kd_tree = None, dist=0.03, k=25, return_pcd = True):
    
    if query_pcd: query_pts = arr(query_pcd.points)
    src_pts = arr(src_pcd.points)
    if not kd_tree:
        kd_tree = sps.KDTree(src_pts)
    print('Finding neighbors in vicinity') 
    dists,nbrs = kd_tree.query(query_pts, k=k, distance_upper_bound= dist) 

    if return_pcd:
        print('concatenating neighbors') 
        chained_nbrs = [x for x in set(itertools.chain.from_iterable(nbrs)) if x< len(src_pts)]
        if len(chained_nbrs)==0:
            return None, None, None
        # nbr_pts = arr(vicinity_pts)[nbrs]
        print('selecting unique neighbors') 
        uniques = np.unique(chained_nbrs)
        print('building pcd') 
        try:
            pts = src_pts[uniques]
        except Exception as e:
            print(f'error {e} when selecting unique neighbors')
            return None, None, None
        colors = arr(src_pcd.colors)[uniques]

        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(pts)   
        pcd.colors = o3d.utility.Vector3dVector(colors)    
        return pcd, nbrs, chained_nbrs
    else:
        return dists,nbrs

def get_nbrs_voxel_grid(comp_pcd, 
                        comp_file_name,
                        tile_dir, 
                        tile_pattern,
                        invert=False,
                        out_folder='detail',
                        out_file_prefix='detail_feats',
                        ):
    log.info('creating voxel grid')
    comp_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(comp_pcd,voxel_size=0.1)
    log.info('voxel grid created')
    nbr_ids = defaultdict(list)
    all_data =  defaultdict(list)
    
    # dodraw = True
    files = glob.glob(f'{tile_dir}/{tile_pattern}')
    pcd=None
    for file in files:
        if 'SKIO-RaffaiEtAlcolor_int_0' in file or 'SKIO-RaffaiEtAlcolor_int_1' in file or 'SKIO-RaffaiEtAlcolor_int_2' in file:
            log.info(f'skipping {file}')
            continue
        file_name = file.split('/')[-1].replace('.pcd','').replace('.npz','')
        log.info(f'processing {file_name}')
        if '.pcd' in file:
            pcd = read_pcd(file)
            data = {'points': np.array(pcd.points), 'colors': np.array(pcd.colors)}
            data_keys = data.keys()
        else:
            data = np.load(file)
            data_keys = data.files
            log.info(f'{data_keys=}')

        # Determine if the boundaries of pcd and comp_pcd intersect at all
        # Compute bounding boxes for both point clouds and check for intersection
        pcd_min = np.min(data['points'], axis=0)
        pcd_max = np.max(data['points'], axis=0)
        comp_min = np.min(np.asarray(comp_pcd.points), axis=0)
        comp_max = np.max(np.asarray(comp_pcd.points), axis=0)
        # Intersection exists if, on all axes, the max of the lower bounds <= min of the upper bounds
        intersect = np.all((pcd_max >= comp_min) & (comp_max >= pcd_min))
        log.info(f'Bounding box intersection: {intersect}')
        if not intersect:
            log.info("Bounding boxes do not intersect, skipping file.")
            continue
        else:
            log.info("Bounding boxes intersect, processing file.")
    
        nbr_dir = f'{tile_dir}/color_int_tree_nbrs/{file_name}'
        uniques = overlap_voxel_grid(data['points'], comp_voxel_grid, invert=invert)

        if not os.path.exists(nbr_dir):
            os.makedirs(nbr_dir)
        np.savez_compressed(f'{nbr_dir}/{out_file_prefix}_{comp_file_name}.npz', nbrs=uniques)
        log.info('filtering data to neighbors') 
        # smooth_intensity = smooth_feature(src_pts, data['intensity'])
        # all_data['smooth_intensity'].append(smooth_intensity)
        filtered_data = {zfile:data[zfile][uniques] for zfile in data_keys}
        for datum_name, datum in filtered_data.items():
            if len(datum.shape) == 1:
                all_data[datum_name].append(datum)#[:,np.newaxis])
            else:
                all_data[datum_name].append(datum)

    log.info('Saving all data')
    for datum_name, datum in all_data.items():
        if len(datum[0].shape) == 1:
            all_data[datum_name] = np.hstack(datum)
        else:
            all_data[datum_name] = np.vstack(datum)

    np.savez_compressed(f'{out_folder}/{comp_file_name}.npz', **all_data)
    return all_data
