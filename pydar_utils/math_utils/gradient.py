import open3d as o3d
import numpy as np
import os
from collections import defaultdict
from set_config import log
from geometry.pc_processing import get_neighbors_kdtree
from viz.viz_utils import color_continuous_map, draw_view
from viz.viz_utils import histogram
import tqdm
from scipy.stats import mode

def get_smoothed_features(all_data, 
                save_file=None,
                profile=False,):
    points = all_data['points']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    smoothed_data = defaultdict(list)
    smoothed_data_file = save_file.replace('.npz', '_smoothed.npz')
    detail_data_file = save_file.replace('.npz', '_detail_feats.npz')

    # Check for existing smoothed data
    if os.path.exists(smoothed_data_file):
        log.info(f'{smoothed_data_file=} already exists, loading from file')
        smoothed_data = np.load(smoothed_data_file)

    try:
        datum_name = 'intensity'
        datum = smoothed_data.get(datum_name)

        detail_file_name = save_file.replace('with_all_feats/','').replace('int_color_data', 'full_ext').replace('.npz', f'_orig_detail.pcd')
        detail_pcd = o3d.io.read_point_cloud(detail_file_name)

        detail_data_file = save_file.replace('.npz', f'_{datum_name}_detail_feats.npz')

        if os.path.exists(detail_data_file):
            final_data = np.load(detail_data_file)['intensity']

        else:
            dists, nbrs = get_neighbors_kdtree(pcd, detail_pcd, return_pcd=False)
            num_pts = len(pcd.points)
            nbrs = [np.array([x for x in nbr_list  if x< num_pts]) for nbr_list in nbrs]
            nbrs = [nbr_list if len(nbr_list) > 0 else np.array([0]) for nbr_list in nbrs]
            final_data = []
            for nbr_list in tqdm(nbrs): final_data.append(np.array(np.mean(datum[nbr_list])))
            final_data = np.array(final_data)
            np.savez_compressed(detail_data_file, intensity = final_data)

        new_detail_pcd, _ = color_continuous_map(detail_pcd, final_data)
        histogram(final_data, datum_name)
        np.sort(final_data)
        
    except Exception as e:
        log.info(f'error {e} when getting detail features')

