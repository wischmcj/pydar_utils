import numpy as np
import re
import logging
from glob import glob
from joblib import Parallel, delayed
from collections import defaultdict
from open3d.io import read_point_cloud as read_pcd
from pydar_utils.utils.io import convert_las, np_to_o3d, load
from pydar_utils.utils.general import list_if
from itertools import product
from typing import Any
import open3d as o3d

log = logging.getLogger()

def get_files_by_seed(data_file_config, 
                        base_dir,
                        key_pattern= re.compile('.*_([0-9]{1,3}).*')
                        ):
    seed_to_files = defaultdict[Any, dict](dict)
    for file_type, file_info in data_file_config.items():
        # Get all matchig files
        folder = f'{base_dir}/{file_info["folder"]}'
        file_pattern = file_info['file_pattern']
    
        files = glob(file_pattern,root_dir=folder)
        # organize files by seed
        for file in files:
            file_key = re.match(key_pattern,file)
            if file_key is None:
                log.info(f'no seed found in seed_to_content: {file}. Ignoring file...')
                continue
            if len(file_key.groups(1)) > 0:
                file_key = file_key.groups(1)[0]
            else:
                file_key = file_key[0]
            seed_to_files[file_key][file_type] = f'{base_dir}/{file_info["folder"]}/{file}'
    return seed_to_files


def np_feature(feature_name):
    def my_feature_func(npz_file):
        npz_data = np.load(npz_file)
        return npz_data[feature_name]
    return my_feature_func

def get_downsample(file = None, 
                    pcd = None, 
                    voxel_size = .15, 
                    uniform_down_sample = None,
                    debug = False,
                    **kwargs):
    """
        for skio voxel_size=.05, uniform_down_sample(3)
    """
    if file: pcd = read_pcd(file)
    log.info('Down sampling pcd with voxel_size={voxel_size}, uniform_down_sample={uniform_down_sample}')
    clean_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if uniform_down_sample is not None:
        clean_pcd = clean_pcd.uniform_down_sample(uniform_down_sample)
    print(f'clean version has {len(clean_pcd.points)} points')
    return clean_pcd

def read_and_downsample(file_path, voxel_size=None, uniform_down_sample=None, **kwargs):
    if file_path.endswith('.npz'):
        pcd = np_to_o3d(file_path)
    elif file_path.endswith('.las'):
        pcd = convert_las(file_path)
    else:
        pcd = read_pcd(file_path, **kwargs)
        clean_pcd = o3d.geometry.PointCloud()
    if voxel_size is not None or uniform_down_sample is not None:
        clean_pcd = get_downsample(pcd=pcd, voxel_size=voxel_size, uniform_down_sample=uniform_down_sample)
    
    return pcd, clean_pcd
    
def get_data_from_config(seed_file_info, data_file_config):
    seed_to_content = defaultdict(dict)
    for file_type, file_path in seed_file_info.items():
        load_func = data_file_config[file_type]['load_func']
        load_kwargs = data_file_config[file_type].get('kwargs',{})
        load_value = load_func(file_path, **load_kwargs)
        file_type = list_if(file_type)
        load_value = list_if(load_value)
        if len(file_type) != len(load_value):
            raise ValueError(f'length of values returned by {load_func.__name__} ({len(load_value)}) is not equal to the length of the file_type {file_type}')
        
        for key, value in zip(file_type, load_value):
            seed_to_content[key] = value
            seed_to_content[f'{file_type}_file'] = file_path
    return seed_to_content

def loop_over_files(func,args = [], kwargs =[],
                    requested_seeds=[],
                    skip_seeds = [],
                    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm',
                    detail_ext_folder = 'ext_detail',
                    data_file_config:dict = { 
                        'src': {
                                'folder': 'ext_detail',
                                'file_pattern': '*orig_detail.pcd',
                                'load_func': read_pcd, # custom, for pickles
                            },
                        'shift_one': {
                                'folder': 'pepi_shift',
                                'file_pattern': '*shift*',
                                'load_func': load, # custom, for pickles
                                'kwargs': {'root_dir': '/'},
                            },
                        'smoothed_feat_data': {
                                'folder': 'ext_detail/with_all_feats/',
                                'file_pattern': 'int_color_data_*_smoothed.npz',  
                                'load_func': np.load,
                            },
                        'detail_feat_data': {
                                'folder': 'ext_detail/with_all_feats/',
                                'file_pattern': 'int_color_data_*_detail_feats.npz',  
                                'load_func': np.load,
                            },
                    },
                    seed_pat = re.compile('.*seed([0-9]{1,3}).*'),
                    parallel = True,
                    ):
    # reads in the files from the indicated directories
    files_by_seed = get_files_by_seed(data_file_config, base_dir, key_pattern=seed_pat)

    files_by_seed = {seed:finfo for seed, finfo in files_by_seed.items() 
                            if ((requested_seeds==[] or seed in requested_seeds)
                              and seed not in skip_seeds)}

    if args ==[]: args = [None]*len(kwargs)
    inputs = [(arg,kwarg) for arg,kwarg in zip(list_if(args),list_if(kwargs))]
    if inputs == []:
        to_run = product(files_by_seed.items(), [([''],{})])
    else:
        to_run =  product(files_by_seed.items(), inputs) 
    results = []
    errors = []
    if parallel:
        content_list = [get_data_from_config(seed_file_info, data_file_config) for seed, seed_file_info in files_by_seed.items()]
        to_call = product(content_list, inputs)
        results = Parallel(n_jobs=3)(delayed(func(content.update({'seed':seed}),*arg_tup,**kwarg_dict)) for (seed,content), (arg_tup, kwarg_dict) in to_call)
    else: 
        for (seed, seed_file_info), (arg_tup, kwarg_dict) in to_run:
            if 1==1:
            # try:
                print(f'Running function for seed {seed=}')
                content = get_data_from_config(seed_file_info, data_file_config)
                content['seed'] = seed
                print(f'running function for {seed} done')
                result = func(content,*arg_tup,**kwarg_dict)
                results.append(result)
            # except Exception as e:
            #     breakpoint()
            #     log.info(f'error {e} when processing seed {seed}')
            #     errors.append(seed)
    print(f'{errors=}')
    print(f'{results=}')
