import os 
import subprocess
import sys
import pickle 
import open3d as o3d
import numpy as np
from set_config import log, config
from numpy import array as arr
from itertools import product
from prettytable import PrettyTable
import laspy

# from open3d.io import read_point_cloud

use_super_user = config['io']['super_user']
data_root = config['io']['data_root']

# def read_pcd(file, root_dir = data_root):
#     pcd = read_point_cloud
#     return pcd

def save_line_set(line_set, base_file = 'skel_stem20_topology',root_dir=data_root):
    base_file.replace('.pkl','')
    save(f'{base_file}_lines.pkl', arr(line_set.lines),root_dir)
    save(f'{base_file}_points.pkl', arr(line_set.points),root_dir)
    
def load_line_set(base_file = 'skel_stem20_topology',root_dir=data_root):
    base_file.replace('.pkl','')
    lines = load(f'{base_file}_lines.pkl',root_dir)
    points = load(f'{base_file}_points.pkl',root_dir)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def update(file, to_write, root_dir = data_root):
    curr = load(file,root_dir)
    curr.extend(to_write)
    save(file,curr,root_dir)

def save(file, to_write, root_dir = data_root):
    if '.pkl' not in file: file=f'{file}.pkl'
    fqp = f'{root_dir}{file}'

    with open(fqp,'wb') as f:
        pickle.dump(to_write,f)

def load(file, root_dir = data_root):
    if '.pkl' not in file: file=f'{file}.pkl'
    if root_dir[-1] != '/': root_dir += '/'
    fqp = f'{root_dir}{file}'

    with open(fqp,'rb') as f:
        ret = pickle.load(f)
    return ret

## Reading in various file types 

def get_attrs_las(las_file, header=None):
    try:
        x = las_file.X * header.scales[0] + header.offsets[0]
        y = las_file.Y * header.scales[1] + header.offsets[1]
        z = las_file.Z * header.scales[2] + header.offsets[2]
    except AttributeError as e:
        log.info(f'No scale attributes found, using coordinates directly')
        x,y,z = las_file.X, las_file.Y, las_file.Z
    red = las_file.red
    blue = las_file.blue
    green = las_file.green
    points = np.vstack((x, y, z)).T
    colors = np.vstack((red, green, blue)).T
    data = np.hstack([points,colors, np.arange(len(x))[:,np.newaxis]])
    return data

def convert_las(file_name, file_dir='', ext='pcd'):
    orig_ext = file_name.split('.')[-1]
    # file_name = 'EpiphytusTV4.pts'
    # # file_dir = 'data/epip/inputs'
    # # file_name = 'cleaned_ds10_epip.pcd'
    file = f'{file_dir}/{file_name}' if file_dir != '' else file_name
    las = laspy.read(file)
    data = get_attrs_las(las)

    if ext == 'pcd':
        pts = data[:, :3]
        colors = data[:, 3:6]/65280
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f'/{file_name.replace(orig_ext,'.pcd')}', pcd)
        return pcd
    else: 
        return data

def np_to_o3d(npz_file):
    data = np.load(npz_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['points'])
    log.info('warning: dividing colors by 255')
    if 'colors' in data.files:
        pcd.colors = o3d.utility.Vector3dVector(data['colors']/255)
    return pcd

def to_o3d(coords=None, colors=None, las=None):
    if las is not None:
        las = np.asarray(las)
        coords = las[:, :3]
        if las.shape[1]>3:
            labels = las[:, 3]
        if las.shape[1]>4:
            colors = las[:, 4:7]       
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return pcd

def create_table(
                results:list[dict] | list[tuple[str,dict[str,dict]]] | dict[str,dict[str,dict]]
                 ,cols=None,sub_cols=None,ids=None):
    if cols is None:
        if isinstance(results,dict):
            cols = [x for x in results.keys()]
        else:
            row1 = results[0]
            if isinstance(row1,tuple):
                cols = list(row1[1].keys())
            elif isinstance(row1,dict):
                cols = list(row1.keys())
            else:
                cols = []
        print(f'{cols=}')

    if sub_cols is not None:
        all_cols =list([x for x in product(cols, sub_cols)])
        col_names = [f'{col}_{sub_col}' for col, sub_col in all_cols]
    else:
        col_names = cols
    print(f'{col_names=}')
    
    fin_cols = ['ID'] + col_names
    myTable = PrettyTable(fin_cols)

    if ids is not None:
        if isinstance(results[0],tuple):
            ids = [x[0] for x in results]
            rows = [x[1] for x in results]
            results = dict(zip(ids,rows))
        elif isinstance(results,dict):
            ids = [x for x in results.keys()]
        elif isinstance(results,list):
            results = dict([(idx,x) for idx,x in enumerate(results)])

    for row_id, row_data in results:
        row = [row_data[col] for col in col_names]
        if sub_cols is not None:
            row = [row[col] for col in sub_cols]
        myTable.add_row([row_id] + row)
    print(myTable)
    return myTable