from pydar_utils.processing import crop_by_percentile
from pydar_utils.viz import draw
import open3d as o3d

def get_terrain_pcd():
    pcd= 0 o3d.io.read_point_cloud('/media/penguaman/writable/lidar_sync/py_qsm/skio/inputs/collective.pcd')
    pcd, _ = crop_by_percentile(pcd, start=0, end=11, axis=2)
    return pcd

if __name__ == '__main__':
    pcd = get_terrain_pcd()
    draw([pcd])
    o3d.io.write_point_cloud('/media/penguaman/writable/lidar_sync/py_qsm/skio/inputs/collective_terrain.pcd', pcd)
    breakpoint()