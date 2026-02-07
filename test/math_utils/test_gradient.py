from __future__ import annotations

import os
import numpy as np
import pytest
from unittest.mock import Mock, patch, mock_open
import open3d as o3d

from pydar_utils.math_utils.gradient import get_smoothed_features


# Test fixtures
@pytest.fixture
def sample_point_data():
    """Sample point data dictionary for testing."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ])

    return {
        'points': points,
        'intensity': np.array([10.0, 20.0, 15.0, 25.0, 30.0, 35.0])
    }


@pytest.fixture
def sample_pcd():
    """Sample point cloud for testing."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def sample_detail_pcd():
    """Sample detail point cloud for testing."""
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1],
        [0.4, 0.9, 0.1],
        [0.4, 0.4, 0.9],
        [0.1, 0.1, 0.9],
        [0.9, 0.1, 0.9],
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@patch('os.path.exists')
@patch('numpy.load')
def test_get_smoothed_features_existing_file(mock_np_load, mock_exists, sample_point_data):
    """Test get_smoothed_features when smoothed data file already exists."""
    # Mock that the smoothed data file exists
    mock_exists.return_value = True

    # Mock the loaded data
    mock_loaded_data = Mock()
    mock_loaded_data.return_value = sample_point_data['intensity']
    mock_np_load.return_value = mock_loaded_data

    save_file = "test_data/test_file.npz"
    result_intensity, result_pcd = get_smoothed_features(sample_point_data, save_file=save_file)

    # Should return the loaded data
    np.testing.assert_array_equal(result_intensity, sample_point_data['intensity'])
    assert result_pcd is not None

    # Check that the file existence was checked
    mock_exists.assert_called()


@patch('os.path.exists')
@patch('open3d.io.read_point_cloud')
@patch('scipy.spatial.KDTree')
def test_get_smoothed_features_with_detail_data(mock_kdtree_cls, mock_read_pcd, mock_exists, sample_point_data, sample_pcd, sample_detail_pcd):
    """Test get_smoothed_features when processing detail data."""
    # Mock file existence checks
    mock_exists.side_effect = [False, False, False]  # smoothed_data_file, detail_file_name, detail_data_file don't exist

    # Mock the point cloud reading
    mock_read_pcd.return_value = sample_detail_pcd

    # Mock KDTree
    mock_kdtree = Mock()
    distances = np.array([[0.1, 0.2, 0.3] for _ in range(len(sample_detail_pcd.points))])
    neighbors = np.array([[0, 1, 2] for _ in range(len(sample_detail_pcd.points))])
    mock_kdtree.query.return_value = (distances, neighbors)
    mock_kdtree_cls.return_value = mock_kdtree

    # Mock tqdm to avoid import issues
    with patch('tqdm.tqdm') as mock_tqdm:
        mock_tqdm.return_value = neighbors  # Return neighbors directly

        save_file = "test_data/test_file.npz"
        result_intensity, result_pcd = get_smoothed_features(sample_point_data, save_file=save_file)

        # Should process and return detail data
        assert len(result_intensity) == len(sample_detail_pcd.points)
        assert result_pcd == sample_detail_pcd

        # Check that KDTree was created and queried
        mock_kdtree_cls.assert_called_once_with(sample_pcd.points)
        mock_kdtree.query.assert_called_once()


@patch('os.path.exists')
@patch('open3d.io.read_point_cloud')
@patch('scipy.spatial.KDTree')
def test_get_smoothed_features_existing_detail_file(mock_kdtree_cls, mock_read_pcd, mock_exists, sample_point_data):
    """Test get_smoothed_features when detail data file already exists."""
    # Mock file existence checks - smoothed data doesn't exist, but detail data does
    mock_exists.side_effect = [False, True, True]

    # Mock the loaded detail data
    expected_detail_data = np.array([15.0, 25.0, 20.0, 30.0, 35.0, 40.0])

    with patch('numpy.load') as mock_np_load:
        mock_loaded_detail = Mock()
        mock_loaded_detail.__getitem__.return_value = expected_detail_data
        mock_np_load.return_value = mock_loaded_detail

        save_file = "test_data/test_file.npz"
        result_intensity, result_pcd = get_smoothed_features(sample_point_data, save_file=save_file)

        # Should return the loaded detail data
        np.testing.assert_array_equal(result_intensity, expected_detail_data)
        assert result_pcd is not None

        # Check that point cloud was read for detail data
        mock_read_pcd.assert_called_once()


@patch('os.path.exists')
def test_get_smoothed_features_error_handling(mock_exists, sample_point_data):
    """Test get_smoothed_features error handling."""
    # Mock file existence - smoothed data exists but loading fails
    mock_exists.return_value = True

    with patch('numpy.load', side_effect=Exception("Load error")):
        save_file = "test_data/test_file.npz"
        # Should continue processing despite the error
        result_intensity, result_pcd = get_smoothed_features(sample_point_data, save_file=save_file)

        # Should still return some result
        assert result_intensity is not None
        assert result_pcd is not None


@patch('os.path.exists')
@patch('scipy.spatial.KDTree')
def test_get_smoothed_features_kdtree_neighbors_filtering(mock_kdtree_cls, mock_exists, sample_point_data, sample_pcd, sample_detail_pcd):
    """Test neighbor filtering in KDTree query."""
    # Mock file existence
    mock_exists.side_effect = [False, False, False]

    # Mock KDTree with some invalid neighbors (index >= num_pts)
    mock_kdtree = Mock()
    num_pts = len(sample_pcd.points)
    distances = np.array([[0.1, 0.2, 0.3] for _ in range(len(sample_detail_pcd.points))])
    neighbors = np.array([[0, 1, num_pts + 1] for _ in range(len(sample_detail_pcd.points))])  # Last neighbor is invalid
    mock_kdtree.query.return_value = (distances, neighbors)
    mock_kdtree_cls.return_value = mock_kdtree

    with patch('tqdm.tqdm') as mock_tqdm, \
         patch('open3d.io.read_point_cloud', return_value=sample_detail_pcd):

        mock_tqdm.return_value = neighbors

        save_file = "test_data/test_file.npz"
        result_intensity, result_pcd = get_smoothed_features(sample_point_data, save_file=save_file)

        # Should filter out invalid neighbors and still work
        assert len(result_intensity) == len(sample_detail_pcd.points)
        assert all(np.isfinite(val) for val in result_intensity)
