from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch
import open3d as o3d

from pydar_utils.math_utils.fit import (
    z_align_and_fit,
    cluster_2d,
    orientation_from_norms,
    evaluate_orientation,
    kmeans_feature,
    fit_shape_RANSAC,
    get_shape,
)


# Test data fixtures
@pytest.fixture
def sample_points():
    """Sample 3D points for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ])


@pytest.fixture
def sample_pcd(sample_points):
    """Sample point cloud for testing."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_points)
    return pcd


@pytest.fixture
def sample_normals():
    """Sample surface normals for testing."""
    return np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])


# Test orientation_from_norms
def test_orientation_from_norms(sample_normals):
    """Test orientation estimation from normals."""
    result = orientation_from_norms(sample_normals, samples=2, max_iter=10)
    assert len(result) == 3
    assert np.linalg.norm(result) > 0  # Should be a non-zero vector


def test_orientation_from_norms_insufficient_normals():
    """Test orientation_from_norms with insufficient normals."""
    norms = np.array([[0.0, 0.0, 1.0]])  # Only one normal
    result = orientation_from_norms(norms, samples=5, max_iter=10)
    # Should return some result even with insufficient data
    assert len(result) == 3


# Test kmeans_feature
def test_kmeans_feature():
    """Test k-means clustering on features."""
    smoothed_feature = np.array([1.0, 2.0, 10.0, 11.0, 12.0])

    cluster_idxs, cluster_features = kmeans_feature(smoothed_feature)

    assert len(cluster_idxs) == 2  # Should have 2 clusters
    assert len(cluster_features) == 2

    # Check that all points are assigned to clusters
    total_points = sum(len(cluster) for cluster in cluster_idxs)
    assert total_points == len(smoothed_feature)


# Test cluster_2d
def test_cluster_2d(sample_pcd):
    """Test 2D clustering."""
    labels = cluster_2d(sample_pcd, axis=2, eps=0.5, min_points=2)

    assert len(labels) == len(sample_pcd.points)
    # Labels should be integers (-1 for noise, 0+ for clusters)
    assert all(isinstance(label, (int, np.integer)) for label in labels)


# Test get_shape
def test_get_shape_sphere(sample_points):
    """Test sphere shape generation."""
    result = get_shape(sample_points, shape="sphere", as_pts=True)

    assert isinstance(result, o3d.geometry.PointCloud)
    assert len(result.points) > 0


def test_get_shape_cylinder(sample_points):
    """Test cylinder shape generation."""
    result = get_shape(
        sample_points,
        shape="cylinder",
        as_pts=True,
        height=2.0
    )

    assert isinstance(result, o3d.geometry.PointCloud)
    assert len(result.points) > 0


# Test fit_shape_RANSAC - need to mock pyransac3d
@patch('pydar_utils.math_utils.fit.pyrsc.Circle')
def test_fit_shape_RANSAC_circle(mock_circle_cls, sample_points):
    """Test RANSAC circle fitting."""
    # Mock the circle fit
    mock_circle = Mock()
    mock_circle.fit.return_value = (
        np.array([0.5, 0.5, 0.0]),  # center
        np.array([0.0, 0.0, 1.0]),  # axis (not used for circle)
        1.0,  # radius
        np.array([0, 1, 2, 3, 4, 5])  # inliers
    )
    mock_circle_cls.return_value = mock_circle

    result = fit_shape_RANSAC(pts=sample_points, shape="circle")

    mesh, in_pcd, inliers, fit_radius, axis = result
    assert fit_radius == 1.0
    assert len(inliers) == 6


@patch('pydar_utils.math_utils.fit.pyrsc.Cylinder')
def test_fit_shape_RANSAC_cylinder(mock_cylinder_cls, sample_points):
    """Test RANSAC cylinder fitting."""
    # Mock the cylinder fit
    mock_cylinder = Mock()
    mock_cylinder.fit.return_value = (
        np.array([0.5, 0.5, 0.5]),  # center
        np.array([0.0, 0.0, 1.0]),  # axis
        1.0,  # radius
        np.array([0, 1, 2, 3, 4, 5])  # inliers
    )
    mock_cylinder_cls.return_value = mock_cylinder

    result = fit_shape_RANSAC(pts=sample_points, shape="cylinder")

    mesh, in_pcd, inliers, fit_radius, axis = result
    assert fit_radius == 1.0
    assert len(inliers) == 6


@patch('pydar_utils.math_utils.fit.pyrsc.Circle')
def test_fit_shape_RANSAC_max_radius_filter(mock_circle_cls, sample_points):
    """Test RANSAC with max_radius filter."""
    # Mock the circle fit with large radius
    mock_circle = Mock()
    mock_circle.fit.return_value = (
        np.array([0.5, 0.5, 0.0]),  # center
        np.array([0.0, 0.0, 1.0]),  # axis
        5.0,  # radius (too large)
        np.array([0, 1, 2, 3, 4, 5])  # inliers
    )
    mock_circle_cls.return_value = mock_circle

    result = fit_shape_RANSAC(pts=sample_points, shape="circle", max_radius=2.0)

    # Should return None due to max_radius filter
    assert result == (None, None, None, None, None)


# Test evaluate_orientation - needs mocking due to open3d dependencies
@patch('pydar_utils.math_utils.fit.orientation_from_norms')
def test_evaluate_orientation(mock_orientation_fn, sample_pcd):
    """Test orientation evaluation."""
    mock_orientation_fn.return_value = np.array([0.0, 0.0, 1.0])

    result = evaluate_orientation(sample_pcd)

    assert len(result) == 3
    mock_orientation_fn.assert_called_once()


# Test z_align_and_fit - complex function with many dependencies
@patch('pydar_utils.math_utils.fit.fit_shape_RANSAC')
@patch('pydar_utils.math_utils.fit.rotation_matrix_from_arr')
def test_z_align_and_fit(mock_rotation_matrix, mock_fit_ransac, sample_pcd):
    """Test z-align and fit functionality."""
    # Mock rotation matrices
    mock_rotation_matrix.side_effect = [
        np.eye(3),  # R_to_z
        np.eye(3),  # R_from_z
    ]

    # Mock fit_shape_RANSAC return
    mock_mesh = Mock()
    mock_fit_ransac.return_value = (mock_mesh, None, [0, 1, 2], 1.0, None)

    result = z_align_and_fit(sample_pcd, axis_guess=np.array([0.0, 0.0, 1.0]))

    assert len(result) == 5
    mock_fit_ransac.assert_called_once()