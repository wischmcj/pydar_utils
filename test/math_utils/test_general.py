from __future__ import annotations

import numpy as np
import pytest
import logging

from pydar_utils.math_utils.general import (
    get_percentile,
    rotation_matrix_from_arr,
    unit_vector,
    angle_from_xy_plane,
    get_angles,
    get_center,
    get_radius,
    filter_by_angle,
)

log= logging.getLogger()


# Test fixtures
@pytest.fixture
def sample_points_3d():
    """Sample 3D points for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [0.5, 1.0, 1.0],
        [0.5, 0.5, 1.5],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.5],
    ])


@pytest.fixture
def sample_vectors():
    """Sample 3D vectors for testing."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, -1.0],
    ])

def get_circle_on_xy(center, radius, num_points = 100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Calculate the x and y coordinates using the parametric equations
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full(num_points, center[2])
    
    # Combine x, y, and z coordinates into a single 3D array
    points = np.stack((x, y, z), axis=-1)
    
    return points


# Test cases for unit_vector
unit_vector_test_cases = [
    pytest.param(np.array([3.0, 4.0, 0.0]), np.array([0.6, 0.8, 0.0]), id="2D vector"),
    pytest.param(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), id="x-axis unit vector"),
    pytest.param(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]), id="z-axis vector"),
    pytest.param(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), id="zero vector"),
]


@pytest.mark.parametrize("vector,expected", unit_vector_test_cases)
def test_unit_vector(vector, expected):
    """Test unit vector calculation."""
    result = unit_vector(vector)
    np.testing.assert_array_almost_equal(result, expected)


# Test cases for rotation_matrix_from_arr
rotation_matrix_test_cases = [
    pytest.param(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        id="90 degree rotation around z-axis"
    ),
    pytest.param(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        id="identity rotation"
    ),
]


@pytest.mark.parametrize("a,b", rotation_matrix_test_cases)
def test_rotation_matrix_from_arr(a, b):
    """Test rotation matrix calculation."""
    result = rotation_matrix_from_arr(a, b)

    # Check that it's a valid rotation matrix (orthogonal with det=1)
    assert np.allclose(result @ result.T, np.eye(3))
    assert np.allclose(np.linalg.det(result), 1.0)


def test_rotation_matrix_from_arr_zero_vector():
    """Test rotation_matrix_from_arr with zero vector."""
    result = rotation_matrix_from_arr(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(result, np.eye(3))


def test_rotation_matrix_from_arr_invalid_unit_vector():
    """Test that rotation_matrix_from_arr raises error for non-unit vectors."""
    with pytest.raises(ValueError, match="b must be a unit vector"):
        rotation_matrix_from_arr(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))


# Test cases for angle_from_xy_plane
angle_from_xy_plane_test_cases = [
    pytest.param(np.array([1.0, 0.0, 0.0]), 0.0, id="x-axis vector"),
    pytest.param(np.array([0.0, 1.0, 0.0]), 0.0, id="y-axis vector"),
    pytest.param(np.array([1.0, 1.0, 0.0]), 0.0, id="45-degree vector across xy_plane"),
    pytest.param(np.array([0.0, 0.0, 1.0]), np.pi/2, id="z-axis vector"),
]


@pytest.mark.parametrize("vector,expected", angle_from_xy_plane_test_cases)
def test_angle_from_xy_plane(vector, expected):
    """Test angle calculation from XY plane."""
    result = angle_from_xy_plane(vector)
    np.testing.assert_almost_equal(result,expected,decimal=4)


# Test cases for get_angles
get_angles_test_cases = [
    pytest.param(np.array([1.0, 0.0, 0.0]), True, "XY", 0.0, id="x-axis XY plane"),
    pytest.param(np.array([0.0, 1.0, 0.0]), True, "XY", 0.0, id="y-axis XY plane"),
    pytest.param(np.array([1.0, 0.0, 1.0]), True, "XY", np.pi/4, id="45-degree XZ plane"),
    pytest.param(np.array([1.0, 0.0, 1.0]), True, "XZ", 0, id="45-degree XZ plane"),
    pytest.param(np.array([0.0, 1.0, 0.0]), True, "XZ", np.pi/2, id="45-degree XZ plane"),
]


@pytest.mark.parametrize("vector,radians,reference,expected", get_angles_test_cases)
def test_get_angles(vector, radians, reference, expected):
    """Test angle calculation with different references."""
    result = get_angles(vector, radians=radians, reference=reference)
    np.testing.assert_almost_equal(result, expected,decimal=4)


def test_get_angles_degrees():
    """Test angle calculation in degrees."""
    vector = np.array([1.0, 0.0, 1.0])
    result = get_angles(vector, radians=False, reference="XY")
    expected_degrees = np.degrees(np.pi/4)
    np.testing.assert_almost_equal(result, expected_degrees,decimal=4)


# Test cases for get_center
get_center_test_cases = [
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        "centroid",
        (1.0, 1.0, 1.0),
        id="simple centroid"
    ),
    pytest.param(
        get_circle_on_xy([1,2,3], 2),
        "centroid",
        (1.0, 2.0, 3.0),  # Average of top 90th percentile
        id="top center calculation"
    ),
]


@pytest.mark.parametrize("points,center_type,expected", get_center_test_cases)
def test_get_center(points, center_type, expected):
    """Test center calculation with different methods."""
    result = get_center(points, center_type)
    np.testing.assert_array_almost_equal(result, expected)



# Test cases for get_radius
get_radius_test_cases = [
    pytest.param(
        get_circle_on_xy([0,0,0], 2),
        "centroid",
        pytest.approx(2, abs=1e-3),
        id="unit square radius"
    ),
]


@pytest.mark.parametrize("points,center_type,expected", get_radius_test_cases)
def test_get_radius(points, center_type, expected):
    """Test radius calculation."""
    result = get_radius(points, center_type)
    assert result == expected


# Test cases for get_percentile
get_percentile_test_cases = [
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        25, 75, 2, False,
        id="z-axis percentile filter"
    ),
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
        0, 50, 1, False,
        id="y-axis percentile filter"
    ),
]


@pytest.mark.parametrize("points,low,high,axis,invert", get_percentile_test_cases)
def test_get_percentile(points, low, high, axis, invert):
    """Test percentile filtering."""
    select_idxs, vals = get_percentile(points, low, high, axis, invert)

    assert len(select_idxs) == len(vals)
    assert all(isinstance(idx, (int, np.integer)) for idx in select_idxs)

    # Check that selected values are within expected percentile range
    all_vals = points[:, axis]
    lower_bound = np.percentile(all_vals, low)
    upper_bound = np.percentile(all_vals, high)

    assert all(val >= lower_bound for val in vals)
    if high < 100:
        assert all(val <= upper_bound for val in vals)


# Test cases for filter_by_angle
filter_by_angle_test_cases = [
    pytest.param(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        5.0, False,
        id="filter near-vertical vectors"
    ),
    pytest.param(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        5.0, True,
        id="filter near-horizontal vectors"
    ),
]


@pytest.mark.parametrize("vectors,angle_thresh,rev", filter_by_angle_test_cases)
def test_filter_by_angle(vectors, angle_thresh, rev):
    """Test angle-based vector filtering."""
    result = filter_by_angle(vectors, angle_thresh, rev)

    assert isinstance(result, np.ndarray)
    assert all(isinstance(idx, (int, np.integer)) for idx in result)
    assert all(idx >= 0 and idx < len(vectors) for idx in result)
