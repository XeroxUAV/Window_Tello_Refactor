"""Pure unit tests — no hardware, no YOLO, no cv2 window. Runs in <10 ms."""

import numpy as np

from domain.geometry import (
    geometric_yaw_error,
    order_quad_corners,
    roll_error_from_corners,
    throttle_error_from_center,
    yaw_error_from_center,
)


def test_yaw_error_from_center_zero_when_aligned():
    assert yaw_error_from_center((320, 240), (320, 240)) == 0.0


def test_yaw_error_from_center_positive_when_right():
    assert yaw_error_from_center((400, 240), (320, 240)) == 80.0


def test_throttle_error_from_center_positive_when_below():
    assert throttle_error_from_center((320, 300), (320, 240)) == 60.0


def test_geometric_yaw_error_zero_on_symmetric_trapezoid():
    # Symmetric trapezoid: left and right edges equal length
    corners = np.array(
        [[100, 50], [500, 50], [400, 400], [200, 400]], dtype=np.float32
    )
    err = geometric_yaw_error(corners)
    assert err is not None
    assert abs(err) < 1e-3


def test_roll_error_zero_when_top_edge_horizontal():
    corners = np.array(
        [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32
    )
    assert abs(roll_error_from_corners(corners)) < 1e-6


def test_roll_error_sign_when_tilted():
    # Top-right lower than top-left -> positive angle (dy > 0)
    corners = np.array(
        [[100, 100], [500, 150], [500, 450], [100, 400]], dtype=np.float32
    )
    err = roll_error_from_corners(corners)
    assert err is not None
    assert err > 0


def test_order_quad_corners_returns_tl_tr_br_bl():
    # Deliberately scrambled input
    pts = np.array(
        [[500, 400], [100, 100], [100, 400], [500, 100]], dtype=np.float32
    )
    ordered = order_quad_corners(pts)
    # Top-left has smallest sum, bottom-right largest
    assert tuple(ordered[0]) == (100.0, 100.0)
    assert tuple(ordered[2]) == (500.0, 400.0)