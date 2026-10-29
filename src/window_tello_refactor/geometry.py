"""Pure geometry helpers — no state, no side effects."""

import cv2
import numpy as np


def order_quad_corners(pts):
    """Order 4 points as [TL, TR, BR, BL]. Robust to rotation/perspective."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )


def extract_quad_from_mask(mask_polygon, min_area: float = 2000):
    """Get a clean 4-corner polygon from a segmentation mask."""
    if mask_polygon is None or len(mask_polygon) < 4:
        return None

    contour = np.asarray(mask_polygon, dtype=np.float32).reshape(-1, 1, 2)
    if cv2.contourArea(contour.astype(np.float32)) < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour.astype(np.int32), 0.02 * perimeter, True)
    if len(approx) == 4 and cv2.contourArea(approx) > 100:
        return order_quad_corners(approx.reshape(4, 2).astype(float))

    rect = cv2.minAreaRect(contour)
    if min(rect[1]) < 2:
        return None
    return order_quad_corners(cv2.boxPoints(rect).astype(float))


def yaw_error_from_center(window_center, camera_center):
    if window_center is None or camera_center is None:
        return None
    return float(window_center[0] - camera_center[0])


def throttle_error_from_center(window_center, camera_center):
    if window_center is None or camera_center is None:
        return None
    return float(window_center[1] - camera_center[1])


def geometric_yaw_error(window_corners, invert=False):
    """Yaw error = percent asymmetry between left and right edge lengths."""
    if window_corners is None or len(window_corners) != 4:
        return None
    pts = order_quad_corners(window_corners)
    left_len = float(np.linalg.norm(pts[3] - pts[0]))
    right_len = float(np.linalg.norm(pts[2] - pts[1]))
    avg_len = (left_len + right_len) / 2.0
    if avg_len < 1.0:
        return None
    err = 100.0 * (right_len - left_len) / avg_len
    return -err if invert else err


def roll_error_from_corners(window_corners, invert=False):
    """Roll error = tilt of the top edge relative to horizontal, in degrees."""
    if window_corners is None or len(window_corners) != 4:
        return None

    pts = order_quad_corners(window_corners)
    tl, tr = pts[0], pts[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]

    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return None

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    return -angle_deg if invert else angle_deg