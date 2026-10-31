"""Aligner tests — fake Detection, no hardware, no YOLO."""

import numpy as np

from src.window_tello_refactor.application.aligner import Aligner
from window_tello_refactor.domain.params import (
    AlignerParams, ApproachParams, MinSpeeds, PassParams, PIDParams,
    SearchParams,
)
from src.window_tello_refactor.domain.entities import Detection


def _params() -> AlignerParams:
    return AlignerParams(
        yaw=PIDParams(kp=0.4, ki=0, kd=0, max_speed=35,
                      threshold=4.0, exit_threshold=8.0,
                      stable_count=3, deadzone=15.0),
        roll=PIDParams(kp=1.5, ki=0, kd=0, max_speed=30,
                       threshold=4.0, exit_threshold=8.0,
                       stable_count=3, deadzone=3.0),
        throttle=PIDParams(kp=0.4, ki=0.05, kd=0, max_speed=25,
                           threshold=13.0, exit_threshold=26.0,
                           stable_count=3, deadzone=10.0),
        search=SearchParams(),
        approach=ApproachParams(),
        passing=PassParams(),
        min_speeds=MinSpeeds(),
    )


def _detection(cx: int, cy: int) -> Detection:
    box = (cx - 50, cy - 50, cx + 50, cy + 50)
    corners = np.array(
        [[cx - 50, cy - 50], [cx + 50, cy - 50],
         [cx + 50, cy + 50], [cx - 50, cy + 50]],
        dtype=np.float32,
    )
    return Detection(box, 10000, 0.9, (cx, cy), corners, None)


def test_aligner_centers_yaw_after_stable_frames():
    a = Aligner(_params())
    frame_shape = (480, 640, 3)
    center = (320, 240)
    for _ in range(5):
        cmd = a.update(_detection(320, 240), frame_shape, center)
    assert cmd.yaw_centered
    assert cmd.roll_centered


def test_aligner_reports_errors_before_centering():
    a = Aligner(_params())
    cmd = a.update(_detection(400, 240), (480, 640, 3), (320, 240))
    assert cmd.yaw_error is not None
    assert cmd.yaw_error > 0