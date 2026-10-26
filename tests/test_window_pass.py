"""
tests/test_window_pass.py

Run with:  pytest -v tests/test_window_pass.py

Covers:
  - geometry.py       (corner ordering, yaw/roll/throttle errors, mask extraction)
  - pid.py            (proportional response, integral clamp, smoothing, reset)
  - aligner.py        (full state machine: SEARCH -> YAW+ROLL -> THROTTLE ->
                       APPROACH -> PASS -> DONE, plus LOST handling)
  - config.py         (defaults, dataclass nesting, custom overrides)
  - detector.py       (parsing mocked YOLO results, annotate no-op on empty)
  - hud.py            (camera center, draw does not crash, DONE banner)
  - drone.py (async)  (RC pump cadence, dedup, dry-mode, async land/connect)
  - app.py (async)    (fps counter, throttled logging, key-driven shutdown)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# FIX #1: package-relative imports to match the source tree.
from aligner import AlignCommand, Aligner
from config import (
    AppConfig,
    ApproachConfig,
    CameraConfig,
    MinSpeeds,
    ModelConfig,
    PassConfig,
    RollConfig,
    RuntimeConfig,
    SearchConfig,
    ThrottleConfig,
    YawConfig,
)
from detector import EMPTY_DETECTION, Detection, Detector
from geometry import (
    extract_quad_from_mask,
    geometric_yaw_error,
    order_quad_corners,
    roll_error_from_corners,
    throttle_error_from_center,
    yaw_error_from_center,
)
from pid import PID


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def cfg():
    """A test config with predictable, small values."""
    c = AppConfig()
    c.search.time_360 = 0.05
    c.search.up_duration = 0.01
    c.search.up_after_360 = True
    c.search.max_up_count = 2
    c.passing.extra_time = 0.02
    c.passing.max_time = 0.1
    c.approach.max_time = 0.1
    c.runtime.max_lost_frames = 3
    c.runtime.smoothing_alpha = 1.0
    c.runtime.pid_output_smoothing = 1.0
    c.yaw.stable_count = 1
    c.roll.stable_count = 1
    c.throttle.stable_count = 1
    return c


@pytest.fixture
def aligner(cfg):
    return Aligner(cfg)


def make_detection(cx, cy, area=5000, corners=None):
    if corners is None:
        corners = np.array(
            [
                [cx - 50, cy - 50],
                [cx + 50, cy - 50],
                [cx + 50, cy + 50],
                [cx - 50, cy + 50],
            ],
            dtype=np.float32,
        )
    return Detection(
        box=(cx - 50, cy - 50, cx + 50, cy + 50),
        area=area,
        confidence=0.9,
        center=(cx, cy),
        corners=corners,
        mask_polygon=corners.astype(np.int32),
    )


CAMERA_CENTER = (320, 125)
FRAME_SHAPE = (480, 640, 3)


# ===========================================================================
# geometry.py
# ===========================================================================
class TestGeometry:
    def test_order_quad_corners_axis_aligned(self):
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        shuffled = pts[[2, 0, 3, 1]]
        ordered = order_quad_corners(shuffled)
        assert np.allclose(ordered[0], [10, 10])
        assert np.allclose(ordered[1], [90, 10])
        assert np.allclose(ordered[2], [90, 90])
        assert np.allclose(ordered[3], [10, 90])

    def test_order_quad_corners_rotated(self):
        pts = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32)
        ordered = order_quad_corners(pts)
        assert ordered.shape == (4, 2)
        assert np.allclose(ordered[0] + ordered[2], ordered[1] + ordered[3])

    def test_extract_quad_from_mask_none(self):
        assert extract_quad_from_mask(None) is None

    def test_extract_quad_from_mask_too_few_points(self):
        assert extract_quad_from_mask(np.array([[0, 0], [1, 1]])) is None

    def test_extract_quad_from_mask_small_area(self):
        tiny = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        assert extract_quad_from_mask(tiny, min_area=2000) is None

    def test_extract_quad_from_mask_valid(self):
        big = np.array(
            [[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32
        )
        quad = extract_quad_from_mask(big, min_area=1000)
        assert quad is not None
        assert quad.shape == (4, 2)

    def test_extract_quad_from_mask_min_area_respected(self):
        medium = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32)
        assert extract_quad_from_mask(medium, min_area=3000) is None
        assert extract_quad_from_mask(medium, min_area=1000) is not None

    def test_yaw_error_from_center(self):
        assert yaw_error_from_center((360, 125), (320, 125)) == 40.0
        assert yaw_error_from_center((280, 125), (320, 125)) == -40.0
        assert yaw_error_from_center(None, (320, 125)) is None
        assert yaw_error_from_center((360, 125), None) is None

    def test_throttle_error_from_center(self):
        assert throttle_error_from_center((320, 200), (320, 125)) == 75.0
        assert throttle_error_from_center((320, 50), (320, 125)) == -75.0
        assert throttle_error_from_center(None, (320, 125)) is None

    def test_geometric_yaw_error_symmetric(self):
        corners = np.array(
            [[100, 100], [540, 100], [600, 400], [40, 400]], dtype=np.float32
        )
        err = geometric_yaw_error(corners)
        assert err is not None
        assert abs(err) < 1e-6

    def test_geometric_yaw_error_asymmetric(self):
        corners = np.array(
            [[100, 100], [540, 100], [500, 400], [40, 400]], dtype=np.float32
        )
        err = geometric_yaw_error(corners)
        assert err is not None
        assert err < 0

    def test_geometric_yaw_error_invert(self):
        corners = np.array(
            [[100, 100], [540, 100], [500, 400], [40, 400]], dtype=np.float32
        )
        normal = geometric_yaw_error(corners, invert=False)
        inverted = geometric_yaw_error(corners, invert=True)
        assert normal is not None and inverted is not None
        assert pytest.approx(normal, abs=1e-6) == -inverted

    def test_geometric_yaw_error_none(self):
        assert geometric_yaw_error(None) is None
        assert geometric_yaw_error(np.array([[0, 0], [1, 1]])) is None

    def test_roll_error_flat(self):
        corners = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32
        )
        assert roll_error_from_corners(corners) == 0.0

    def test_roll_error_tilted(self):
        corners = np.array(
            [[0, 0], [100, 100], [0, 200], [-100, 100]], dtype=np.float32
        )
        err = roll_error_from_corners(corners)
        assert err is not None
        assert pytest.approx(abs(err), abs=1.0) == 45.0

    def test_roll_error_invert(self):
        corners = np.array(
            [[0, 0], [100, 20], [80, 120], [-20, 100]], dtype=np.float32
        )
        normal = roll_error_from_corners(corners, invert=False)
        inverted = roll_error_from_corners(corners, invert=True)
        assert normal is not None and inverted is not None
        assert pytest.approx(normal, abs=1e-6) == -inverted

    def test_roll_error_none(self):
        assert roll_error_from_corners(None) is None


# ===========================================================================
# pid.py
# ===========================================================================
class TestPID:
    def test_proportional_only(self):
        pid = PID(Kp=2.0, Ki=0.0, Kd=0.0, output_smoothing=1.0)
        assert pid.update(10.0) == pytest.approx(20.0)

    def test_integral_accumulates_and_clamps(self):
        pid = PID(Kp=0.0, Ki=1.0, Kd=0.0, integral_limit=5.0, output_smoothing=1.0)
        pid.update(100.0)
        assert pid.integral == pytest.approx(5.0)

    def test_output_smoothing(self):
        pid = PID(Kp=1.0, Ki=0.0, Kd=0.0, output_smoothing=0.5)
        first = pid.update(10.0)
        second = pid.update(10.0)
        assert first == pytest.approx(5.0)
        assert second == pytest.approx(7.5)

    def test_reset(self):
        pid = PID(Kp=1.0, Ki=1.0, Kd=1.0)
        pid.update(5.0)
        pid.update(5.0)
        assert pid.integral != 0.0
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        assert pid.last_output == 0.0

    def test_derivative_responds_to_change(self):
        pid = PID(Kp=0.0, Ki=0.0, Kd=1.0, output_smoothing=1.0)
        pid.update(0.0)
        out = pid.update(100.0)
        assert out > 0.0


# ===========================================================================
# config.py
# ===========================================================================
class TestConfig:
    def test_defaults_exist(self):
        c = AppConfig()
        assert isinstance(c.model, ModelConfig)
        assert isinstance(c.search, SearchConfig)
        assert isinstance(c.yaw, YawConfig)
        assert isinstance(c.roll, RollConfig)
        assert isinstance(c.throttle, ThrottleConfig)
        assert isinstance(c.approach, ApproachConfig)
        assert isinstance(c.passing, PassConfig)
        assert isinstance(c.camera, CameraConfig)
        assert isinstance(c.min_speeds, MinSpeeds)
        assert isinstance(c.runtime, RuntimeConfig)

    def test_sane_values(self):
        c = AppConfig()
        assert 0.0 < c.model.confidence <= 1.0
        assert c.runtime.max_lost_frames > 0
        assert c.approach.area_ratio_threshold > 0
        assert c.passing.area_drop > 0
        assert c.camera.frame_width > 0 and c.camera.frame_height > 0
        # async additions
        assert c.runtime.rc_send_interval > 0
        assert c.runtime.key_poll_interval > 0

    def test_override(self):
        c = AppConfig()
        c.yaw.kp = 9.9
        assert c.yaw.kp == 9.9


# ===========================================================================
# aligner.py
# ===========================================================================
class TestAligner:
    def test_initial_state(self, aligner):
        assert aligner.stage == "YAW"
        assert aligner.control_mode == "IDLE"
        assert aligner.yaw_centered is False
        assert aligner.throttle_centered is False

    def test_reset(self, aligner):
        aligner.stage = "APPROACH"
        aligner.yaw_centered = True
        aligner.reset()
        assert aligner.stage == "YAW"
        assert aligner.yaw_centered is False

    def test_centered_detection_completes_yaw_roll_throttle(self, aligner):
        det = make_detection(320, 125)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.yaw_centered is True
        assert info.roll_centered is True
        assert info.throttle_centered is True
        info2 = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info2.stage.startswith("APPROACH")

    def test_yaw_not_centered_when_offset(self, aligner):
        det = make_detection(400, 125)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.yaw_centered is False
        assert info.yaw_cmd > 0

    def test_yaw_not_centered_when_left(self, aligner):
        det = make_detection(240, 125)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.yaw_centered is False
        assert info.yaw_cmd < 0

    def test_throttle_direction(self, aligner):
        det = make_detection(320, 200)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.throttle_centered is False
        assert info.throttle_cmd < 0

    def test_throttle_direction_up(self, aligner):
        det = make_detection(320, 50)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.throttle_centered is False
        assert info.throttle_cmd > 0

    def test_roll_disabled(self, cfg):
        cfg.roll.enabled = False
        aligner = Aligner(cfg)
        det = make_detection(320, 125)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.roll_centered is True
        assert info.roll_cmd == 0

    def test_lost_increments_then_searches(self, aligner, cfg):
        empty = EMPTY_DETECTION
        for _ in range(cfg.runtime.max_lost_frames):
            info = aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
            assert info.stage.startswith("LOST")
        info = aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
        assert "SEARCH" in info.stage
        assert info.yaw_cmd != 0

    def test_found_after_lost_resets_counter(self, aligner, cfg):
        empty = EMPTY_DETECTION
        for _ in range(cfg.runtime.max_lost_frames - 1):
            aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
        assert aligner.lost_frames == cfg.runtime.max_lost_frames - 1
        det = make_detection(320, 125)
        aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert aligner.lost_frames == 0

    def test_search_360_rotates(self, aligner, cfg):
        empty = EMPTY_DETECTION
        for _ in range(cfg.runtime.max_lost_frames + 1):
            info = aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
        assert "SEARCH 360" in info.stage
        assert info.yaw_cmd == cfg.search.yaw_speed

    def test_search_up_after_360(self, aligner, cfg):
        empty = EMPTY_DETECTION
        for _ in range(cfg.runtime.max_lost_frames + 1):
            aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
        time.sleep(cfg.search.time_360 + 0.02)
        info = aligner.update(empty, FRAME_SHAPE, CAMERA_CENTER)
        assert "SEARCH UP" in info.stage
        assert info.throttle_cmd == cfg.search.up_throttle

    def _drive_to_approach(self, aligner):
        det = make_detection(320, 125)
        aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert aligner.stage == "APPROACH"

    def test_approach_drives_forward(self, aligner, cfg):
        self._drive_to_approach(aligner)
        det = make_detection(320, 125, area=5000)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.stage.startswith("APPROACH")
        assert info.forward_cmd == cfg.approach.speed

    def test_approach_to_pass_on_area(self, aligner, cfg):
        self._drive_to_approach(aligner)
        large_area = (
            int(FRAME_SHAPE[0] * FRAME_SHAPE[1] * cfg.approach.area_ratio_threshold)
            + 1
        )
        det = make_detection(320, 125, area=large_area)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert info.stage.startswith("PASS START")
        assert info.forward_cmd == cfg.passing.forward_speed
        assert aligner.stage == "PASS"

    def test_approach_timeout_to_pass(self, aligner, cfg):
        self._drive_to_approach(aligner)
        time.sleep(cfg.approach.max_time + 0.02)
        det = make_detection(320, 125, area=5000)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert "PASS START (timeout)" in info.stage
        assert aligner.stage == "PASS"

    def test_pass_extra_time_to_done(self, aligner, cfg):
        self._drive_to_approach(aligner)
        large_area = (
            int(FRAME_SHAPE[0] * FRAME_SHAPE[1] * cfg.approach.area_ratio_threshold)
            + 1
        )
        det = make_detection(320, 125, area=large_area)
        aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert aligner.stage == "PASS"

        det_big = make_detection(320, 125, area=large_area * 2)
        aligner.update(det_big, FRAME_SHAPE, CAMERA_CENTER)
        time.sleep(0.6)
        small_area = int(large_area * (cfg.passing.area_drop - 0.1))
        det_small = make_detection(320, 125, area=small_area)
        info = aligner.update(det_small, FRAME_SHAPE, CAMERA_CENTER)
        assert "PASS extra" in info.stage
        assert aligner.pass_extra_time is not None

        time.sleep(cfg.passing.extra_time + 0.02)
        info = aligner.update(det_small, FRAME_SHAPE, CAMERA_CENTER)
        assert info.stage == "DONE"
        assert info.pass_completed is True
        assert aligner.stage == "DONE"

    def test_done_short_circuits(self, aligner):
        aligner.stage = "DONE"
        info = aligner.update(EMPTY_DETECTION, FRAME_SHAPE, CAMERA_CENTER)
        assert info.stage == "DONE"
        assert info.pass_completed is True

    def test_pass_timeout_to_done(self, aligner, cfg):
        self._drive_to_approach(aligner)
        large_area = (
            int(FRAME_SHAPE[0] * FRAME_SHAPE[1] * cfg.approach.area_ratio_threshold)
            + 1
        )
        det = make_detection(320, 125, area=large_area)
        aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        time.sleep(cfg.passing.max_time + 0.02)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert "DONE (timeout)" in info.stage
        assert info.pass_completed is True

    def test_area_none_during_approach_returns_to_yaw(self, aligner):
        self._drive_to_approach(aligner)
        det = Detection(
            box=(270, 75, 370, 175),
            area=0,
            confidence=0.9,
            center=(320, 125),
            corners=None,
            mask_polygon=None,
        )
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert "lost" in info.stage.lower()
        assert aligner.stage == "YAW+ROLL"

    def test_align_command_fields_populated(self, aligner):
        det = make_detection(350, 130)
        info = aligner.update(det, FRAME_SHAPE, CAMERA_CENTER)
        assert isinstance(info, AlignCommand)
        assert info.yaw_error is not None
        assert info.throttle_error is not None
        assert isinstance(info.yaw_cmd, int)
        assert isinstance(info.throttle_cmd, int)
        assert isinstance(info.forward_cmd, int)
        assert isinstance(info.roll_cmd, int)

    def test_align_command_defaults(self):
        cmd = AlignCommand()
        assert cmd.yaw_cmd == 0
        assert cmd.stage == "IDLE"
        assert cmd.pass_completed is False
        assert cmd.control_mode == "IDLE"


# ===========================================================================
# detector.py
# ===========================================================================
class TestDetector:
    def _make_detector(self, min_mask_area=100):
        with patch("detector.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model
            det = Detector(
                model_path="fake.pt",
                confidence=0.5,
                inference_size=640,
                max_detections=5,
                device="cpu",
                min_mask_area=min_mask_area,
            )
        return det

    def _fake_results(self, boxes, masks_xy=None):
        result = MagicMock()
        result.boxes = []
        for (x1, y1, x2, y2), conf in boxes:
            box = MagicMock()
            box.conf = [conf]
            box.xyxy = [np.array([x1, y1, x2, y2])]
            result.boxes.append(box)

        if masks_xy is not None:
            masks = MagicMock()
            masks.xy = masks_xy
            result.masks = masks
        else:
            result.masks = None
        return [result]

    def test_empty_result(self):
        det = self._make_detector()
        det.model.predict.return_value = self._fake_results([])
        out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.box is None
        assert out.center is None

    def test_picks_largest_box(self):
        det = self._make_detector()
        det.model.predict.return_value = self._fake_results(
            [
                ((10, 10, 60, 60), 0.9),
                ((100, 100, 300, 300), 0.8),
            ]
        )
        out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.box == (100, 100, 300, 300)
        assert out.area == 40000
        assert out.confidence == 0.8
        assert out.center == (200, 200)

    def test_low_confidence_filtered(self):
        det = self._make_detector()
        det.model.predict.return_value = self._fake_results([((10, 10, 60, 60), 0.1)])
        out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.box is None

    def test_mask_polygon_attached(self):
        det = self._make_detector()
        mask = np.array(
            [[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32
        )
        det.model.predict.return_value = self._fake_results(
            [((100, 100, 300, 300), 0.9)], masks_xy=[mask]
        )
        out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.mask_polygon is not None
        assert out.corners is not None
        assert out.corners.shape == (4, 2)

    def test_annotate_empty_is_noop(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        Detector.annotate(frame, EMPTY_DETECTION)
        assert frame.sum() == 0

    def test_annotate_draws(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = make_detection(320, 240)
        Detector.annotate(frame, det)
        assert frame.sum() > 0


# ===========================================================================
# hud.py
# ===========================================================================
class TestHUD:
    def test_camera_center_uses_offset(self):
        from hud import HUD
        cam = CameraConfig(offset_x=10, offset_y=130, frame_width=640, frame_height=480)
        hud = HUD(cam)
        cx, cy = hud.camera_center((480, 640, 3))
        assert cx == 320 + 10
        assert cy == 130

    def test_draw_does_not_crash(self):
        from hud import HUD
        hud = HUD(CameraConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = AlignCommand(
            yaw_error=10.0,
            roll_error=2.0,
            throttle_error=-5.0,
            area=5000,
            yaw_cmd=5,
            roll_cmd=3,
            throttle_cmd=-2,
            forward_cmd=0,
            stage="YAW+ROLL",
            control_mode="SIMULTANEOUS",
        )
        hud.draw(
            frame,
            info,
            window_center=(330, 130),
            camera_center=(320, 125),
            fps=30.0,
            inference_size=640,
        )
        assert frame.sum() > 0

    def test_draw_done_banner(self):
        from hud import HUD
        hud = HUD(CameraConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = AlignCommand(stage="DONE", pass_completed=True)
        hud.draw(frame, info)
        assert frame[:, :, 1].sum() > 0


# ===========================================================================
# drone.py (async)
# ===========================================================================
class TestDrone:
    def _make_drone(self, flight_mode=True, min_battery=20, rc_send_interval=0.005):
        with patch("drone.Tello") as mock_tello_cls:
            mock_tello = MagicMock()
            mock_tello.get_battery.return_value = 80
            mock_tello_cls.return_value = mock_tello
            from drone import Drone
            d = Drone(flight_mode, min_battery, rc_send_interval=rc_send_interval)
        return d, mock_tello

    async def test_connect_ok(self):
        d, mock = self._make_drone()
        batt = await d.connect()
        assert batt == 80
        mock.connect.assert_called_once()

    async def test_connect_low_battery_exits(self):
        d, mock = self._make_drone(min_battery=50)
        mock.get_battery.return_value = 10
        with pytest.raises(SystemExit):
            await d.connect()

    async def test_rc_pump_forwards_new_command(self):
        d, mock = self._make_drone(flight_mode=True, rc_send_interval=0.005)
        d.start_rc_pump()
        d.set_rc(1, 2, 3, 4)
        await asyncio.sleep(0.05)
        mock.send_rc_control.assert_any_call(1, 2, 3, 4)
        await d.shutdown()

    async def test_rc_pump_dedups_identical_commands(self):
        d, mock = self._make_drone(flight_mode=True, rc_send_interval=0.005)
        d.start_rc_pump()
        for _ in range(20):
            d.set_rc(5, 5, 5, 5)
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        calls_with_target = [
            c for c in mock.send_rc_control.call_args_list if c.args == (5, 5, 5, 5)
        ]
        assert len(calls_with_target) <= 1
        await d.shutdown()

    async def test_send_rc_blocked_in_dry_mode(self):
        d, mock = self._make_drone(flight_mode=False, rc_send_interval=0.005)
        d.start_rc_pump()
        d.set_rc(1, 2, 3, 4)
        await asyncio.sleep(0.05)
        mock.send_rc_control.assert_not_called()
        await d.shutdown()

    async def test_land_only_in_flight_mode(self):
        d, mock = self._make_drone(flight_mode=False)
        await d.land()
        mock.land.assert_not_called()

        d2, mock2 = self._make_drone(flight_mode=True)
        await d2.land()
        mock2.land.assert_called_once()

    async def test_rc_pump_survives_exceptions(self):
        d, mock = self._make_drone(flight_mode=True, rc_send_interval=0.005)
        mock.send_rc_control.side_effect = Exception("boom")
        d.start_rc_pump()
        d.set_rc(1, 1, 1, 1)
        await asyncio.sleep(0.05)
        assert d._rc_task is not None and not d._rc_task.done()
        await d.shutdown()

    async def test_shutdown_cancels_pump(self):
        d, _ = self._make_drone(flight_mode=True, rc_send_interval=0.005)
        d.start_rc_pump()
        assert d._rc_task is not None
        await d.shutdown()
        assert d._rc_task is None

    async def test_stop_zeros_rc(self):
        d, mock = self._make_drone(flight_mode=True, rc_send_interval=0.005)
        d.start_rc_pump()
        d.set_rc(10, 10, 10, 10)
        await asyncio.sleep(0.02)
        d.stop()
        await asyncio.sleep(0.05)
        mock.send_rc_control.assert_any_call(0, 0, 0, 0)
        await d.shutdown()


# ===========================================================================
# app.py (async smoke tests)
# ===========================================================================
class TestApp:
    def _make_app(self):
        # FIX #2 + #15: use DI + AsyncMock instead of patching every collaborator.
        with patch("app.torch.cuda.is_available", return_value=False):
            from app import WindowPassApp

            detector = MagicMock()
            aligner = MagicMock()
            hud = MagicMock()
            drone = MagicMock()
            drone.connect = AsyncMock()
            drone.start_stream = AsyncMock()
            drone.takeoff = AsyncMock()
            drone.hover_kick = AsyncMock()
            drone.land = AsyncMock()
            drone.shutdown = AsyncMock()
            app = WindowPassApp(
                AppConfig(),
                detector=detector,
                aligner=aligner,
                hud=hud,
                drone=drone,
            )
        return app

    async def test_fps_counter(self):
        app = self._make_app()
        app._fps_last_time = time.perf_counter()
        app._fps_frames = 0
        for _ in range(100):
            app._update_fps()
        app._fps_last_time -= 2.0
        app._update_fps()
        assert app._fps > 0.0

    def test_log_status_throttled(self, capsys):
        app = self._make_app()
        info = AlignCommand(
            stage="YAW", yaw_cmd=5, roll_cmd=2, throttle_cmd=-1, forward_cmd=0,
        )
        app._last_status_log = 0.0
        app._log_status(info)
        out1 = capsys.readouterr().out
        assert "[YAW]" in out1

        app._log_status(info)
        out2 = capsys.readouterr().out
        assert out2 == ""

    async def test_key_queue_drives_shutdown(self):
        app = self._make_app()
        app._key_queue.put_nowait("q")

        class _FR:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        app.detector.detect = MagicMock(return_value=EMPTY_DETECTION)
        app.detector.annotate = MagicMock()
        app.aligner.update = MagicMock(return_value=AlignCommand())
        app.hud.draw = MagicMock()
        app.hud.camera_center = MagicMock(return_value=(320, 125))
        app.drone.set_rc = MagicMock()

        with patch("app.cv2.imshow"), patch("app.cv2.waitKey", return_value=255), patch(
            "app.cv2.resize", side_effect=lambda f, *a, **k: f
        ):
            await asyncio.wait_for(app._loop(_FR()), timeout=2.0)

    async def test_cv2_q_breaks_loop(self):
        app = self._make_app()

        class _FR:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        app.detector.detect = MagicMock(return_value=EMPTY_DETECTION)
        app.detector.annotate = MagicMock()
        app.aligner.update = MagicMock(return_value=AlignCommand())
        app.hud.draw = MagicMock()
        app.hud.camera_center = MagicMock(return_value=(320, 125))
        app.drone.set_rc = MagicMock()

        with patch("app.cv2.imshow"), patch(
            "app.cv2.waitKey", return_value=ord("q")
        ), patch("app.cv2.resize", side_effect=lambda f, *a, **k: f):
            await asyncio.wait_for(app._loop(_FR()), timeout=2.0)

    async def test_cleanup_is_idempotent_safe(self):
        app = self._make_app()
        app.drone.stop = MagicMock()
        app.drone.land = AsyncMock()
        app.drone.shutdown = AsyncMock()
        with patch("app.cv2.destroyAllWindows"):
            await app._cleanup()        