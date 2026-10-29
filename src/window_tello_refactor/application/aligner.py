"""State machine: SEARCH → YAW+ROLL → THROTTLE → APPROACH → PASS → DONE.

Application layer: orchestrates pure domain geometry & PID, holds state,
and emits AlignCommand value objects. No hardware, no cv2, no config schema.
"""

import time

import numpy as np

from application.params import AlignerParams
from domain.entities import AlignCommand
from domain.geometry import (
    geometric_yaw_error,
    roll_error_from_corners,
    throttle_error_from_center,
    yaw_error_from_center,
)
from domain.pid import PID


class Aligner:
    def __init__(self, params: AlignerParams):
        self.params = params
        self._build_pids()
        self.reset()

    # ------------------------------------------------------------------
    def _build_pids(self):
        p = self.params
        self.pid_yaw = PID(
            p.yaw.kp, p.yaw.ki, p.yaw.kd, p.yaw.integral_limit,
            p.pid_output_smoothing,
        )
        self.pid_roll = PID(
            p.roll.kp, p.roll.ki, p.roll.kd, p.roll.integral_limit,
            p.pid_output_smoothing,
        )
        self.pid_throttle = PID(
            p.throttle.kp, p.throttle.ki, p.throttle.kd,
            p.throttle.integral_limit, p.pid_output_smoothing,
        )

    def reset(self):
        p = self.params

        # Yaw
        self.yaw_stable_count = 0
        self.yaw_centered = False

        # Roll
        self.roll_stable_count = 0
        self.roll_centered = not p.roll.enabled

        # Throttle
        self.throttle_stable_count = 0
        self.throttle_centered = False

        # Search
        self.yaw_speed = p.search.yaw_speed
        self.search_start_time = None
        self.search_direction = 1
        self.up_after_search = False
        self.up_start_time = None
        self.up_count = 0

        # Approach / Pass
        self.approach_start_time = None
        self.pass_start_time = None
        self.pass_initial_area = None
        self.pass_previous_area = None
        self.pass_area_increased = False
        self.pass_extra_time = None

        # Smoothing
        self.smoothed_yaw = None
        self.smoothed_throttle = None
        self.smoothed_roll = None
        self.ALPHA = p.smoothing_alpha

        # Lost
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = p.max_lost_frames

        # State
        self.stage = "YAW"
        self.control_mode = "IDLE"

        # Reset PIDs
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.pid_throttle.reset()

    # ------------------------------------------------------------------
    def update(self, detection, frame_shape, camera_center) -> AlignCommand:
        if self.stage == "PASS":
            return self._handle_pass(detection.area)
        if self.stage == "DONE":
            return AlignCommand(
                stage="DONE",
                pass_completed=True,
                yaw_centered=True,
                roll_centered=True,
                throttle_centered=True,
                control_mode="DONE",
            )

        if detection.center is None:
            return self._handle_lost()

        self.lost_frames = 0
        if self.search_start_time is not None:
            print("[SEARCH] Window found — reset 360")
            self.search_start_time = None
        if self.up_after_search:
            print("[SEARCH] Window found during UP — cancel")
            self.up_after_search = False
            self.up_start_time = None

        yaw_err_raw = yaw_error_from_center(detection.center, camera_center)
        yaw_align_err = geometric_yaw_error(
            detection.corners, self.params.yaw.invert
        )
        thr_err_raw = throttle_error_from_center(
            detection.center, camera_center
        )
        roll_err_raw = roll_error_from_corners(
            detection.corners, self.params.roll.invert
        )

        self.smoothed_yaw = self._smooth(self.smoothed_yaw, yaw_err_raw)
        self.smoothed_throttle = self._smooth(
            self.smoothed_throttle, thr_err_raw
        )
        if roll_err_raw is not None:
            self.smoothed_roll = self._smooth(self.smoothed_roll, roll_err_raw)

        yaw_err = self.smoothed_yaw
        thr_err = self.smoothed_throttle
        roll_err = self.smoothed_roll

        yaw_cmd = roll_cmd = throttle_cmd = forward_cmd = 0

        # ---- YAW + ROLL ----
        if not (self.yaw_centered and self.roll_centered):
            self.stage = "YAW+ROLL"
            self.control_mode = "SIMULTANEOUS"
            yaw_cmd = self._yaw_step(yaw_align_err)
            roll_cmd = self._roll_step(roll_err)
            if self.yaw_centered and self.roll_centered:
                print("[INFO] YAW + ROLL ALIGNED")

        # ---- THROTTLE ----
        if self.yaw_centered and self.roll_centered and not self.throttle_centered:
            self.stage = "THROTTLE"
            self.control_mode = "THROTTLE"
            throttle_cmd = self._throttle_step(thr_err)
        elif self.throttle_centered and thr_err is not None:
            if abs(thr_err) > self.params.throttle.exit_threshold:
                print(f"[INFO] EXIT THROTTLE (err={thr_err:+.1f}px)")
                self.throttle_centered = False
                self.throttle_stable_count = 0
                self.pid_throttle.reset()

        # ---- APPROACH ----
        if (
            self.yaw_centered
            and self.roll_centered
            and self.throttle_centered
            and self.stage != "APPROACH"
        ):
            self.stage = "APPROACH"
            self.control_mode = "APPROACH"
            self.approach_start_time = time.time()
            print("[INFO] START APPROACH")

        if self.stage == "APPROACH":
            return self._handle_approach(
                yaw_err, roll_err, thr_err, detection.area, frame_shape
            )

        return self._make(
            yaw_err, roll_err, thr_err, detection.area,
            yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
            self.stage,
            self.yaw_centered, self.roll_centered, self.throttle_centered,
            False,
        )

    # ------------------------------------------------------------------
    def _yaw_step(self, yaw_align_err):
        p = self.params.yaw
        if self.yaw_centered:
            if (
                yaw_align_err is not None
                and abs(yaw_align_err) > p.exit_threshold
            ):
                print(f"[INFO] EXIT YAW (err={yaw_align_err:+.1f}%)")
                self.yaw_centered = False
                self.yaw_stable_count = 0
                self.pid_yaw.reset()
                self.stage = "YAW+ROLL"
                out = self.pid_yaw.update(yaw_align_err)
                return int(np.clip(out, -p.max_speed, p.max_speed))
            return 0

        if yaw_align_err is None:
            self.yaw_stable_count = 0
            return 0

        if abs(yaw_align_err) < p.threshold:
            self.pid_yaw.reset()
            self.yaw_stable_count += 1
            if self.yaw_stable_count >= p.stable_count:
                self.yaw_centered = True
                print(f"[INFO] YAW CENTERED (err={yaw_align_err:+.1f}%)")
            return 0

        out = self.pid_yaw.update(yaw_align_err)
        yaw_cmd = int(np.clip(out, -p.max_speed, p.max_speed))
        yaw_cmd = self._apply_min_speed(
            yaw_cmd, yaw_align_err,
            self.params.min_speeds.yaw, p.deadzone,
        )
        self.yaw_stable_count = 0
        return yaw_cmd

    def _roll_step(self, roll_err):
        p = self.params.roll
        if not p.enabled or roll_err is None:
            self.roll_stable_count = 0
            return 0

        if self.roll_centered:
            if abs(roll_err) > p.exit_threshold:
                print(f"[INFO] EXIT ROLL (err={roll_err:+.2f}deg)")
                self.roll_centered = False
                self.roll_stable_count = 0
                self.pid_roll.reset()
            return 0

        if abs(roll_err) < p.threshold:
            self.pid_roll.reset()
            self.roll_stable_count += 1
            if self.roll_stable_count >= p.stable_count:
                self.roll_centered = True
                print(f"[INFO] ROLL CENTERED (err={roll_err:+.2f}deg)")
            return 0

        out = self.pid_roll.update(roll_err)
        roll_cmd = int(np.clip(out, -p.max_speed, p.max_speed))
        roll_cmd = self._apply_min_speed(
            roll_cmd, roll_err, self.params.min_speeds.roll, p.deadzone,
        )
        self.roll_stable_count = 0
        return roll_cmd

    def _throttle_step(self, thr_err):
        p = self.params.throttle
        if thr_err is None:
            self.throttle_stable_count = 0
            return 0

        if abs(thr_err) < p.threshold:
            self.pid_throttle.reset()
            self.throttle_stable_count += 1
            if self.throttle_stable_count >= p.stable_count:
                self.throttle_centered = True
                print(f"[INFO] THROTTLE CENTERED (err={thr_err:+.1f}px)")
            return 0

        out = self.pid_throttle.update(-thr_err)
        throttle_cmd = int(np.clip(out, -p.max_speed, p.max_speed))
        throttle_cmd = self._apply_min_speed(
            throttle_cmd, thr_err,
            self.params.min_speeds.throttle, deadzone=p.deadzone,
            invert=True,
        )
        self.throttle_stable_count = 0
        return throttle_cmd

    @staticmethod
    def _apply_min_speed(
        cmd, err, min_speed, deadzone, invert=False,
    ):
        if abs(cmd) >= min_speed or err is None:
            return cmd
        direction_err = -err if invert else err
        if abs(direction_err) <= deadzone:
            return 0
        return min_speed if direction_err > 0 else -min_speed

    # ------------------------------------------------------------------
    def _handle_lost(self):
        now = time.time()
        p = self.params

        if self.stage in ("APPROACH", "THROTTLE", "YAW+ROLL", "YAW"):
            self.lost_frames += 1
            if self.lost_frames < self.MAX_LOST_FRAMES:
                return self._make(
                    None, None, None, None, 0, 0, 0, 0,
                    f"LOST ({self.lost_frames}/{self.MAX_LOST_FRAMES})",
                    False, False, False, False,
                )
            print(f"[INFO] Lost for {self.lost_frames} frames -> SEARCH")
            self.yaw_centered = False
            self.roll_centered = not p.roll.enabled
            self.throttle_centered = False
            self.pid_yaw.reset()
            self.pid_roll.reset()
            self.pid_throttle.reset()
            self.search_start_time = None
            self.up_after_search = False
            self.up_start_time = None

        if self.up_after_search:
            elapsed_up = now - self.up_start_time
            if elapsed_up >= p.search.up_duration:
                print("[SEARCH] UP done -> restart 360")
                self.up_after_search = False
                self.up_start_time = None
                self.search_start_time = now
                self.search_direction = 1
                self.up_count += 1
                return self._make(
                    None, None, None, None, 0, 0, 0, 0,
                    "SEARCH 360 (restart)",
                    False, False, False, False,
                )
            return self._make(
                None, None, None, None, 0, p.search.up_throttle, 0, 0,
                f"SEARCH UP ({elapsed_up:.1f}/{p.search.up_duration}s, "
                f"#{self.up_count})",
                False, False, False, False,
            )

        if self.search_start_time is None:
            self.search_start_time = now
            print(f"[SEARCH] Start 360 (dir={self.search_direction})")

        elapsed_search = now - self.search_start_time
        if elapsed_search >= p.search.time_360:
            print("[SEARCH] 360 completed")
            if (
                p.search.up_after_360
                and self.up_count < p.search.max_up_count
            ):
                print("[SEARCH] Going UP")
                self.up_after_search = True
                self.up_start_time = now
                self.search_start_time = None
                return self._make(
                    None, None, None, None, 0, p.search.up_throttle, 0, 0,
                    f"SEARCH UP (start, #{self.up_count + 1})",
                    False, False, False, False,
                )
            if p.search.alternate:
                self.search_direction *= -1
            self.search_start_time = now
            elapsed_search = 0

        yaw_cmd = self.search_direction * self.yaw_speed
        self.yaw_centered = False
        if p.roll.enabled:
            self.roll_centered = False
        self.throttle_centered = False
        self.stage = "SEARCH 360"
        self.control_mode = "SEARCH"
        self.yaw_stable_count = self.roll_stable_count = 0
        self.throttle_stable_count = 0
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.pid_throttle.reset()
        return self._make(
            None, None, None, None, yaw_cmd, 0, 0, 0,
            f"SEARCH 360 ({elapsed_search:.1f}/{p.search.time_360}s)",
            False, False, False, False,
        )

    # ------------------------------------------------------------------
    def _handle_approach(self, yaw_err, roll_err, thr_err, area, frame_shape):
        now = time.time()
        p = self.params
        elapsed = now - self.approach_start_time

        yaw_cmd = roll_cmd = throttle_cmd = 0
        forward_cmd = p.approach.speed

        if yaw_err is not None and abs(yaw_err) > p.approach.deadzone_yaw:
            out = self.pid_yaw.update(yaw_err)
            yaw_cmd = int(np.clip(out, -p.approach.center_max, p.approach.center_max))
            yaw_cmd = self._apply_min_speed(
                yaw_cmd, yaw_err,
                p.min_speeds.yaw, p.approach.deadzone_yaw,
            )

        if (
            p.roll.enabled
            and roll_err is not None
            and abs(roll_err) > p.approach.deadzone_roll
        ):
            out = self.pid_roll.update(roll_err)
            roll_cmd = int(np.clip(out, -p.approach.center_max, p.approach.center_max))
            roll_cmd = self._apply_min_speed(
                roll_cmd, roll_err,
                p.min_speeds.roll, p.approach.deadzone_roll,
            )

        if thr_err is not None and abs(thr_err) > p.approach.deadzone_thr:
            out = self.pid_throttle.update(-thr_err)
            throttle_cmd = int(
                np.clip(out, -p.approach.center_max, p.approach.center_max)
            )
            throttle_cmd = self._apply_min_speed(
                throttle_cmd, thr_err,
                p.min_speeds.throttle, p.approach.deadzone_thr,
                invert=True,
            )

        if area is None:
            print("[INFO] Lost during APPROACH")
            self.stage = "YAW+ROLL"
            self.control_mode = "SIMULTANEOUS"
            self.yaw_centered = False
            if p.roll.enabled:
                self.roll_centered = False
            self.throttle_centered = False
            self.pid_yaw.reset()
            self.pid_roll.reset()
            self.pid_throttle.reset()
            return self._make(
                yaw_err, roll_err, thr_err, None, 0, 0, 0, 0,
                "YAW+ROLL (lost)", False, False, False, False,
            )

        total_pixels = frame_shape[0] * frame_shape[1]
        area_ratio = area / total_pixels

        if area_ratio >= p.approach.area_ratio_threshold:
            print(f"[INFO] STOP APPROACH (area_ratio={area_ratio:.3f}) -> PASS")
            self._enter_pass()
            return self._make(
                yaw_err, roll_err, thr_err, area, 0, 0,
                p.passing.forward_speed, 0,
                f"PASS START (area_ratio={area_ratio:.3f})",
                True, True, True, False,
            )

        if elapsed >= p.approach.max_time:
            print("[INFO] APPROACH timeout -> PASS")
            self._enter_pass()
            return self._make(
                yaw_err, roll_err, thr_err, area, 0, 0,
                p.passing.forward_speed, 0,
                "PASS START (timeout)", True, True, True, False,
            )

        return self._make(
            yaw_err, roll_err, thr_err, area,
            yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
            f"APPROACH ({elapsed:.1f}s, area={area})",
            True, True, True, False,
        )

    def _enter_pass(self):
        self.stage = "PASS"
        self.control_mode = "PASS"
        self.pass_start_time = None
        self.pass_initial_area = None
        self.pass_previous_area = None
        self.pass_area_increased = False
        self.pass_extra_time = None
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.pid_throttle.reset()

    # ------------------------------------------------------------------
    def _handle_pass(self, area):
        now = time.time()
        p = self.params.passing

        if self.pass_start_time is None:
            self.pass_start_time = now
            self.pass_initial_area = area or 0
            self.pass_previous_area = self.pass_initial_area
            self.pass_area_increased = False
            self.pass_extra_time = None
            print(f"[PASS] START (initial_area={self.pass_initial_area})")

        if self.pass_extra_time is not None:
            elapsed_extra = now - self.pass_extra_time
            if elapsed_extra >= p.extra_time:
                print("[PASS] DONE")
                self.stage = "DONE"
                self.control_mode = "DONE"
                return self._make(
                    None, None, None, area, 0, 0, 0, 0,
                    "DONE", True, True, True, True,
                )
            return self._make(
                None, None, None, area, 0, 0,
                p.forward_speed, 0,
                f"PASS extra ({elapsed_extra:.1f}/{p.extra_time}s)",
                True, True, True, False,
            )

        elapsed = now - self.pass_start_time

        if (
            self.pass_previous_area is not None
            and self.pass_previous_area > 0
            and area is not None
            and area > self.pass_previous_area * 1.05
        ):
            self.pass_area_increased = True

        if (
            self.pass_area_increased
            and elapsed > 0.5
            and (
                area is None
                or area < self.pass_initial_area * p.area_drop
            )
        ):
            print("[PASS] Passing detected")
            self.pass_extra_time = now

        if area is not None:
            self.pass_previous_area = area

        if elapsed >= p.max_time:
            print("[PASS] Timeout -> DONE")
            self.stage = "DONE"
            self.control_mode = "DONE"
            return self._make(
                None, None, None, area, 0, 0, 0, 0,
                "DONE (timeout)", True, True, True, True,
            )

        return self._make(
            None, None, None, area, 0, 0,
            p.forward_speed, 0,
            f"PASS ({elapsed:.1f}s, area={area})",
            True, True, True, False,
        )

    # ------------------------------------------------------------------
    def _smooth(self, prev, new):
        if new is None:
            return prev
        if prev is None:
            return new
        return (1 - self.ALPHA) * prev + self.ALPHA * new

    def _make(
        self, yaw_err, roll_err, thr_err, area,
        yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
        stage, yaw_c, roll_c, thr_c, pass_done,
    ) -> AlignCommand:
        return AlignCommand(
            yaw_error=yaw_err,
            roll_error=roll_err,
            throttle_error=thr_err,
            area=area,
            yaw_cmd=int(yaw_cmd),
            roll_cmd=int(roll_cmd),
            throttle_cmd=int(throttle_cmd),
            forward_cmd=int(forward_cmd),
            stage=stage,
            yaw_centered=yaw_c,
            roll_centered=roll_c,
            throttle_centered=thr_c,
            pass_completed=pass_done,
            control_mode=self.control_mode,
        )