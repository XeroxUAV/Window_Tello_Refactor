"""OpenCV HUD — presentation only, no business logic."""

import cv2


class HUD:
    def __init__(self, camera_cfg):
        self.camera_cfg = camera_cfg

    def camera_center(self, frame_shape):
        _h, w = frame_shape[:2]
        cx = w // 2 + self.camera_cfg.offset_x
        cy = self.camera_cfg.offset_y
        return (cx, cy)

    def draw(
        self,
        frame,
        info,
        window_center=None,
        camera_center=None,
        dist_left=None,
        dist_right=None,
        fps=0.0,
        inference_size=640,
    ):
        h, w = frame.shape[:2]
        if camera_center is None:
            camera_center = self.camera_center(frame.shape)
        ccx, ccy = camera_center

        self._draw_camera_center(frame, ccx, ccy, h, w)

        if window_center is not None:
            self._draw_window_center(
                frame, ccx, ccy, window_center, dist_left, dist_right, w
            )

        self._draw_stage_banner(frame, info)
        self._draw_metrics(frame, info)
        self._draw_area_and_fps(frame, info, fps, inference_size, h, w)

        if info.pass_completed:
            cv2.putText(
                frame, "DONE!", (w // 2 - 60, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
            )

    def _draw_camera_center(self, frame, ccx, ccy, h, w):
        cv2.circle(frame, (ccx, ccy), 8, (0, 0, 255), -1)
        cv2.circle(frame, (ccx, ccy), 14, (0, 0, 255), 2)
        cv2.putText(
            frame, "CAM", (ccx + 18, ccy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
        cv2.line(frame, (ccx, 0), (ccx, h), (0, 0, 255), 1)
        cv2.line(frame, (0, ccy), (w, ccy), (0, 0, 255), 1)

    def _draw_window_center(
        self, frame, ccx, ccy, window_center, dist_left, dist_right, w
    ):
        wcx, wcy = window_center
        cv2.circle(frame, (wcx, wcy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (wcx, wcy), 14, (0, 255, 0), 2)
        cv2.putText(
            frame, "WIN", (wcx + 18, wcy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        cv2.line(frame, (ccx, ccy), (wcx, wcy), (255, 255, 0), 2)

        if dist_left is not None:
            cv2.putText(
                frame, f"L={dist_left}px", (10, wcy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
            )
        if dist_right is not None:
            cv2.putText(
                frame, f"R={dist_right}px", (w - 100, wcy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
            )

    def _draw_stage_banner(self, frame, info):
        stage = info.stage
        colors = {
            "YAW": (0, 200, 255),
            "YAW+ROLL": (0, 200, 255),
            "THROTTLE": (0, 165, 255),
            "APPROACH": (255, 150, 0),
            "PASS": (0, 255, 0),
            "DONE": (0, 255, 0),
            "SEARCH": (255, 0, 255),
            "LOST": (0, 0, 255),
        }
        color = colors.get(stage.split()[0], (255, 255, 255))
        cv2.putText(
            frame, f"STAGE: {stage}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        cm = info.control_mode
        cm_color = (0, 255, 255) if cm == "SIMULTANEOUS" else (200, 200, 200)
        cv2.putText(
            frame, f"MODE: {cm}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cm_color, 2,
        )

    def _draw_metrics(self, frame, info):
        if info.yaw_error is not None:
            c = (0, 255, 0) if info.yaw_centered else (0, 255, 255)
            cv2.putText(
                frame,
                f"YAW = {info.yaw_error:+.1f}px  CMD = {info.yaw_cmd:+d}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
            )

        if info.roll_error is not None:
            c = (0, 255, 0) if info.roll_centered else (255, 0, 255)
            cv2.putText(
                frame,
                f"ROLL = {info.roll_error:+.2f}deg  CMD = {info.roll_cmd:+d}",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
            )

        if info.throttle_error is not None:
            c = (0, 255, 0) if info.throttle_centered else (0, 255, 255)
            cv2.putText(
                frame,
                f"THR = {info.throttle_error:+.1f}px  CMD = {info.throttle_cmd:+d}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
            )

    def _draw_area_and_fps(self, frame, info, fps, inference_size, h, w):
        if info.area:
            ratio = info.area / (h * w)
            cv2.putText(
                frame,
                f"AREA = {info.area} ({ratio * 100:.1f}%)",
                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | YOLO: {inference_size}",
            (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )