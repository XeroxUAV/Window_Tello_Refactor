"""Main application class — async wires everything together."""

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch

# FIX #1: package-relative imports.
from aligner import Aligner
from config import AppConfig
from detector import Detector
from drone import Drone
from hud import HUD


class WindowPassApp:
    # FIX #2: dependency injection — collaborator construction is optional.
    def __init__(
        self,
        cfg: AppConfig = None,
        detector: Detector | None = None,
        aligner: Aligner | None = None,
        hud: HUD | None = None,
        drone: Drone | None = None,
    ):
        self.cfg = cfg or AppConfig()
        self._setup_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = detector or Detector(
            model_path=self.cfg.model.path,
            confidence=self.cfg.model.confidence,
            inference_size=self.cfg.model.inference_size,
            max_detections=self.cfg.model.max_detections,
            device=self.device,
            min_mask_area=self.cfg.runtime.min_mask_area,
        )
        self.aligner = aligner or Aligner(self.cfg)
        self.hud = hud or HUD(self.cfg.camera)
        self.drone = drone or Drone(
            self.cfg.runtime.flight_mode,
            self.cfg.runtime.min_battery,
            rc_send_interval=self.cfg.runtime.rc_send_interval,
        )

        # Dedicated single-worker pool for YOLO (only used if async_detect).
        self._detect_pool: ThreadPoolExecutor | None = None
        if self.cfg.runtime.async_detect:
            self._detect_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="yolo"
            )

        self._fps_last_time = time.perf_counter()
        self._fps_frames = 0
        self._fps = 0.0
        self._last_status_log = 0.0

        # Keyboard events queued by a background thread.
        self._key_queue: asyncio.Queue[str] = asyncio.Queue()
        self._key_thread: threading.Thread | None = None
        self._key_thread_stop = threading.Event()

        # Signal for the main loop to exit.
        self._shutdown = asyncio.Event()

    # ------------------------------------------------------------------
    def _setup_torch(self) -> None:
        if not torch.cuda.is_available():
            try:
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        cv2.setNumThreads(2)

    # ------------------------------------------------------------------
    # Keyboard: `keyboard.is_pressed` blocks, so isolate in a thread.
    # ------------------------------------------------------------------
    def _start_key_thread(self) -> None:
        try:
            import keyboard  # lazy import — headless tests don't need it
        except ImportError:
            print("[App] `keyboard` not installed — only OpenCV 'q' works.")
            return

        interval = self.cfg.runtime.key_poll_interval
        queue = self._key_queue
        stop = self._key_thread_stop

        def _pump() -> None:
            while not stop.is_set():
                try:
                    # FIX #14: include 'r' so the branch in _loop is reachable.
                    for key in ("q", "e", "r"):
                        if keyboard.is_pressed(key):
                            queue.put_nowait(key)
                            return
                except Exception:
                    pass
                time.sleep(interval)

        self._key_thread = threading.Thread(target=_pump, name="key-pump", daemon=True)
        self._key_thread.start()

    def _stop_key_thread(self) -> None:
        self._key_thread_stop.set()
        if self._key_thread is not None:
            self._key_thread.join(timeout=1.0)
            self._key_thread = None

    # ------------------------------------------------------------------
    async def run(self) -> None:
        try:
            await self.drone.connect()
            await self.drone.start_stream()
            await self.drone.takeoff()
            frame_read = self.drone.frame_reader()

            self.drone.start_rc_pump()
            await self.drone.hover_kick()

            self._start_key_thread()
            print("[App] Starting async main loop...")
            await self._loop(frame_read)
        except KeyboardInterrupt:
            print("\n[App] Interrupted")
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    async def _loop(self, frame_read) -> None:
        loop = asyncio.get_running_loop()

        while not self._shutdown.is_set():
            # ---- 1. Drain queued key events (non-blocking) -------------
            try:
                key = self._key_queue.get_nowait()
            except asyncio.QueueEmpty:
                key = None

            if key is not None:
                if key in ("q", "e"):
                    print(f"[App] {'Q' if key == 'q' else 'EMERGENCY (E)'} pressed")
                    break
                if key == "r":
                    self.aligner.reset()
                    print("[App] Reset")

            # ---- 2. Frame acquisition ----------------------------------
            frame = frame_read.frame
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            frame = cv2.resize(
                frame,
                (self.cfg.camera.frame_width, self.cfg.camera.frame_height),
                interpolation=cv2.INTER_AREA,
            )
            camera_center = self.hud.camera_center(frame.shape)

            # ---- 3. Vision ---------------------------------------------
            if self._detect_pool is not None:
                detection = await loop.run_in_executor(
                    self._detect_pool, self.detector.detect, frame
                )
            else:
                detection = self.detector.detect(frame)
            self.detector.annotate(frame, detection)

            # ---- 4. FSM + control --------------------------------------
            info = self.aligner.update(detection, frame.shape, camera_center)
            self._update_fps()

            # ---- 5. HUD ------------------------------------------------
            self.hud.draw(
                frame,
                info,
                window_center=detection.center,
                camera_center=camera_center,
                fps=self._fps,
                inference_size=self.cfg.model.inference_size,
            )
            self._log_status(info)

            # ---- 6. RC (pump does the actual send) ---------------------
            self.drone.set_rc(
                info.roll_cmd,
                info.forward_cmd,
                info.throttle_cmd,
                info.yaw_cmd,
            )

            # ---- 7. UI (must run on main thread for OpenCV) ------------
            cv2.imshow("Window PASS", frame)
            key_cv = cv2.waitKey(1) & 0xFF
            if key_cv == ord("q"):
                print("[App] Q pressed (cv2)")
                break
            if key_cv == ord("r"):
                self.aligner.reset()
                print("[App] Reset (cv2)")

            # Yield to the event loop so the RC pump / key queue can run.
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    def _update_fps(self) -> None:
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_last_time = now

    def _log_status(self, info) -> None:
        now = time.time()
        if now - self._last_status_log < self.cfg.runtime.log_interval:
            return
        print(
            f"[{info.stage}] MODE:{info.control_mode} "
            f"LR:{info.roll_cmd:+d} FB:{info.forward_cmd:+d} "
            f"UD:{info.throttle_cmd:+d} YAW:{info.yaw_cmd:+d} | "
            f"FPS:{self._fps:.1f}"
        )
        self._last_status_log = now

    # ------------------------------------------------------------------
    async def _cleanup(self) -> None:
        print("[App] Cleaning up...")
        self._stop_key_thread()
        self.drone.stop()
        await asyncio.sleep(0.1)  # allow the RC pump to flush zeros
        await self.drone.land()
        await self.drone.shutdown()
        if self._detect_pool is not None:
            self._detect_pool.shutdown(wait=False)
        cv2.destroyAllWindows()
        print("[App] Program finished")