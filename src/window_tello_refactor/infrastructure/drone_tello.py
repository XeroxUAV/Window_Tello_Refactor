"""Tello adapter — implements DronePort via djitellopy."""

import asyncio
import time

from djitellopy import Tello

from src.window_tello_refactor.infrastructure.ports import DronePort


class TelloDrone(DronePort):
    def __init__(
        self,
        flight_mode: bool,
        min_battery: int = 20,
        rc_send_interval: float = 0.02,
    ):
        self.flight_mode = flight_mode
        self.min_battery = min_battery
        self.rc_send_interval = rc_send_interval

        self.tello = Tello()

        self._last_rc_log = 0.0
        self._log_interval = 0.5

        self._rc: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._rc_event: asyncio.Event | None = None
        self._rc_task: asyncio.Task | None = None
        self._closed = False

    def _get_event(self) -> asyncio.Event:
        if self._rc_event is None:
            self._rc_event = asyncio.Event()
        return self._rc_event

    # ------------------------------------------------------------------
    async def connect(self) -> int:
        await asyncio.to_thread(self.tello.connect)
        battery = await asyncio.to_thread(self.tello.get_battery)
        print(f"[Drone] Battery: {battery}%")
        if battery < self.min_battery:
            raise SystemExit(
                f"[SAFETY] Battery {battery}% < {self.min_battery}%"
            )
        return battery

    async def start_stream(self) -> None:
        await asyncio.to_thread(self.tello.streamon)
        print("[Drone] Video stream ON")

    async def takeoff(self) -> None:
        print("[Drone] Takeoff...")
        await asyncio.to_thread(self.tello.takeoff)
        print("[Drone] Takeoff successful")

    async def hover_kick(
        self, throttle: int = 50, up_time: float = 1.0, pause: float = 0.5
    ) -> None:
        print("[Drone] Initial hover kick ...")
        self.set_rc(0, 0, throttle, 0)
        await asyncio.sleep(up_time)
        self.set_rc(0, 0, 0, 0)
        await asyncio.sleep(pause)

    async def land(self) -> None:
        if not self.flight_mode:
            return
        try:
            await asyncio.to_thread(self.tello.land)
            print("[Drone] Landed")
        except Exception as e:
            print(f"[Drone] Land error: {e}")

    async def shutdown(self) -> None:
        self._closed = True
        if self._rc_task is not None:
            self._rc_task.cancel()
            try:
                await self._rc_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Drone] RC task error on shutdown: {e}")
            finally:
                self._rc_task = None
        try:
            await asyncio.to_thread(self.tello.streamoff)
        except Exception:
            pass

    def frame_reader(self):
        return self.tello.get_frame_read()

    # ------------------------------------------------------------------
    def set_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._rc = (int(lr), int(fb), int(ud), int(yaw))
        self._get_event().set()

    def stop(self) -> None:
        self.set_rc(0, 0, 0, 0)

    def start_rc_pump(self) -> None:
        if self._rc_task is None or self._rc_task.done():
            self._rc_task = asyncio.create_task(self._rc_pump(), name="rc-pump")

    async def _rc_pump(self) -> None:
        last_sent: tuple[int, int, int, int] | None = None
        event = self._get_event()

        while not self._closed:
            try:
                await asyncio.wait_for(
                    event.wait(), timeout=self.rc_send_interval
                )
                event.clear()
            except TimeoutError:
                pass

            cmd = self._rc
            if cmd == last_sent:
                continue

            lr, fb, ud, yaw = cmd

            now = time.time()
            if now - self._last_rc_log >= self._log_interval:
                print(f"RC -> LR:{lr:+d} FB:{fb:+d} UD:{ud:+d} YAW:{yaw:+d}")
                self._last_rc_log = now

            if self.flight_mode:
                try:
                    await asyncio.to_thread(
                        self.tello.send_rc_control, lr, fb, ud, yaw
                    )
                except Exception as e:
                    print(f"[Drone] RC error: {e}")

            last_sent = cmd