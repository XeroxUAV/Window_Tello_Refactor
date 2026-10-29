"""Ports (interfaces) owned by the inner layers.

Outer layers (infrastructure / presentation) provide adapters that
implement these. The domain & application layers only ever depend on
these abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, Protocol

from domain.entities import Detection


class DetectorPort(Protocol):
    def detect(self, frame) -> Detection: ...
    def annotate(self, frame, detection: Detection) -> None: ...


class FrameReaderPort(Protocol):
    @property
    def frame(self): ...


class DronePort(ABC):
    @abstractmethod
    async def connect(self) -> int: ...

    @abstractmethod
    async def start_stream(self) -> None: ...

    @abstractmethod
    async def takeoff(self) -> None: ...

    @abstractmethod
    async def hover_kick(self, *args, **kwargs) -> None: ...

    @abstractmethod
    async def land(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def frame_reader(self) -> FrameReaderPort: ...

    @abstractmethod
    def set_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def start_rc_pump(self) -> None: ...


class HUDWriterPort(Protocol):
    def camera_center(self, frame_shape) -> tuple[int, int]: ...
    def draw(self, frame, info, **kwargs) -> None: ...