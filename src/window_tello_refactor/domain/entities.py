"""Value objects shared across layers — no logic, no I/O."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple | None
    area: int
    confidence: float
    center: tuple | None
    corners: np.ndarray | None
    mask_polygon: np.ndarray | None


EMPTY_DETECTION = Detection(None, 0, 0.0, None, None, None)


@dataclass
class AlignCommand:
    """Output of the aligner for one frame."""

    yaw_error: float | None = None
    roll_error: float | None = None
    throttle_error: float | None = None
    area: int | None = None
    yaw_cmd: int = 0
    roll_cmd: int = 0
    throttle_cmd: int = 0
    forward_cmd: int = 0
    stage: str = "IDLE"
    yaw_centered: bool = False
    roll_centered: bool = False
    throttle_centered: bool = False
    pass_completed: bool = False
    control_mode: str = "IDLE"