"""Central configuration for the Window PASS system."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    path: str = r"D:\Window_Tello\best.pt"
    confidence: float = 0.5
    inference_size: int = 640
    max_detections: int = 5


@dataclass
class SearchConfig:
    yaw_speed: int = 25
    time_360: float = 15.0
    alternate: bool = True
    up_after_360: bool = True
    up_duration: float = 2.0
    up_throttle: int = 20
    max_up_count: int = 5


@dataclass
class YawConfig:
    threshold: float = 4.0
    stable_count: int = 3
    max_speed: int = 35
    deadzone: float = 15.0
    kp: float = 0.4
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 100.0
    geometry_invert: bool = False


@dataclass
class RollConfig:
    enabled: bool = True
    threshold: float = 4.0
    exit_threshold: float = 8.0
    stable_count: int = 3
    invert: bool = False
    max_speed: int = 30
    deadzone: float = 3.0
    kp: float = 1.5
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 20.0


@dataclass
class ThrottleConfig:
    threshold: float = 13.0
    stable_count: int = 3
    kp: float = 0.4
    ki: float = 0.05
    kd: float = 0.0
    integral_limit: float = 100.0
    max_speed: int = 25


@dataclass
class ApproachConfig:
    speed: int = 12
    deadzone_yaw: int = 9
    deadzone_thr: int = 9
    deadzone_roll: float = 3.0
    center_max: int = 15
    area_ratio_threshold: float = 0.35
    max_time: float = 20.0


@dataclass
class PassConfig:
    forward_speed: int = 40
    extra_time: float = 1.25
    max_time: float = 15.0
    area_drop: float = 0.45


@dataclass
class CameraConfig:
    offset_x: int = 0
    offset_y: int = 125
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class MinSpeeds:
    throttle: int = 10
    yaw: int = 7
    roll: int = 15


@dataclass
class RuntimeConfig:
    flight_mode: bool = True
    min_battery: int = 20
    log_interval: float = 0.5
    min_mask_area: int = 2000
    smoothing_alpha: float = 0.3
    max_lost_frames: int = 25
    pid_output_smoothing: float = 0.3
    rc_send_interval: float = 0.02
    key_poll_interval: float = 0.05
    async_detect: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    yaw: YawConfig = field(default_factory=YawConfig)
    roll: RollConfig = field(default_factory=RollConfig)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    passing: PassConfig = field(default_factory=PassConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    min_speeds: MinSpeeds = field(default_factory=MinSpeeds)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)