"""Narrow parameter objects the application layer consumes.

Keeps the application layer independent of the full AppConfig schema.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIDParams:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 100.0
    max_speed: int = 30
    threshold: float = 4.0
    exit_threshold: float = 0.0
    stable_count: int = 3
    deadzone: float = 0.0
    invert: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class SearchParams:
    yaw_speed: int = 25
    time_360: float = 15.0
    alternate: bool = True
    up_after_360: bool = True
    up_duration: float = 2.0
    up_throttle: int = 20
    max_up_count: int = 5


@dataclass(frozen=True)
class ApproachParams:
    speed: int = 12
    deadzone_yaw: float = 9.0
    deadzone_thr: float = 9.0
    deadzone_roll: float = 3.0
    center_max: int = 15
    area_ratio_threshold: float = 0.35
    max_time: float = 20.0


@dataclass(frozen=True)
class PassParams:
    forward_speed: int = 40
    extra_time: float = 1.25
    max_time: float = 15.0
    area_drop: float = 0.45


@dataclass(frozen=True)
class MinSpeeds:
    throttle: int = 10
    yaw: int = 7
    roll: int = 15


@dataclass(frozen=True)
class AlignerParams:
    yaw: PIDParams
    roll: PIDParams
    throttle: PIDParams
    search: SearchParams = field(default_factory=SearchParams)
    approach: ApproachParams = field(default_factory=ApproachParams)
    passing: PassParams = field(default_factory=PassParams)
    min_speeds: MinSpeeds = field(default_factory=MinSpeeds)
    smoothing_alpha: float = 0.3
    max_lost_frames: int = 25
    pid_output_smoothing: float = 0.3


def params_from_config(cfg) -> AlignerParams:
    """Adapter: full AppConfig -> narrow AlignerParams."""
    return AlignerParams(
        yaw=PIDParams(
            kp=cfg.yaw.kp,
            ki=cfg.yaw.ki,
            kd=cfg.yaw.kd,
            integral_limit=cfg.yaw.integral_limit,
            max_speed=cfg.yaw.max_speed,
            threshold=cfg.yaw.threshold,
            # exit threshold is 2x threshold in the original code
            exit_threshold=cfg.yaw.threshold * 2,
            stable_count=cfg.yaw.stable_count,
            deadzone=cfg.yaw.deadzone,
            invert=cfg.yaw.geometry_invert,
        ),
        roll=PIDParams(
            kp=cfg.roll.kp,
            ki=cfg.roll.ki,
            kd=cfg.roll.kd,
            integral_limit=cfg.roll.integral_limit,
            max_speed=cfg.roll.max_speed,
            threshold=cfg.roll.threshold,
            exit_threshold=cfg.roll.exit_threshold,
            stable_count=cfg.roll.stable_count,
            deadzone=cfg.roll.deadzone,
            invert=cfg.roll.invert,
            enabled=cfg.roll.enabled,
        ),
        throttle=PIDParams(
            kp=cfg.throttle.kp,
            ki=cfg.throttle.ki,
            kd=cfg.throttle.kd,
            integral_limit=cfg.throttle.integral_limit,
            max_speed=cfg.throttle.max_speed,
            threshold=cfg.throttle.threshold,
            # throttle exit is also 2x threshold in the original
            exit_threshold=cfg.throttle.threshold * 2,
            stable_count=cfg.throttle.stable_count,
            deadzone=cfg.min_speeds.throttle,
        ),
        search=SearchParams(
            yaw_speed=cfg.search.yaw_speed,
            time_360=cfg.search.time_360,
            alternate=cfg.search.alternate,
            up_after_360=cfg.search.up_after_360,
            up_duration=cfg.search.up_duration,
            up_throttle=cfg.search.up_throttle,
            max_up_count=cfg.search.max_up_count,
        ),
        approach=ApproachParams(
            speed=cfg.approach.speed,
            deadzone_yaw=cfg.approach.deadzone_yaw,
            deadzone_thr=cfg.approach.deadzone_thr,
            deadzone_roll=cfg.approach.deadzone_roll,
            center_max=cfg.approach.center_max,
            area_ratio_threshold=cfg.approach.area_ratio_threshold,
            max_time=cfg.approach.max_time,
        ),
        passing=PassParams(
            forward_speed=cfg.passing.forward_speed,
            extra_time=cfg.passing.extra_time,
            max_time=cfg.passing.max_time,
            area_drop=cfg.passing.area_drop,
        ),
        min_speeds=MinSpeeds(
            throttle=cfg.min_speeds.throttle,
            yaw=cfg.min_speeds.yaw,
            roll=cfg.min_speeds.roll,
        ),
        smoothing_alpha=cfg.runtime.smoothing_alpha,
        max_lost_frames=cfg.runtime.max_lost_frames,
        pid_output_smoothing=cfg.runtime.pid_output_smoothing,
    )