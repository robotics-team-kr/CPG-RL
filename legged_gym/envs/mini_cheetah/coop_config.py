"""Opt-in research settings; legacy AllGaits configurations are never changed."""

from dataclasses import asdict, dataclass

from .config import MiniCheetahAllGaitsCfg


@dataclass
class CoopSettings:
    enabled: bool = True
    slow_enabled: bool = True
    reflex_enabled: bool = True
    messages: bool = True
    slow_to_reflex: bool = True
    reflex_to_slow: bool = True
    phase_enabled: bool = False
    spatial_enabled: bool = True
    slow_interval: int = 5
    reflex_interval: int = 1
    slow_scale: tuple = (0.10, 0.15)
    foot_scale: tuple = (0.025, 0.015, 0.030)
    phase_limit: float = 3.141592653589793
    filter_tau: float = 0.02
    summary_tau: float = 0.20
    fixed_start: bool = False
    gait_names: tuple = ()
    course_length: float = 4.0
    terrain: str = "flat"
    terrain_seed: int = 0
    randomize_terrain: bool = False
    terrain_height: float = 0.025
    terrain_friction: float = 0.6
    contact_delay_steps: int = 0
    contact_dropout: float = 0.0
    velocity_noise: float = 0.0
    push_start: float = 3.0
    push_duration: float = 0.10
    push_force: tuple = (0.0, 0.0, 0.0)
    actuator_delay_steps: int = 0
    mass_scale: float = 1.0
    motor_scale: float = 1.0
    randomize_dynamics: bool = False
    mass_range: tuple = (0.9, 1.1)
    motor_range: tuple = (0.9, 1.1)
    friction_range: tuple = (0.4, 1.2)
    com_offset: tuple = (0.0, 0.0, 0.0)
    intervention_weight: float = 1.0
    failure_penalty: float = 10.0

    def validate(self):
        import math

        for value in (self.slow_interval, self.reflex_interval, self.contact_delay_steps, self.actuator_delay_steps):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Decision/delay steps must be integers")
        if self.slow_interval < 1 or self.reflex_interval < 1:
            raise ValueError("Decision intervals must be positive")
        if self.reflex_interval != 1 and self.slow_interval != self.reflex_interval:
            raise ValueError("Slow reflex currently supports the same-rate control only")
        if self.terrain not in ("flat", "raised", "lowered", "irregular", "friction"):
            raise ValueError("Unknown terrain")
        if self.filter_tau <= 0 or self.summary_tau <= 0 or self.course_length <= 0:
            raise ValueError("Time constants and course length must be positive")
        if not 0 <= self.contact_dropout <= 1 or self.contact_delay_steps < 0 or self.actuator_delay_steps < 0:
            raise ValueError("Invalid sensor/actuator corruption")
        if min(self.mass_scale, self.motor_scale, self.terrain_friction) <= 0:
            raise ValueError("Mass, motor scale and friction must be positive")
        for bounds in (self.mass_range, self.motor_range, self.friction_range):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("Dynamics ranges must be positive ordered pairs")
        if len(self.foot_scale) != 3 or min(self.foot_scale) <= 0 or len(self.slow_scale) != 2:
            raise ValueError("Invalid residual scales")
        if len(self.push_force) != 3 or len(self.com_offset) != 3:
            raise ValueError("Force and CoM offset require three axes")
        if self.terrain_height < 0 or self.velocity_noise < 0 or self.push_duration < 0:
            raise ValueError("Negative terrain/noise/duration")
        if self.terrain_height >= 0.19:
            raise ValueError("Box courses support terrain_height < 0.19 m")
        for value in asdict(self).values():
            seq = value if isinstance(value, (list, tuple)) else [value]
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in seq):
                raise ValueError("Nonfinite research setting")


class MiniCheetahCoopCfg(MiniCheetahAllGaitsCfg):
    """Separate task defaults; the training CLI restores the base snapshot first."""

    pass
