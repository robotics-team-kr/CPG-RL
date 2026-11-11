"""Single-policy training with externally selected oscillator coupling matrices."""

import numpy as np

from legged_gym.envs.base.allgaits import AllGaitsCPG

from .mini_cheetah import MiniCheetah


class MiniCheetahAllGaits(MiniCheetah):
    cpg_type = AllGaitsCPG

    def __init__(self, cfg, **kwargs):
        if cfg.env.num_actions != 8 or cfg.control.mode != "cpg":
            raise ValueError("AllGaits requires 8 actions and CPG control")
        if cfg.gait.name != "random" and cfg.gait.name not in cfg.cpg.gait_contact_times:
            raise ValueError(f"Unknown gait: {cfg.gait.name}")
        if not np.isfinite(cfg.gait.resampling_time) or cfg.gait.resampling_time <= 0:
            raise ValueError("Gait resampling time must be finite and positive")
        self._next_gait = np.zeros(cfg.env.num_envs)
        super().__init__(cfg, **kwargs)

    def _resample_gaits(self, ids):
        if not len(ids):
            return
        if self.cfg.gait.name == "random":
            choices = self.rng.integers(len(self.cpg.gait_names), size=len(ids))
        else:
            choices = np.full(len(ids), self.cpg.gait_names.index(self.cfg.gait.name))
        self.cpg.set_gaits(ids, choices)
        self._next_gait[ids] = self.cfg.gait.resampling_time

    def reset_idx(self, ids):
        # Select the reset gait before CPG.reset initializes its relative phases.
        self._resample_gaits(ids)
        super().reset_idx(ids)

    def _before_physics(self):
        if self.cfg.gait.name == "random":
            self._resample_gaits(np.flatnonzero(self._next_gait <= 1e-9))
            self._next_gait -= self.dt
        super()._before_physics()

    def _reward_lateral_vertical_velocity(self):
        return np.square(self.base_lin_vel[:, 1:]).sum(axis=1)

    def _reward_all_body_ang_vel(self):
        return np.square(self.base_ang_vel).sum(axis=1)

    def _reward_power(self):
        return np.abs((self.sim.actuator_force * self.dof_vel).sum(axis=1))
