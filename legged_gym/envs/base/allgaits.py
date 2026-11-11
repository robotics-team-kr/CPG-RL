# SPDX-License-Identifier: MIT
"""AllGaits rhythm/pattern formation, https://arxiv.org/abs/2411.04787.

Equations (1)-(4): PPO modulates mu/omega; externally specified W and Phi
enforce a gait. Neither the matrices nor geometry are policy observations.
Mini Cheetah geometry ranges and periodic phase encoding are adaptations.
"""

import numpy as np

from .cpg import CPG

# Fig. 3 touchdown times as cycle fractions, ordered FR, FL, HR, HL.
# Left/right here refer to the robot (the figure is viewed from above).
GAIT_CONTACT_TIMES = {
    "walk": [0.0, 0.5, 0.75, 0.25],
    "amble": [0.0, 0.5, 0.8, 0.3],
    "trot": [0.0, 0.5, 0.5, 0.0],
    "pace": [0.0, 0.5, 0.0, 0.5],
    "bound": [0.0, 0.0, 0.5, 0.5],
    "pronk": [0.0, 0.0, 0.0, 0.0],
    "canter": [0.0, 0.7, 0.3, 0.0],
    "transverse_gallop": [0.6, 0.5, 0.1, 0.0],
    "rotary_gallop": [0.5, 0.6, 0.1, 0.0],
}


class AllGaitsCPG(CPG):
    def __init__(self, num_envs, cfg, dt):
        super().__init__(num_envs, cfg, dt)
        self.gait_names = tuple(cfg.gait_contact_times)
        times = np.asarray(list(cfg.gait_contact_times.values()), dtype=float)
        if times.shape != (len(self.gait_names), 4) or not len(times) or not np.isfinite(times).all():
            raise ValueError("Each gait needs four finite contact times (FR, FL, HR, HL)")
        if np.any((times < 0) | (times >= 1)):
            raise ValueError("Contact times must be cycle fractions in [0, 1)")
        if not np.isfinite(cfg.coupling_strength) or cfg.coupling_strength <= 0:
            raise ValueError("AllGaits requires positive finite coupling strength")
        # Later touchdown requires a lagging phase: theta_i(t) = 2*pi*(f*t - t_i).
        # A common pi shift sets touchdown at theta=pi without affecting Phi.
        self.gait_offsets = -2 * np.pi * times
        self.gait_matrices = self.gait_offsets[:, None, :] - self.gait_offsets[:, :, None]
        self.gait_ids = np.zeros(num_envs, dtype=int)
        self.phase_bias = np.zeros((num_envs, 4, 4))
        self.foot_offset = np.full((num_envs, 1), cfg.foot_offset)
        self.penetration = np.full((num_envs, 1), cfg.penetration)
        self.set_gaits(np.arange(num_envs), self.gait_ids)

    def set_gaits(self, ids, gait_ids):
        """Change Phi without resetting oscillator phase/amplitude or robot state."""
        ids, gait_ids = np.asarray(ids, dtype=int), np.asarray(gait_ids, dtype=int)
        if np.any((gait_ids < 0) | (gait_ids >= len(self.gait_names))):
            raise ValueError("Unknown gait index")
        self.gait_ids[ids] = gait_ids
        self.phase_bias[ids] = self.gait_matrices[gait_ids]

    def reset(self, ids, rng):
        super().reset(ids, rng)
        self.theta[ids] = self.gait_offsets[self.gait_ids[ids]] % (2 * np.pi)
        self.foot_offset[ids], self.penetration[ids] = self.cfg.foot_offset, self.cfg.penetration
        if self.cfg.randomize_geometry:
            self.foot_offset[ids] = rng.uniform(*self.cfg.foot_offset_range, (len(ids), 1))
            self.penetration[ids] = rng.uniform(*self.cfg.penetration_range, (len(ids), 1))

    def set_actions(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (len(self.r), 8) or not np.isfinite(actions).all():
            raise ValueError("AllGaits requires finite actions with shape (num_envs, 8)")
        # Reuse the amplitude/frequency scaling; AllGaits has no direction action.
        super().set_actions(np.concatenate((actions, np.zeros((len(self.r), 4))), axis=1))
        self.psi.fill(0)

    def feet(self):
        feet = super().feet()
        feet[:, :, 0] += self.foot_offset
        sine = np.sin(self.theta)
        feet[:, :, 2] = -self.height + np.where(sine > 0, self.clearance, self.penetration) * sine
        return feet

    def observations(self):
        return np.concatenate(
            (
                self.r - 1,
                self.r_dot / self.cfg.convergence,
                np.sin(self.theta),
                np.cos(self.theta),
                self.theta_dot / (2 * np.pi * self.cfg.frequency_range[1]),
            ),
            axis=1,
        )
