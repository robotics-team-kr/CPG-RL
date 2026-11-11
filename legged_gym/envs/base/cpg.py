# SPDX-License-Identifier: MIT
# Inspired by CPG-RL, Copyright (c) 2024 EPFL BioRob; see licenses/CPG_RL_LICENSE.
"""Bellegarda & Ijspeert (2022), https://arxiv.org/abs/2211.00458, Eqs. (3)-(8).

Order everywhere: FR, FL, HR, HL. Actions are [mu(4), omega(4), psi(4)].
Frequencies in configuration are Hz; internal phase rates are rad/s.
The critically damped amplitude ODE has an exact constant-input update, avoiding
the old implementation's startup transient and timestep-dependent integration.
"""

import numpy as np


def inverse_kinematics(feet, lengths=(0.062, 0.209, 0.18)):
    """Torso-aligned XYZ at each abduction origin -> (abad, hip, knee).

    Input shape is (num_envs, 4, 3), ordered FR, FL, HR, HL. Targets refer to
    the foot sphere centers (MJCF foot sites), not the ground contact points.
    The input axes do NOT rotate with the actuated abduction body.

    For roll a, hip h, knee k and side s=(-1,+1,-1,+1), the MJCF has
    +X abduction and -Y hip/knee axes. Its forward kinematics is:
        D = l2*cos(h) + l3*cos(h+k)
        x = l2*sin(h) + l3*sin(h+k)
        y = s*l1*cos(a) + D*sin(a)
        z = s*l1*sin(a) - D*cos(a)
    We select the downward-leg branch D >= 0 and negative knee angle.
    Thus the hip solution uses atan2(+x, D), consistent with the -Y axes.

    Radial projection and numerical clamps keep unreachable inputs finite;
    they are not a joint-limit-aware Cartesian projection. The environment
    clips joint angles separately, which can change the achieved foot position.
    """
    l1, l2, l3 = lengths
    feet = np.asarray(feet, dtype=float)
    x, y, z = np.moveaxis(feet, -1, 0)
    side = np.array([-1, 1, -1, 1])
    planar_z = np.sqrt(np.maximum(y * y + z * z - l1 * l1, 1e-10))
    roll = np.arctan2(y, -z) - np.arctan2(side * l1, planar_z)
    distance = np.sqrt(x * x + planar_z * planar_z)
    scale = np.clip(distance, abs(l2 - l3) + 1e-6, l2 + l3 - 1e-6) / distance
    x, planar_z = x * scale, planar_z * scale
    cosine = np.clip((x * x + planar_z * planar_z - l2 * l2 - l3 * l3) / (2 * l2 * l3), -1, 1)
    knee = -np.arccos(cosine)
    hip = np.arctan2(x, planar_z) - np.arctan2(l3 * np.sin(knee), l2 + l3 * np.cos(knee))
    return np.stack((roll, hip, knee), axis=-1).reshape(feet.shape[0], 12)


class CPG:
    def __init__(self, num_envs, cfg, dt):
        self.cfg, self.dt = cfg, dt
        self.r = np.ones((num_envs, 4))
        self.r_dot = np.zeros_like(self.r)
        self.theta = np.zeros_like(self.r)
        self.phi = np.zeros_like(self.r)
        self.mu = np.ones_like(self.r)
        self.omega = np.zeros_like(self.r)
        self.psi = np.zeros_like(self.r)
        self.theta_dot = np.zeros_like(self.r)
        self.height = np.full((num_envs, 1), cfg.height)
        self.clearance = np.full((num_envs, 1), cfg.clearance)
        self.offsets = np.array([0, np.pi, np.pi, 0])
        self.phase_bias = self.offsets[None, :] - self.offsets[:, None]
        self.coupling_weights = cfg.coupling_strength * (np.ones((4, 4)) - np.eye(4))

    def reset(self, ids, rng):
        self.r[ids], self.mu[ids] = 1.0, 1.0
        self.theta[ids] = self.offsets
        for value in (self.r_dot, self.phi, self.omega, self.psi, self.theta_dot):
            value[ids] = 0
        self.height[ids], self.clearance[ids] = self.cfg.height, self.cfg.clearance
        if self.cfg.randomize_geometry:
            self.height[ids] = rng.uniform(*self.cfg.height_range, (len(ids), 1))
            self.clearance[ids] = rng.uniform(*self.cfg.clearance_range, (len(ids), 1))

    def set_actions(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (len(self.r), 12) or not np.isfinite(actions).all():
            raise ValueError("CPG requires finite actions with shape (num_envs, 12)")
        a = np.clip(actions, -1, 1)
        for dest, block, bounds in (
            (self.mu, a[:, :4], self.cfg.amplitude_range),
            (self.omega, a[:, 4:8], self.cfg.frequency_range),
            (self.psi, a[:, 8:12], self.cfg.direction_rate_range),
        ):
            dest[:] = bounds[0] + (block + 1) * 0.5 * (bounds[1] - bounds[0])
        self.omega *= 2 * np.pi
        self.psi *= 2 * np.pi

    def step(self, phase_rate=None):
        # r'' + a r' + a^2/4 (r-mu) = 0, exact over one physics tick.
        k = self.cfg.convergence / 2
        error = self.r - self.mu
        b = self.r_dot + k * error
        decay = np.exp(-k * self.dt)
        self.r[:] = self.mu + (error + b * self.dt) * decay
        self.r_dot[:] = (self.r_dot - k * b * self.dt) * decay
        self.theta_dot[:] = self.omega
        if self.cfg.coupling_strength:
            delta = self.theta[:, None, :] - self.theta[:, :, None] - self.phase_bias
            # Coupling affects phase derivative, never the policy's intrinsic omega.
            self.theta_dot += (self.coupling_weights * self.r[:, None, :] * np.sin(delta)).sum(axis=2)
        if phase_rate is not None:
            phase_rate = np.asarray(phase_rate)
            if phase_rate.shape != self.theta.shape or not np.isfinite(phase_rate).all():
                raise ValueError("Phase rate must be finite and match oscillator shape")
            self.theta_dot += phase_rate
        self.theta[:] = (self.theta + self.theta_dot * self.dt) % (2 * np.pi)
        self.phi[:] = (self.phi + self.psi * self.dt + np.pi) % (2 * np.pi) - np.pi
        return self.feet()

    def feet(self):
        step = -self.cfg.step_length * (self.r - 1) * np.cos(self.theta)
        sine = np.sin(self.theta)
        x = step * np.cos(self.phi)
        y = step * np.sin(self.phi) + np.array([-1, 1, -1, 1]) * self.cfg.link_lengths[0]
        z = -self.height + np.where(sine > 0, self.clearance, self.cfg.penetration) * sine
        return np.stack((x, y, z), axis=-1)

    def observations(self):
        # Periodic encoding avoids angle-wrap discontinuities (an adaptation of the paper).
        return np.concatenate(
            (
                self.r - 1,
                self.r_dot / self.cfg.convergence,
                np.sin(self.theta),
                np.cos(self.theta),
                self.theta_dot / (2 * np.pi * 4.5),
                np.sin(self.phi),
                np.cos(self.phi),
                self.psi / (2 * np.pi * 1.5),
            ),
            axis=1,
        )
