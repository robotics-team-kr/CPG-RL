"""Policy -> CPG -> Cartesian feet -> IK -> torque-limited PD -> MuJoCo."""

import numpy as np
import torch
from tensordict import TensorDict

from legged_gym.envs.base.cpg import CPG, inverse_kinematics
from legged_gym.envs.base.legged_robot import LeggedRobot


class MiniCheetah(LeggedRobot):
    cpg_type = CPG

    def __init__(self, cfg, **kwargs):
        if cfg.control.mode not in ("cpg", "joint") or cfg.env.observation not in ("full", "medium"):
            raise ValueError("Expected cpg/joint control and full/medium observations")
        self.cpg = self.cpg_type(cfg.env.num_envs, cfg.cpg, cfg.sim.dt)
        self.cpg.reset(np.arange(cfg.env.num_envs), np.random.default_rng(cfg.seed))
        feet = self.cpg.feet()[:1].copy()
        # Nominal posture must not depend on sampled geometry or batch size.
        feet[:, :, 0] = getattr(cfg.cpg, "foot_offset", 0.0)
        feet[:, :, 2] = -cfg.cpg.height
        nominal = inverse_kinematics(feet, cfg.cpg.link_lengths)[0]
        cfg.init_state.default_joint_angles = dict(zip(cfg.asset.joint_names, nominal.tolist()))
        self.previous_dof_vel = np.zeros((cfg.env.num_envs, 12))
        super().__init__(cfg, **kwargs)

    def _before_physics(self):
        self.previous_dof_vel[:] = self.dof_vel
        if self.cfg.control.mode == "cpg":
            self.cpg.set_actions(self.actions)

    def _compute_targets(self):
        if self.cfg.control.mode == "joint":
            return super()._compute_targets()
        targets = inverse_kinematics(self.cpg.step(), self.cfg.cpg.link_lengths) - self.sim.encoder_bias
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        return np.clip(targets, limits[:, 0], limits[:, 1])

    def reset_idx(self, ids):
        super().reset_idx(ids)
        ids = np.asarray(ids, dtype=int)
        self.cpg.reset(ids, self.rng)
        self.previous_dof_vel[ids] = 0
        # Start with all feet on the ground even when geometry is randomized.
        feet = self.cpg.feet()[ids].copy()
        feet[:, :, 2] = -self.cpg.height[ids]
        self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] = inverse_kinematics(feet, self.cfg.cpg.link_lengths)
        self.sim.qpos[ids, 2] = self.cpg.height[ids, 0] + 0.0175

    def check_termination(self):
        super().check_termination()
        # The paper terminates on torso/thigh contact; shank contacts are allowed.
        columns = [
            i for i, name in enumerate(self.sim.nonfoot_geoms) if name.startswith("body_") or name.startswith("thigh_")
        ]
        force = np.linalg.norm(self._collision_history[:, :, columns], axis=-1)
        self._contact_failure = force.max(axis=(0, 2)) > self.cfg.termination.illegal_contact_threshold
        self.terminated_buf = self._tilt_failure | self._contact_failure
        self.reset_buf = self.terminated_buf | self.time_out_buf

    def compute_observations(self, ids=None):
        # Command(3), body(9), contacts(4), optional joints(24)/last action,
        # and CPG feedback: full/medium CPG = 84/48, AllGaits = 68/36.
        body = np.concatenate((self.base_lin_vel, self.base_ang_vel, self.projected_gravity), axis=1)
        joints = np.concatenate((self.dof_pos - self.default_dof_pos, self.dof_vel * 0.05), axis=1)
        if self.cfg.noise.add_noise:
            body[:, 3:6] += self.rng.uniform(-1, 1, (self.num_envs, 3)) * self.cfg.noise.ang_vel
            body[:, 6:9] += self.rng.uniform(-1, 1, (self.num_envs, 3)) * self.cfg.noise.gravity
            joints[:, :12] += self.rng.uniform(-1, 1, (self.num_envs, 12)) * self.cfg.noise.joint_pos
            joints[:, 12:] += self.rng.uniform(-1, 1, (self.num_envs, 12)) * self.cfg.noise.joint_vel * 0.05
        parts = [self.commands, body, self.sim.foot_contacts[:, :, 0] > 0]
        if self.cfg.env.observation == "full":
            parts.extend((joints, self.actions))
        if self.cfg.control.mode == "cpg":
            parts.append(self.cpg.observations())
        values = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite Mini Cheetah observation")
        obs = torch.from_numpy(values)
        self.obs_buf = TensorDict({"policy": obs, "critic": obs.clone()}, batch_size=[self.num_envs])

    def _reward_tracking_x(self):
        return np.exp(-np.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) / 0.25)

    def _reward_tracking_y(self):
        return np.exp(-np.square(self.commands[:, 1] - self.base_lin_vel[:, 1]) / 0.25)

    def _reward_tracking_yaw(self):
        return np.exp(-np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / 0.25)

    def _reward_vertical_velocity(self):
        return np.square(self.base_lin_vel[:, 2])

    def _reward_work(self):
        torque = self.sim.actuator_force
        return np.abs((torque * (self.dof_vel - self.previous_dof_vel)).sum(axis=1))
