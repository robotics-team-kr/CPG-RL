"""Opt-in, asynchronously scheduled CPG/reflex environment.

The legacy step() path is retained for identity checks. step_coop() additionally
supports pausing individual simulations while draining on-policy slow macros.
Actor velocity/contact are explicitly ideal simulated observations, optionally
corrupted; slip is zero in actors and ground truth only in rewards/critics.
"""

from copy import deepcopy

import numpy as np
import torch

from legged_gym.envs.base.cpg import inverse_kinematics
from legged_gym.utils.coop_events import ContactEvents

from .allgaits import MiniCheetahAllGaits
from .coop_config import CoopSettings


def numpy_action(value, shape):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite action with shape {shape}")
    return value


class MiniCheetahCoopAllGaits(MiniCheetahAllGaits):
    def __init__(self, cfg, settings=None, **kwargs):
        cfg = deepcopy(cfg)
        self.settings = deepcopy(settings or CoopSettings())
        self.settings.validate()
        if cfg.env.observation != "full":
            raise ValueError("Cooperative schema v1 requires a full 68D AllGaits base")
        if not np.isclose(cfg.sim.dt * cfg.control.decimation, 0.01):
            raise ValueError("Base and reflex control timestep must remain 0.01 s")
        if self.settings.terrain != "flat" and not self.settings.fixed_start:
            raise ValueError("Course terrain requires fixed_start for world +X traversal")
        cfg.coop_settings = self.settings
        if self.settings.terrain != "flat":
            cfg.asset.contact_all_terrain = True
        n = cfg.env.num_envs
        self.events = ContactEvents(n)
        self.sensor_rng = np.random.default_rng(cfg.seed + 8191)
        self.terrain_rng = np.random.default_rng(self.settings.terrain_seed)
        self.slow_raw = np.zeros((n, 8))
        self.slow = np.zeros((n, 8))
        self.reflex_raw = np.zeros((n, 20))
        self.gate = np.zeros((n, 4))
        self.foot_target = np.zeros((n, 4, 3))
        self.foot_filtered = np.zeros((n, 4, 3))
        self.phase_target = np.zeros((n, 4))
        self.phase_filtered = np.zeros((n, 4))
        self.summary = np.zeros((n, 17))  # events8, slip4(oracle-free zero), gates4, foot magnitude1
        self.slow_countdown = np.zeros(n, dtype=int)
        self.reflex_countdown = np.zeros(n, dtype=int)
        self.slow_change = np.zeros(n)
        self.foot_change = np.zeros(n)
        self.start_xy = np.zeros((n, 2))
        self.positive_energy = np.zeros(n)
        self.absolute_energy = np.zeros(n)
        self.path_length = np.zeros(n)
        self.intervention = np.zeros(n)
        self.tracking_squared = np.zeros((n, 3))
        self.settled_time = np.zeros(n)
        self.recovery_time = np.full(n, np.nan)
        self.step_positive = np.zeros(n)
        self.step_absolute = np.zeros(n)
        self.saturation = np.zeros(n)
        self.reach_projection = np.zeros(n)
        self.active = np.ones(n, dtype=bool)
        self.coop_contact_cursor = np.zeros(n, dtype=int)
        self.sensor_contact = np.zeros((n, 4), dtype=bool)
        self.contact_queue = np.zeros(
            (self.settings.contact_delay_steps * cfg.control.decimation + 1, n, 4), dtype=bool
        )
        self.target_queue = np.zeros((self.settings.actuator_delay_steps + 1, n, 12))
        self._sensor_ready = False
        super().__init__(cfg, **kwargs)
        self.xfrc = self.sim.batch.bind("xfrc_applied")
        dynamics_rng = np.random.default_rng(cfg.seed + 4093)
        mass_scale = np.full(n, self.settings.mass_scale)
        motor_scale = np.full(n, self.settings.motor_scale)
        friction = np.full(n, self.settings.terrain_friction)
        if self.settings.randomize_dynamics:
            mass_scale *= dynamics_rng.uniform(*self.settings.mass_range, n)
            motor_scale *= dynamics_rng.uniform(*self.settings.motor_range, n)
            friction *= dynamics_rng.uniform(*self.settings.friction_range, n) / 0.6
        if np.any(mass_scale != 1):
            self.sim.batch.expand("body_mass")[:] *= mass_scale[:, None]
            self.sim.batch.expand("body_inertia")[:] *= mass_scale[:, None, None]
            self.sim.batch.set_const()
        if np.any(motor_scale != 1):
            self.sim.batch.expand("actuator_forcerange")[:] *= motor_scale[:, None, None]
        if np.any(friction != 0.6):
            friction_array = self.sim.batch.expand("geom_friction")
            for i in range(self.sim.model.ngeom):
                name = self.sim.model.geom(i).name
                if name in self.sim.foot_geoms or name.startswith("course_"):
                    friction_array[:, i, 0] *= friction / self.settings.terrain_friction
                    if name in self.sim.foot_geoms and self.settings.terrain == "flat":
                        friction_array[:, i, 0] = friction
        if any(self.settings.com_offset):
            self.sim.batch.expand("body_ipos")[:, self.sim.root_id] += self.settings.com_offset
            self.sim.batch.set_const()
        self.mass_per_env = self.sim.model.body_mass.sum() * mass_scale
        self.model_digest = self.sim.asset_digest
        self._course_ids = np.array(
            [
                i
                for i in range(self.sim.model.ngeom)
                if self.sim.model.geom(i).name.startswith("course_") and self.sim.model.geom(i).name != "course_0"
            ],
            dtype=int,
        )
        self._randomize_course(np.arange(n))
        self._sense(self.active)
        self.compute_observations()

    def _randomize_course(self, ids):
        if not self.settings.randomize_terrain or not len(ids) or not len(self._course_ids):
            return
        if self.settings.terrain != "irregular":
            raise ValueError("Per-episode layout randomization currently requires irregular terrain")
        heights = self.terrain_rng.uniform(
            -self.settings.terrain_height, self.settings.terrain_height, (len(ids), len(self._course_ids))
        )
        pos = self.sim.batch.expand("geom_pos")
        size = self.sim.batch.expand("geom_size")
        pos[np.ix_(ids, self._course_ids, [2])] = ((heights - 0.20) / 2)[..., None]
        size[np.ix_(ids, self._course_ids, [2])] = ((heights + 0.20) / 2)[..., None]
        self.sim.batch.set_const()

    def _resample_gaits(self, ids):
        if self.settings.gait_names:
            names = self.settings.gait_names
            if any(name not in self.cpg.gait_names for name in names):
                raise ValueError("Unknown requested gait")
            choices = self.rng.integers(len(names), size=len(ids))
            self.cpg.set_gaits(ids, [self.cpg.gait_names.index(names[i]) for i in choices])
            self._next_gait[ids] = self.cfg.gait.resampling_time
        else:
            super()._resample_gaits(ids)
        self.events.guard[ids] = 0.07

    def reset_idx(self, ids):
        super().reset_idx(ids)
        ids = np.asarray(ids, dtype=int)
        if self.settings.fixed_start:
            self.sim.qpos[ids, :2] = 0
            self.sim.qpos[ids, 3:7] = [1, 0, 0, 0]
        for name in (
            "slow_raw",
            "slow",
            "reflex_raw",
            "gate",
            "foot_target",
            "foot_filtered",
            "phase_target",
            "phase_filtered",
            "summary",
            "slow_countdown",
            "reflex_countdown",
            "slow_change",
            "foot_change",
            "positive_energy",
            "absolute_energy",
            "path_length",
            "intervention",
            "tracking_squared",
            "settled_time",
            "step_positive",
            "step_absolute",
            "saturation",
            "reach_projection",
            "coop_contact_cursor",
            "sensor_contact",
        ):
            getattr(self, name)[ids] = 0
        self.recovery_time[ids] = np.nan
        self.start_xy[ids] = self.sim.qpos[ids, :2]
        self.contact_queue[:, ids] = False
        self.target_queue[:, ids] = self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)]
        self.events.reset(ids)
        if hasattr(self, "xfrc"):
            self.xfrc[ids] = 0
        if hasattr(self, "_course_ids"):
            self._randomize_course(ids)

    def _get_cpg_actions(self):
        if not self.settings.enabled or not self.settings.slow_enabled:
            return self.actions
        scales = np.array([self.settings.slow_scale[0]] * 4 + [self.settings.slow_scale[1]] * 4)
        return np.clip(self.actions + scales * self.slow, -1, 1)

    def _get_cpg_feet(self):
        if not self.settings.enabled or not self.settings.reflex_enabled:
            return self.cpg.step()
        alpha = -np.expm1(-self.cfg.sim.dt / self.settings.filter_tau)
        self.foot_filtered[self.active] += alpha * (self.foot_target - self.foot_filtered)[self.active]
        self.phase_filtered[self.active] += alpha * (self.phase_target - self.phase_filtered)[self.active]
        phase = self.phase_filtered if self.settings.phase_enabled else None
        return self.cpg.step(phase_rate=phase) + self.foot_filtered

    def _sense(self, active):
        raw = self.sim.foot_contacts[:, :, 0] > 0
        self.contact_queue[:-1, active] = self.contact_queue[1:, active]
        self.contact_queue[-1, active] = raw[active]
        contact = self.contact_queue[0, active].copy()
        if self.settings.contact_dropout:
            contact &= self.sensor_rng.random(contact.shape) >= self.settings.contact_dropout
        self.sensor_contact[active] = contact
        self._sensor_ready = True

    def compute_observations(self, ids=None):
        super().compute_observations(ids)
        if not self.settings.enabled or not self._sensor_ready:
            return
        clean = self.obs_buf["critic"]
        clean[:, 3:12] = torch.tensor(
            np.concatenate((self.base_lin_vel, self.base_ang_vel, self.projected_gravity), axis=1), dtype=torch.float32
        )
        clean[:, 12:16] = torch.tensor(self.sim.foot_contacts[:, :, 0] > 0, dtype=torch.float32)
        clean[:, 16:40] = torch.tensor(
            np.concatenate((self.dof_pos - self.default_dof_pos, self.dof_vel * 0.05), axis=1), dtype=torch.float32
        )
        # Both frozen base and residual actors receive exactly the same corrupt sensor channels.
        self.obs_buf["policy"][:, 12:16] = torch.as_tensor(self.sensor_contact, dtype=torch.float32)
        # Noise is sampled only here, not every time an actor requests an observation.
        if self.settings.velocity_noise:
            noise = self.sensor_rng.normal(0, self.settings.velocity_noise, (self.num_envs, 3))
            self.obs_buf["policy"][:, 3:6] += torch.tensor(noise, dtype=torch.float32)

    def observations_coop(self):
        base = self.obs_buf["policy"].numpy().copy()
        gait = np.eye(9)[self.cpg.gait_ids]
        events = np.concatenate((self.events.early, self.events.late), axis=1)
        # No oracle foot velocity in either actor.
        slip = np.zeros((self.num_envs, 4))
        previous_r = np.tanh(self.reflex_raw)[:, [0, 1, 2, 3, *range(8, 20)]]
        message_s = self.slow if self.settings.messages and self.settings.slow_to_reflex else np.zeros_like(self.slow)
        summary = self.summary.copy()
        if not self.settings.messages or not self.settings.reflex_to_slow:
            summary[:, 12:] = 0
        reflex = np.concatenate(
            (
                base,
                message_s,
                self.events.early - self.events.late,
                events,
                np.clip(self.events.since_transition / 0.5, 0, 1),
                slip,
                previous_r,
                gait,
            ),
            axis=1,
        )
        slow = np.concatenate((base, gait, summary, self.slow), axis=1)
        # Clean base critic features and executed past actions, never a newly sampled own action.
        clean = self.obs_buf["critic"].numpy()
        privileged = np.concatenate(
            (
                clean,
                self.base_height[:, None],
                self.sim.foot_contacts[:, :, 1:].reshape(self.num_envs, -1),
                self.feet_vel.reshape(self.num_envs, -1),
                self.slow,
                np.tanh(self.reflex_raw),
                self.foot_filtered.reshape(self.num_envs, -1),
                self.phase_filtered,
                self.slow_countdown[:, None] / self.settings.slow_interval,
                gait,
            ),
            axis=1,
        )
        assert reflex.shape[1] == 121 and slow.shape[1] == 102
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in (("base", base), ("slow", slow), ("reflex", reflex), ("critic", privileged))
        }

    def set_slow(self, raw, ids):
        ids = np.asarray(ids, dtype=int)
        raw = numpy_action(raw, (len(ids), 8))
        if not self.settings.slow_enabled:
            raw = np.zeros_like(raw)
        projected = np.tanh(raw) if self.settings.slow_enabled else np.zeros_like(raw)
        self.slow_change[ids] = np.square(projected - self.slow[ids]).sum(axis=1)
        self.slow_raw[ids], self.slow[ids] = raw, projected
        self.slow_countdown[ids] = self.settings.slow_interval

    def set_reflex(self, raw, ids):
        ids = np.asarray(ids, dtype=int)
        raw = numpy_action(raw, (len(ids), 20))
        if not self.settings.reflex_enabled:
            raw = np.zeros_like(raw)
        r = np.tanh(raw)
        gate = 1 / (1 + np.exp(-(3 * r[:, :4] - 1.5)))
        if not self.settings.reflex_enabled:
            gate[:] = 0
        foot = r[:, 8:].reshape(-1, 3, 4).transpose(0, 2, 1) * np.array(self.settings.foot_scale)
        foot *= gate[:, :, None]
        if not self.settings.spatial_enabled:
            foot[:] = 0
        self.foot_change[ids] = np.square((foot - self.foot_target[ids]) / self.settings.foot_scale).sum(axis=(1, 2))
        self.reflex_raw[ids] = raw
        self.gate[ids], self.foot_target[ids] = gate, foot
        self.phase_target[ids] = gate * self.settings.phase_limit * r[:, 4:8] if self.settings.phase_enabled else 0
        self.reflex_countdown[ids] = self.settings.reflex_interval

    def _physics(self, ids):
        # CPG integration is vectorized; inactive environments must not advance during macro draining.
        inactive = ~self.active
        cpg_saved = {
            k: v[inactive].copy()
            for k, v in vars(self.cpg).items()
            if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == self.num_envs
        }
        feet = self._get_cpg_feet()
        q = inverse_kinematics(feet, self.cfg.cpg.link_lengths) - self.sim.encoder_bias
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        targets = np.clip(q, limits[:, 0], limits[:, 1])
        self.saturation[self.active] += np.mean(q != targets, axis=1)[self.active] * self.cfg.sim.dt
        l1, l2, l3 = self.cfg.cpg.link_lengths
        d = np.sqrt(np.maximum(np.sum(feet * feet, axis=-1) - l1 * l1, 0))
        projected = (d >= l2 + l3 - 1e-6) | (d <= abs(l2 - l3) + 1e-6)
        self.reach_projection[self.active] += projected.mean(axis=1)[self.active] * self.cfg.sim.dt
        for k, value in cpg_saved.items():
            getattr(self.cpg, k)[inactive] = value
        self.target_queue[:-1, self.active] = self.target_queue[1:, self.active]
        self.target_queue[-1, self.active] = targets[self.active]
        targets = self.target_queue[0].copy()
        vel_before = self.sim.qvel[:, self.sim.qvel_ids].copy()
        self.sim.step(targets, ids)
        power = self.sim.actuator_force * vel_before
        self.step_positive[self.active] += np.maximum(power, 0).sum(axis=1)[self.active] * self.cfg.sim.dt
        self.step_absolute[self.active] += np.abs(power).sum(axis=1)[self.active] * self.cfg.sim.dt
        contacts = self.sim.foot_contacts[:, :, 0] > 0
        liftoff = (self.feet_contact_time > 0) & ~contacts
        self._liftoff_count[ids] += liftoff.sum(axis=1)[ids]
        self.feet_air_time[ids] = np.where(contacts[ids], 0, self.feet_air_time[ids] + self.cfg.sim.dt)
        self.feet_contact_time[ids] = np.where(contacts[ids], self.feet_contact_time[ids] + self.cfg.sim.dt, 0)
        self._feet_contact[ids] = contacts[ids]
        self._collision_history[self.coop_contact_cursor[ids], ids] = self.sim.history_contacts[ids, :, 1:]
        self.coop_contact_cursor[ids] = (self.coop_contact_cursor[ids] + 1) % len(self._collision_history)
        self._sense(self.active)
        self.events.update(self.sensor_contact, self.cpg.theta, self.cfg.sim.dt, self.active)

    def _team_reward(self):
        error = self.commands - np.column_stack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2]))
        tracking = (np.exp(-error * error / 0.25) * [0.75, 0.75, 0.50]).sum(axis=1)
        slip = np.square(self.feet_vel[:, :, :2]).sum(axis=2) * self._feet_contact
        effort = (
            0.020 * np.square(self.slow).mean(axis=1)
            + 0.025 * self.gate.mean(axis=1)
            + 0.015 * np.square(self.foot_filtered / self.settings.foot_scale).mean(axis=(1, 2))
            + 0.005 * self.slow_change
            + 0.003 * self.foot_change
        )
        reward = self.dt * (
            tracking
            - 2 * self.base_lin_vel[:, 2] ** 2
            - 0.05 * np.square(self.base_ang_vel[:, :2]).sum(axis=1)
            - 0.10 * slip.mean(axis=1)
            - self.settings.intervention_weight * effort
        )
        reward -= 0.003 * self.step_positive / (self.mass_per_env * 9.81 * 0.5)
        reward -= self.settings.failure_penalty * self.terminated_buf
        reward[~self.active] = 0
        self.tracking_squared[self.active] += np.square(error)[self.active] * self.dt
        return reward

    def step_coop(self, base_action, active=None):
        if not self.settings.enabled:
            if active is not None and not np.all(active):
                raise ValueError("The explicit legacy identity mode does not pause simulations")
            _, reward, done, info = super().step(base_action)
            return self.observations_coop(), reward, done, {**info, "identity_only": True}
        previous_observations = self.obs_buf.clone()
        self.active[:] = True if active is None else active
        ids = np.flatnonzero(self.active)
        if not len(ids):
            raise ValueError("At least one environment must be active")
        base_action = numpy_action(base_action, (self.num_envs, 8))
        self.last_actions[ids] = self.actions[ids]
        self.actions[ids] = np.clip(base_action[ids], -1, 1)
        self.cpg.set_actions(self._get_cpg_actions())
        self.step_positive[:] = 0
        self.step_absolute[:] = 0
        previous_xy = self.base_pos[:, :2].copy()
        # Physical force is applied over a specified interval, never as qvel edits.
        age = self.episode_length_buf.numpy() * self.dt
        pushing = (
            self.active
            & (age >= self.settings.push_start)
            & (age < self.settings.push_start + self.settings.push_duration)
        )
        self.xfrc[:] = 0
        self.xfrc[pushing, self.sim.root_id, :3] = self.settings.push_force
        for _ in range(self.cfg.control.decimation):
            self._physics(ids)
        self.episode_length_buf[ids] += 1
        self.common_step_counter += 1
        self.sim.forward(ids)
        self._refresh_state()
        self._foot_sensor_cache = self.sim.foot_contacts
        self.check_termination()
        progress = self.base_pos[:, 0] - self.start_xy[:, 0]
        complete = (
            self.settings.fixed_start & (progress >= self.settings.course_length) & (np.abs(self.base_pos[:, 1]) < 1.4)
        )
        failure = self.terminated_buf & self.active
        complete = complete & ~failure & self.active
        timeout = self.time_out_buf & ~failure & ~complete & self.active
        done = failure | complete | timeout
        self.positive_energy[ids] += self.step_positive[ids]
        self.absolute_energy[ids] += self.step_absolute[ids]
        self.path_length[ids] += np.linalg.norm(self.base_pos[ids, :2] - previous_xy[ids], axis=1)
        self.intervention[ids] += (
            np.linalg.norm(self.foot_filtered / self.settings.foot_scale, axis=2).sum(axis=1)[ids] * self.dt
        )
        reward = self._team_reward()
        alpha = -np.expm1(-self.dt / self.settings.summary_tau)
        summary = np.concatenate(
            (
                self.events.early,
                self.events.late,
                np.zeros((self.num_envs, 4)),
                self.gate,
                np.linalg.norm(self.foot_filtered / self.settings.foot_scale, axis=2).mean(axis=1, keepdims=True),
            ),
            axis=1,
        )
        self.summary[ids] += alpha * (summary[ids] - self.summary[ids])
        self.slow_countdown[ids] = np.maximum(0, self.slow_countdown[ids] - 1)
        self.reflex_countdown[ids] = np.maximum(0, self.reflex_countdown[ids] - 1)
        self.slow_change[ids] = 0
        self.foot_change[ids] = 0
        self.compute_observations()
        terminal_obs = self.observations_coop()
        trace = np.concatenate(
            (
                self.episode_length_buf.numpy()[:, None] * self.dt,
                self.base_pos,
                self.base_lin_vel,
                self.commands,
                self.gate,
                self.slow,
                self.foot_filtered.reshape(self.num_envs, -1),
                self.cpg.theta,
                self.sim.foot_contacts[:, :, 0] > 0,
                self.sensor_contact,
                np.sin(self.cpg.theta) <= 0,
                self.events.early,
                self.events.late,
                self.phase_filtered,
                self.positive_energy[:, None],
            ),
            axis=1,
        )
        error = np.abs(self.commands - np.column_stack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2])))
        stable = (error <= [0.1, 0.1, 0.15]).all(axis=1) & (age >= self.settings.push_start)
        self.settled_time[ids] = np.where(stable[ids], self.settled_time[ids] + self.dt, 0)
        recovered = (self.settled_time >= 0.5) & np.isnan(self.recovery_time) & self.active
        self.recovery_time[recovered] = age[recovered] + self.dt - self.settings.push_start - 0.5
        episodes = []
        for i in np.flatnonzero(done):
            seconds = float(self.episode_length_buf[i]) * self.dt
            episodes.append(
                dict(
                    env_id=int(i),
                    completed=bool(complete[i]),
                    failure=bool(failure[i]),
                    timeout=bool(timeout[i]),
                    duration=seconds,
                    progress=float(progress[i]),
                    energy_pos=float(self.positive_energy[i]),
                    energy_abs=float(self.absolute_energy[i]),
                    cot=float(self.positive_energy[i] / (self.mass_per_env[i] * 9.81 * progress[i]))
                    if complete[i] and progress[i] > 1
                    else float("nan"),
                    tracking_rmse=float(np.sqrt(self.tracking_squared[i].mean() / seconds)),
                    intervention=float(self.intervention[i]),
                    recovery_time=float(self.recovery_time[i]),
                    saturation=float(self.saturation[i] / seconds),
                    ik_projection=float(self.reach_projection[i] / seconds),
                    gait=self.cpg.gait_names[self.cpg.gait_ids[i]],
                )
            )
        render_state = None
        if getattr(self, "capture_render", False):
            render_state = (self.sim.qpos.copy(), self.sim.qvel.copy())
        self.reset_idx(np.flatnonzero(done))
        # Change requested gait before constructing the observation for the next action.
        ongoing = self.active & ~done
        self._next_gait[ongoing] -= self.dt
        if self.cfg.gait.name == "random":
            self._resample_gaits(np.flatnonzero(ongoing & (self._next_gait <= 1e-9)))
        self._next_command[ongoing] -= self.dt
        self._resample_commands(np.flatnonzero(ongoing & (self._next_command <= 0)))
        self._update_commands()
        self.sim.forward(ids)
        self._refresh_state()
        # Preserve the sampled observation on ordinary steps; rebuild only after reset/gait/command changes.
        sensed_base = self.obs_buf["policy"][:, 3:16].clone()
        self.compute_observations()
        keep = torch.as_tensor(~done)
        self.obs_buf["policy"][keep, 3:16] = sensed_base[keep]
        for key in ("policy", "critic"):
            self.obs_buf[key][~self.active] = previous_observations[key][~self.active]
        self.xfrc[:] = 0
        return (
            self.observations_coop(),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done),
            {
                "time_outs": torch.tensor(timeout),
                "terminal": terminal_obs,
                "episodes": episodes,
                "active": self.active.copy(),
                "trace": trace,
                "render_state": render_state,
            },
        )

    def state_dict(self):
        def arrays(obj):
            return {
                k: v.copy() if isinstance(v, np.ndarray) else v.clone()
                for k, v in vars(obj).items()
                if isinstance(v, (np.ndarray, torch.Tensor))
            }

        fields = ("qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart", "xfrc_applied", "qfrc_applied")
        return dict(
            env=arrays(self),
            cpg=arrays(self.cpg),
            events=arrays(self.events),
            sim={name: self.sim.batch.bind(name).copy() for name in fields},
            rng=deepcopy(self.rng.bit_generator.state),
            sensor_rng=deepcopy(self.sensor_rng.bit_generator.state),
            terrain_rng=deepcopy(self.terrain_rng.bit_generator.state),
            terrain_model={name: self.sim.batch.expand(name).copy() for name in ("geom_pos", "geom_size")}
            if self.settings.randomize_terrain
            else {},
            scalars={k: v for k, v in vars(self).items() if isinstance(v, (int, float, bool))},
            common_step_counter=self.common_step_counter,
            observations=self.obs_buf.clone(),
        )

    def load_state_dict(self, state):
        for obj, name in ((self, "env"), (self.cpg, "cpg"), (self.events, "events")):
            for key, value in state[name].items():
                target = getattr(obj, key, None)
                if target is None:
                    setattr(obj, key, value.copy() if isinstance(value, np.ndarray) else value.clone())
                elif isinstance(target, torch.Tensor):
                    target.copy_(value)
                else:
                    target[...] = value
        for name, value in state["sim"].items():
            self.sim.batch.bind(name)[...] = value
        self.rng.bit_generator.state = state["rng"]
        self.sensor_rng.bit_generator.state = state["sensor_rng"]
        self.terrain_rng.bit_generator.state = state["terrain_rng"]
        for name, value in state["terrain_model"].items():
            self.sim.batch.expand(name)[...] = value
        if state["terrain_model"]:
            self.sim.batch.set_const()
        for key, value in state["scalars"].items():
            setattr(self, key, value)
        self.common_step_counter = state["common_step_counter"]
        self.sim.forward()
        self._refresh_state()
        self.obs_buf = state["observations"].clone()
        self._foot_sensor_cache = None
