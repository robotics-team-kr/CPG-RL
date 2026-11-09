# SPDX-License-Identifier: BSD-3-Clause
# Adapted from the local legged-gym MuJoCo Viser training monitor.
"""Read-only Viser monitoring of a selected training environment and learning progress.

The trainer copies at most one state per display interval. Rendering uses a
private MuJoCo model/data pair on a worker thread, with a single pending frame.
"""

import copy
import math
import threading
import time
from collections import deque

import mujoco
import numpy as np


class TrainingMonitor:
    def __init__(self, env, runner, cfg, *, task):
        import viser
        from mjviser import ViserMujocoScene

        if not math.isfinite(cfg.max_fps) or cfg.max_fps <= 0 or cfg.history_points < 2:
            raise ValueError("viewer.max_fps must be positive and history_points must be at least 2")
        self.env = env
        self.runner = runner
        self._interval = 1.0 / cfg.max_fps
        self._start = time.monotonic()
        self._last_capture = -math.inf
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending_frame = None
        self._stats = {}
        self._stats_version = 0
        self._shown_stats_version = -1
        self._reward_history = deque(maxlen=cfg.history_points)
        self._velocity_history = deque(maxlen=cfg.history_points)
        self._selected_env = None
        self._last_root = None
        self._thread = None
        self.server = None
        self._attached = False
        self._closed = False
        self._step_method = "step_wait" if callable(getattr(env, "step_wait", None)) else "step"
        self._original_step = getattr(env, self._step_method)
        self._original_log = runner.logger.log
        try:
            self.server = viser.ViserServer(host=cfg.host, port=cfg.port, label=f"Legged Gym · {task}", verbose=False)
            self.server.scene.set_up_direction("+z")
            # mjviser changes visualization properties; never pass the training model.
            self.model = copy.copy(env.get_monitor_model())
            self.data = mujoco.MjData(self.model)
            self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
            self.scene.camera_tracking_enabled = False
            self.scene.site_groups_visible = [False] * 6
            self.scene.geom_groups_visible = [True, True, True, False, False, False]
            self.body_axes = self.server.scene.add_frame(
                "/motion/body", axes_length=0.15, axes_radius=0.003, origin_radius=0.005
            )
            self.server.scene.add_label("/motion/body/front", "Front (+X)", position=(0.2, 0, 0))
            points = np.zeros((1, 2, 3), dtype=np.float32)
            self.command_arrow = self.server.scene.add_arrows(
                "/motion/command",
                points,
                (245, 158, 11),
                shaft_radius=0.006,
                head_radius=0.015,
                head_length=0.035,
                visible=False,
            )
            self.velocity_arrow = self.server.scene.add_arrows(
                "/motion/actual",
                points,
                (34, 197, 94),
                shaft_radius=0.006,
                head_radius=0.015,
                head_length=0.035,
                visible=False,
            )
            self._make_gui(task, cfg.max_fps)
            self._publish_frame(self._snapshot(0))

            @self.server.on_client_connect
            def on_connect(client):
                with self._lock:
                    root = self._last_root.copy() if self._last_root is not None else np.zeros(3)
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.position = root + np.array([2.5, -3.0, 1.5])
                client.camera.look_at = root

            setattr(env, self._step_method, self._step)
            runner.logger.log = self._log
            self._attached = True
            self._thread = threading.Thread(target=self._publish_loop, name="legged-gym-viser", daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    @property
    def url(self):
        host = self.server.get_host()
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.server.get_port()}"

    def _make_gui(self, task, max_fps):
        gui = self.server.gui
        gui.add_markdown(
            f"## {task.upper()} · Live training\nObserving sampled training actions, including exploration."
        )
        self.environment = gui.add_number("Environment", initial_value=0, min=0, max=self.env.num_envs - 1, step=1)
        self.follow = gui.add_checkbox("Follow robot", initial_value=True)
        self.freeze = gui.add_checkbox(
            "Freeze view", initial_value=False, hint="Training continues while the view is frozen."
        )
        self.progress = gui.add_markdown("Waiting for the first training iteration…")
        self.motion = gui.add_markdown("Waiting for a training frame…")
        self.cpg_status = gui.add_markdown("") if self.env.cfg.control.mode == "cpg" else None
        empty = np.array([0.0], dtype=np.float64)
        self.velocity_plot = gui.add_uplot(
            (empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="X · Forward velocity · m/s",
            scales={"x": {"time": False}},
            height=180,
        )
        self.lateral_velocity_plot = gui.add_uplot(
            (empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="Y · Lateral velocity · m/s",
            scales={"x": {"time": False}},
            height=180,
        )
        self.angular_velocity_plot = gui.add_uplot(
            (empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="Yaw angular velocity · rad/s (body Z)",
            scales={"x": {"time": False}},
            height=180,
        )
        self.reward_plot = gui.add_uplot(
            (empty, empty),
            ({"label": "Iteration"}, {"label": "Mean episode return", "stroke": "#60a5fa"}),
            title="Training reward",
            scales={"x": {"time": False}},
            height=180,
        )
        gui.add_markdown(
            f"{self.env.num_envs:,} environments · {self.runner.device} learner\n\nDisplay limit: {max_fps:g} FPS"
        )

    def _snapshot(self, index):
        """Called only on the trainer thread; all mutable arrays are copied."""
        frame = self.env.get_monitor_frame(index)
        frame["time"] = time.monotonic() - self._start
        if self.cpg_status is not None:
            cpg = self.env.cpg
            frame["cpg"] = np.stack(
                (cpg.r[index], cpg.mu[index], cpg.omega[index] / (2 * np.pi), cpg.theta[index], cpg.phi[index])
            ).copy()
            if hasattr(cpg, "gait_names"):
                frame["gait"] = cpg.gait_names[cpg.gait_ids[index]]
                frame["phase_bias"] = cpg.phase_bias[index].copy()
                frame["coupling_weights"] = cpg.coupling_weights.copy()
        return frame

    def _step(self, *args):
        result = self._original_step(*args)
        now = time.monotonic()
        if now - self._last_capture >= self._interval and not self._stop.is_set():
            self._last_capture = now
            if self.server.get_clients() and not self.freeze.value:
                index = int(np.clip(self.environment.value, 0, self.env.num_envs - 1))
                frame = self._snapshot(index)
                with self._lock:
                    self._pending_frame = frame  # Drop stale frames instead of slowing training.
        return result

    def _log(self, **kwargs):
        result = self._original_log(**kwargs)
        logger = self.runner.logger
        reward = float(np.mean(logger.rewbuffer)) if logger.rewbuffer else None
        length = float(np.mean(logger.lenbuffer)) * self.env.dt if logger.lenbuffer else None
        elapsed = kwargs["collect_time"] + kwargs["learn_time"]
        with self._lock:
            self._stats = {
                "iteration": kwargs["it"],
                "total": kwargs["total_it"],
                "reward": reward,
                "length": length,
                "lr": kwargs["learning_rate"],
                "fps": kwargs.get("collection_size_override", self.env.num_envs * self.runner.cfg["num_steps_per_env"])
                / max(elapsed, 1e-9),
                "collect": kwargs["collect_time"],
                "learn": kwargs["learn_time"],
            }
            if reward is not None:
                self._reward_history.append((kwargs["it"] + 1, reward))
            self._stats_version += 1
        return result

    def _publish_loop(self):
        try:
            while not self._stop.wait(self._interval):
                with self._lock:
                    frame, self._pending_frame = self._pending_frame, None
                    stats = self._stats.copy()
                    version = self._stats_version
                    history = list(self._reward_history)
                if frame is not None and not self.freeze.value:
                    self._publish_frame(frame)
                if stats and version != self._shown_stats_version:
                    self._publish_stats(stats, history)
                    self._shown_stats_version = version
        except Exception as error:
            self._stop.set()
            print(f"Training monitor stopped updating: {error}. Training continues.")

    def _publish_frame(self, frame):
        self.data.qpos[:] = frame["qpos"]
        self.data.qvel[:] = frame["qvel"]
        # Kinematics is sufficient for geometry and avoids another physics step.
        mujoco.mj_kinematics(self.model, self.data)
        root = frame["qpos"][:3].copy()
        with self._lock:
            previous = self._last_root
            self._last_root = root
        if self._selected_env != frame["env"]:
            self._velocity_history.clear()
            self._selected_env = frame["env"]
        self._velocity_history.append(
            (
                frame["time"],
                frame["command"][0],
                frame["velocity"][0],
                frame["command"][1],
                frame["velocity"][1],
                frame["command"][2],
                frame["velocity"][2],
            )
        )
        with self.server.atomic():
            self.scene.update_from_mjdata(self.data)
            if previous is not None and self.follow.value:
                delta = root - previous
                for client in self.server.get_clients().values():
                    client.camera.position = np.asarray(client.camera.position) + delta
                    client.camera.look_at = np.asarray(client.camera.look_at) + delta
            command, velocity = frame["command"], frame["velocity"]
            # Use the exact rendered body pose, not a world-axis interpretation
            # of the body-frame velocity commands. Both arrows share its frame.
            body_id = self.env.sim.root_id
            rotation = self.data.xmat[body_id].reshape(3, 3)
            origin = self.data.xpos[body_id]
            self.body_axes.position = origin + np.array([0, 0, 0.16])
            self.body_axes.wxyz = self.data.xquat[body_id]
            for arrow, vector, height in ((self.command_arrow, command, 0.25), (self.velocity_arrow, velocity, 0.32)):
                start = origin + np.array([0, 0, height])
                delta = rotation @ np.array([vector[0], vector[1], 0.0]) * 0.5
                arrow.points = np.array([[start, start + delta]], dtype=np.float32)
                arrow.visible = np.linalg.norm(delta) > 0.005
            contacts = " / ".join("contact" if value else "air" for value in frame["contacts"])
            self.motion.content = (
                f"**Environment {frame['env']}**\n\n"
                "**Body frame:** +X forward · +Y left · +Yaw left turn.\n\n"
                "**Orange:** command · **Green:** actual XY velocity (0.5 s scale).\n\n"
                "| | Command | Actual |\n|---|---:|---:|\n"
                f"| X / Forward m/s | {command[0]:.2f} | {velocity[0]:.2f} |\n"
                f"| Y / Lateral m/s | {command[1]:.2f} | {velocity[1]:.2f} |\n"
                f"| Yaw rad/s | {command[2]:.2f} | {velocity[2]:.2f} |\n\n"
                f"Feet (FR / FL / HR / HL): {contacts}"
            )
            if self.cpg_status is not None:
                rows = ["| Leg | r | μ | Hz | θ rad | φ rad |", "|---|---:|---:|---:|---:|---:|"]
                for leg, values in zip(("FR", "FL", "HR", "HL"), frame["cpg"].T):
                    rows.append(f"| {leg} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
                content = "**CPG state**\n\n" + "\n".join(rows)
                if "gait" in frame:
                    content += f"\n\n**Gait: {frame['gait']}** · FR / FL / HR / HL"
                    for label, matrix in (
                        ("W", frame["coupling_weights"]),
                        ("Φ / 2π", frame["phase_bias"] / (2 * np.pi)),
                    ):
                        content += f"\n\n{label}\n\n| | FR | FL | HR | HL |\n|---|---:|---:|---:|---:|\n"
                        for leg, row in zip(("FR", "FL", "HR", "HL"), matrix):
                            content += f"| {leg} | " + " | ".join(f"{value:.2f}" for value in row) + " |\n"
                self.cpg_status.content = content
            history = np.asarray(self._velocity_history, dtype=float).T
            self.velocity_plot.data = tuple(history[[0, 1, 2]])
            self.lateral_velocity_plot.data = tuple(history[[0, 3, 4]])
            self.angular_velocity_plot.data = tuple(history[[0, 5, 6]])

    def _publish_stats(self, stats, history):
        reward = "—" if stats["reward"] is None else f"{stats['reward']:.2f}"
        length = "—" if stats["length"] is None else f"{stats['length']:.1f} s"
        self.progress.content = (
            f"**Iteration {stats['iteration'] + 1:,} / {stats['total']:,}**\n\n"
            f"Episode return: **{reward}** · Length: **{length}**\n\n"
            f"{stats['fps']:,.0f} samples/s · LR {stats['lr']:.2g}\n\n"
            f"Collection {stats['collect']:.2f}s · Update {stats['learn']:.2f}s"
        )
        if history:
            self.reward_plot.data = tuple(np.asarray(history, dtype=float).T)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._attached:
            setattr(self.env, self._step_method, self._original_step)
            self.runner.logger.log = self._original_log
            self._attached = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.server is not None:
            self.server.stop()
