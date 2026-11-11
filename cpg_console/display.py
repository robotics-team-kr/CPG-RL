import csv
import sys

import mujoco
import numpy as np


class Viewer:
    def __init__(self, env, callback):
        import mujoco.viewer

        self.env = env
        self.model = env.sim.batch.get_model(0)
        self.data = env.sim.batch.get_data(0)
        mujoco.mj_forward(self.model, self.data)
        self.handle = mujoco.viewer.launch_passive(self.model, self.data, key_callback=callback)
        self.handle.cam.distance = 2.5
        self.handle.cam.elevation = -20

    def sync(self, row, source, reason=""):
        with self.handle.lock():
            self.env.sim.batch.get_data(0, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.handle.cam.lookat[:] = self.env.base_pos[0]
        self.handle.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "State\nInput\nStick target\nCommand\nLast action\nControls",
            f"{row['state']}{': ' + reason if reason else ''}\n{source}\n"
            + " / ".join(f"{row[key]:+.2f}" for key in ("input_vx", "input_vy", "input_wz")) + "\n"
            + " / ".join(f"{row[key]:+.2f}" for key in ("cmd_vx", "cmd_vy", "cmd_wz")) + "\n"
            + f"{row['last_action']}\nSticks move | A reset | B zero command | Start pause/resume",
        ))
        self.handle.sync()

    def close(self):
        self.handle.close()


def sample(runtime, controls, elapsed, steps):
    env = runtime.env
    velocity = env.get_tracking_velocity()[0]
    return {
        "time_s": steps * env.dt,
        "state": "FALLEN" if runtime.failed else "PAUSED" if controls.paused else "RUNNING",
        "last_action": controls.last_action,
        **dict(zip(("cmd_vx", "cmd_vy", "cmd_wz"), runtime.command)),
        **dict(zip(("vx", "vy", "wz"), velocity)),
        "height_m": float(env.base_height[0]),
        "rtf": steps * env.dt / max(elapsed, 1e-9),
        **{f"contact_{leg}": int(value) for leg, value in zip(("fr", "fl", "hr", "hl"), env._feet_contact[0])},
        **{f"frequency_{leg}": float(value) for leg, value in zip(("fr", "fl", "hr", "hl"), env.cpg.omega[0] / (2 * np.pi))},
        **{f"amplitude_{leg}": float(value) for leg, value in zip(("fr", "fl", "hr", "hl"), env.cpg.r[0])},
    }


class Dashboard:
    def __init__(self, title):
        self.title = title
        self.interactive = sys.stdout.isatty()

    def __enter__(self):
        if self.interactive:
            print("\x1b[?1049h\x1b[?25l", end="", flush=True)
        return self

    def draw(self, row, source, reason=""):
        def vector(keys):
            return " ".join(f"{row[key]:+6.2f}" for key in keys)
        if self.interactive:
            lines = [
                self.title,
                f"{row['state']}  t={row['time_s']:.2f}s  height={row['height_m']:.3f}m  RTF={row['rtf']:.2f}",
                f"Input: {source} | Last action: {row['last_action']}",
                "                  VX     VY     WZ",
                "Stick target  " + vector(("input_vx", "input_vy", "input_wz")),
                "Command       " + vector(("cmd_vx", "cmd_vy", "cmd_wz")),
                "Measured      " + vector(("vx", "vy", "wz")),
                "                  FR     FL     HR     HL",
                "Contact       " + vector([f"contact_{leg}" for leg in ("fr", "fl", "hr", "hl")]),
                "CPG Hz        " + vector([f"frequency_{leg}" for leg in ("fr", "fl", "hr", "hl")]),
                "CPG amplitude " + vector([f"amplitude_{leg}" for leg in ("fr", "fl", "hr", "hl")]),
                "W/S forward | A/D lateral | Q/E yaw | Space zero command",
                "P pause/resume | R reset | Esc quit | Gamepad: sticks move, A reset, B stop, Start pause/resume",
                f"Failure: {reason}. Press R to reset." if reason else "Velocity: m/s; yaw rate: rad/s. Commands are body-relative.",
            ]
            print("\x1b[H\x1b[J" + "\n".join(lines), end="", flush=True)
        else:
            print(f"{row['state']} t={row['time_s']:.2f} cmd=[{vector(('cmd_vx', 'cmd_vy', 'cmd_wz'))}] "
                  f"vel=[{vector(('vx', 'vy', 'wz'))}] height={row['height_m']:.3f} "
                  f"rtf={row['rtf']:.2f} input=[{source}] "
                  f"stick=[{vector(('input_vx', 'input_vy', 'input_wz'))}] action={row['last_action']}"
                  + (f" failure={reason}" if reason else ""), flush=True)

    def __exit__(self, *_):
        if self.interactive:
            print("\x1b[?25h\x1b[?1049l", end="", flush=True)


class Recorder:
    def __init__(self, path):
        self.stream = open(path, "x", newline="")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.stream.close()
