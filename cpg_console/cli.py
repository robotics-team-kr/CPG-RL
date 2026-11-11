import argparse
import os
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np
import torch

from .controls import Controls, Joystick, Keyboard
from .display import Dashboard, Recorder, Viewer, sample
from .runtime import Runtime, find_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MuJoCo console for CPG-RL checkpoints")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="model_*.pt with adjacent run_config.json")
    source.add_argument("--demo", choices=("stand", "trot"), help="Untrained physics demo")
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="Find the latest run/checkpoint here")
    parser.add_argument("--task", help="Inferred from checkpoint; demos default to mini_cheetah_cpg")
    parser.add_argument("--gait", help="AllGaits gait name or random")
    parser.add_argument("--headless", action="store_true", help="Console only, no native viewer")
    parser.add_argument("--steps", type=int, help="Exit after this many policy steps")
    parser.add_argument("--fast", action="store_true", help="Run without real-time pacing (headless only)")
    parser.add_argument("--velocity", type=float, nargs=3, default=[0, 0, 0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--acceleration", type=float, default=1.0, help="Command slew limit per second")
    parser.add_argument("--device", default="cpu", help="Policy device: cpu, mps, cuda or auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joystick", type=int, nargs="?", const=0, help="Gamepad device index (default 0)")
    parser.add_argument("--axes", type=int, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="Override automatic stick mapping with raw axis indices")
    parser.add_argument("--axis-signs", type=int, nargs=3, default=[-1, -1, -1], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--hold-to-move", action="store_true", help="Require holding mapped LB to move (default: off)")
    parser.add_argument("--deadman-button", type=int, help="Require this raw enable button; -1 disables")
    parser.add_argument("--reset-button", type=int, help="Raw reset button override (default: mapped A); -1 disables")
    parser.add_argument("--stop-button", type=int, help="Raw stop button override (default: mapped B); -1 disables")
    parser.add_argument("--pause-button", type=int, help="Raw pause toggle override (default: mapped Start); -1 disables")
    parser.add_argument("--csv", type=Path, help="Write telemetry to a new CSV file")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seed < 0 or (args.joystick is not None and args.joystick < 0):
        parser.error("seed and joystick index must be nonnegative")
    if not np.isfinite(args.velocity).all():
        parser.error("--velocity must be finite")
    if not np.isfinite(args.acceleration) or args.acceleration <= 0:
        parser.error("--acceleration must be finite and positive")
    if (not 0 <= args.deadzone < 1 or (args.axes is not None and min(args.axes) < 0)
            or any(v not in (-1, 1) for v in args.axis_signs)):
        parser.error("Use deadzone in [0, 1), nonnegative axes, and axis signs -1 or 1")
    for name in ("deadman_button", "reset_button", "stop_button", "pause_button"):
        value = getattr(args, name)
        if value is not None and value < -1:
            parser.error(f"--{name.replace('_', '-')} must be -1 or a button index")
    if args.fast and (not args.headless or args.joystick is not None):
        parser.error("--fast requires --headless without --joystick")
    if args.joystick is not None and any(args.velocity):
        parser.error("--velocity and --joystick cannot be combined")
    if args.demo and any(args.velocity):
        parser.error("Demos do not track velocity; select a checkpoint for velocity commands")
    return args


def ensure_mjpython(args):
    if args.headless or sys.platform != "darwin":
        return
    import mujoco.viewer

    if mujoco.viewer._MJPYTHON is not None:
        return
    executable = Path(sys.executable).parent / "mjpython"
    if not executable.is_file():
        found = shutil.which("mjpython")
        if not found:
            raise RuntimeError("Native macOS viewing needs mjpython; reinstall mujoco or use --headless")
        executable = Path(found)
    os.execv(str(executable), [str(executable), "-m", "cpg_console", *sys.argv[1:]])


def advance_simulation(runtime, controls, acceleration):
    if controls.reset_requested:
        runtime.reset()
        controls.reset_requested = False
    if controls.paused or runtime.failed:
        runtime.set_command(np.zeros(3))
        return False
    runtime.set_command(controls.advance(runtime.env.dt, acceleration))
    runtime.step()
    if runtime.failed:
        controls.stop()
    return True


def run(args):
    from legged_gym.utils.helpers import resolve_rl_device

    torch.set_num_threads(1)
    device = resolve_rl_device(args.device)
    checkpoint = args.checkpoint
    if checkpoint is None and not args.demo:
        checkpoint = find_checkpoint(args.log_root, args.task)
    with ExitStack() as stack:
        runtime = Runtime(checkpoint, demo=args.demo, task=args.task, device=device, seed=args.seed, gait=args.gait)
        stack.callback(runtime.close)
        controls = Controls(runtime.bounds)
        initial = np.asarray(args.velocity)
        if np.any((initial != 0) & ((initial < runtime.bounds[:, 0]) | (initial > runtime.bounds[:, 1]))):
            raise ValueError(f"--velocity exceeds checkpoint command ranges: {runtime.bounds.tolist()}")
        controls.target[:] = initial
        events = SimpleQueue()

        def on_key(code):
            if code == 256:
                events.put("\x1b")
            elif 0 <= code < 128:
                events.put(chr(code))

        joystick = None
        if args.joystick is not None:
            deadman = args.deadman_button
            if deadman is None and not args.hold_to_move:
                deadman = -1
            joystick = Joystick(
                args.joystick, args.axes, args.axis_signs, args.deadzone, deadman,
                reset_button=args.reset_button, stop_button=args.stop_button, pause_button=args.pause_button,
            )
            stack.callback(joystick.close)
        viewer = None
        if not args.headless:
            viewer = Viewer(runtime.env, on_key)
            stack.callback(viewer.close)
        recorder = None
        if args.csv:
            recorder = Recorder(args.csv)
            stack.callback(recorder.close)
        keyboard = stack.enter_context(Keyboard())
        title = f"CPG Console | {runtime.task} | {checkpoint or ('UNTRAINED ' + args.demo)}"
        print(title, flush=True)
        dashboard = stack.enter_context(Dashboard(title))
        steps = 0
        start = deadline = time.perf_counter()
        next_view = next_status = start
        previous_status = None
        while not controls.quit and (args.steps is None or steps < args.steps):
            if viewer and not viewer.handle.is_running():
                break
            keys = keyboard.poll()
            while True:
                try:
                    keys.append(events.get_nowait())
                except Empty:
                    break
            if joystick:
                velocity, enabled, buttons = joystick.poll(runtime.bounds)
                keys.extend(buttons)
                controls.joystick(np.zeros(3) if args.demo else velocity, enabled)
            for key in keys:
                controls.key(key, joystick=joystick is not None or args.demo is not None)
            if controls.quit:
                break
            stepped = advance_simulation(runtime, controls, args.acceleration)
            if stepped:
                steps += 1
            now = time.perf_counter()
            row = sample(runtime, controls, now - start, steps)
            requested = joystick.velocity if joystick else controls.target
            row.update(dict(zip(("input_vx", "input_vy", "input_wz"), requested)))
            source = f"{joystick.name}: {joystick.status(controls)}" if joystick else "keyboard"
            status = (row["state"], source, controls.last_action)
            if recorder and stepped:
                recorder.write(row)
            if now >= next_view and viewer:
                viewer.sync(row, source, runtime.reason)
                next_view = now + 1 / 50
            if now >= next_status or status != previous_status:
                dashboard.draw(row, source, runtime.reason)
                next_status = now + 0.5
                previous_status = status
            if runtime.failed and args.steps is not None:
                break
            if not args.fast or controls.paused or runtime.failed:
                deadline += runtime.env.dt
                time.sleep(max(0, deadline - time.perf_counter()))
                if time.perf_counter() - deadline > runtime.env.dt:
                    deadline = time.perf_counter()
        final = sample(runtime, controls, time.perf_counter() - start, steps)
    print(f"Finished: {steps} steps, {final['time_s']:.2f}s, height={final['height_m']:.3f}m, "
          f"state={final['state']}, RTF={final['rtf']:.2f}")
    return 2 if runtime.failed else 0


def main():
    args = parse_args()
    try:
        ensure_mjpython(args)
        code = run(args)
    except KeyboardInterrupt:
        code = 130
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"cpg-console: {error}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
