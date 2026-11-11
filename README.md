# CPG Console

Native MuJoCo viewer and terminal controls for Mini Cheetah CPG-RL checkpoints: policy → CPG → IK → PD. Python 3.10–3.13; Linux, macOS and Windows. The default floor uses a checkerboard tile grid.

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cpg-console                       # Latest checkpoint under ./logs
cpg-console --checkpoint /path/to/model_1100.pt
cpg-console --joystick            # Device 0
cpg-console --headless --steps 1000 --velocity 0.5 0 0 --csv tracking.csv
```

Keep `run_config.json` beside each checkpoint. Only load trusted PyTorch checkpoints. Robot assets, observations, CPG and control settings are checked before inference. macOS automatically launches the native viewer through `mjpython`.

W/S adjusts forward speed, A/D lateral speed, Q/E yaw rate (0.1 per press). Space sets zero velocity; P pauses physics; R resets; Esc quits. Keys work in the terminal or viewer. The console shows commanded/measured velocity, foot contacts, CPG frequency/amplitude and simulation speed. A fall freezes playback until reset; bounded runs exit with code 2 on a fall. Zero velocity is a policy command, not a guaranteed standing pose.

Gamepads use SDL's device mapping: left stick for translation, right stick for yaw; no enable button is required by default. A resets, B sets zero command, Start toggles physics pause/resume once per press. A reset and Start resume accept the current stick input immediately, with acceleration limiting. After B stops motion, center the sticks before moving again. Disconnecting sets zero command without pausing physics. The console and viewer show input availability, stick target, applied command and the last action.

`--joystick 1` selects device 1; `--deadzone 0.12` adjusts stick deadzone. Optional `--hold-to-move` requires holding LB; releasing it sets zero command. Raw overrides: `--axes`, `--axis-signs`, `--deadman-button`, `--reset-button`, `--stop-button`, `--pause-button` (button `-1` disables that binding). Unrecognized joysticks fall back to axes `1 0 2` and reset/stop/pause buttons `0/1/7` (enable button `4` with `--hold-to-move`); override these for your device. Keyboard velocity keys are disabled with a gamepad. Demos ignore velocity input.

```bash
cpg-console --demo stand             # Untrained posture
cpg-console --demo trot              # Untrained open-loop CPG
cpg-train --num_envs 64 --headless
```

Training runs and checkpoints remain in `logs/`. Training supports `mini_cheetah_cpg` and `mini_cheetah_allgaits`; install `pip install -e '.[train]'` for the optional web training monitor. Use `--gait trot` with an AllGaits checkpoint. `cpg-play --evaluate --checkpoint_path /path/to/model.pt --headless` runs the existing evaluation suite (set `--task` to match).

`cpg-console --help` lists options. `--fast --headless --steps 1000` runs without real-time pacing. CSV files are created without overwriting existing files.

Licenses: [LICENSE](LICENSE), [licenses/](licenses/). Asset sources and redistribution status: [provenance.json](resources/robots/provenance.json).
