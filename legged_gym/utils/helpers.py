"""Configuration, CLI and device helpers for the MuJoCo training path."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        key: class_to_dict(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def update_class_from_dict(obj, values, *, strict=True):
    for key, value in values.items():
        if not hasattr(obj, key):
            if not strict:
                continue
            raise ValueError(f"Unknown configuration field: {key}")
        target = getattr(obj, key)
        if isinstance(value, dict) and hasattr(target, "__dict__"):
            update_class_from_dict(target, value, strict=strict)
        else:
            setattr(obj, key, value)


def set_seed(seed):
    if seed < 0:
        seed = random.SystemRandom().randrange(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def resolve_rl_device(device="auto"):
    """Validate the learner device and canonicalize CUDA to an explicit ordinal.

    An explicit unavailable device raises instead of silently moving training.
    Automatic selection is the default and prefers CUDA, then MPS, then CPU.
    """
    if str(device) == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("rl_device must be mps, cpu, cuda[:N] or auto") from error
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build or use --rl_device cpu/auto")
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(f"CUDA index {index} is out of range for {count} visible device(s)")
        return f"cuda:{index}"
    if parsed.type not in ("mps", "cpu") or parsed.index not in (None, 0):
        raise ValueError("rl_device must be mps, cpu, cuda[:N] or auto")
    if parsed.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select --rl_device cuda, cpu or auto")
    return parsed.type


def get_load_path(root, load_run=-1, checkpoint=-1):
    root = Path(root)
    if str(load_run) == "-1":
        runs = sorted((p for p in root.glob("*") if p.is_dir() and any(p.glob("model_*.pt"))), key=lambda p: p.name)
        if not runs:
            raise FileNotFoundError(f"No checkpoints under {root}")
        run = runs[-1]
    else:
        run = root / str(load_run)
    if checkpoint == -1:
        models = [p for p in run.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if not models:
            raise FileNotFoundError(f"No checkpoints in {run}")
        return str(max(models, key=lambda p: int(p.stem.split("_")[-1])))
    path = run / f"model_{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def get_args(argv=None, *, playback=False):
    parser = argparse.ArgumentParser(description="Mini Cheetah: CPU MuJoCo physics + PPO")
    parser.add_argument(
        "--task",
        choices=["mini_cheetah_cpg", "mini_cheetah_joint", "mini_cheetah_allgaits"],
        default="mini_cheetah_cpg",
    )
    parser.add_argument("--rl_device", "--rl-device", default="auto")
    parser.add_argument("--num_envs", "--num-envs", type=int)
    parser.add_argument("--num_threads", "--num-threads", type=int)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", "--max-iterations", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--observation", choices=["full", "medium"])
    parser.add_argument("--coupling", type=float, help="Coupling weight: CPG default 0, AllGaits default 10")
    parser.add_argument(
        "--gait",
        help="AllGaits only: random (resample every 3s), walk, amble, trot, pace, "
        "bound, pronk, canter, transverse_gallop, rotary_gallop",
    )
    parser.add_argument("--randomize_geometry", action="store_true", default=None)
    parser.add_argument("--experiment_name")
    parser.add_argument("--run_name")
    parser.add_argument("--log_root", help="Training log directory (default: ./logs/<task>)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--checkpoint_path", "--checkpoint-path")
    parser.add_argument("--headless", action="store_true", help="Disable the training web monitor or playback viewer")
    if not playback:
        parser.add_argument("--viewer_host", help="Viser bind address (default: 127.0.0.1)")
        parser.add_argument("--viewer_port", type=int, help="Viser port (default: 8080)")
        parser.add_argument("--viewer_fps", type=float, help="Maximum display rate (default: 15)")
    if playback:
        parser.add_argument("--steps", type=int)
        parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
        parser.add_argument("--demo", choices=["stand", "trot"], help="Untrained controller; no checkpoint required")
        parser.add_argument("--evaluate", action="store_true")
        parser.add_argument("--eval_seeds", nargs="+", type=int)
        parser.add_argument("--eval_output")
    args = parser.parse_args(argv)
    if args.gait is not None and args.task != "mini_cheetah_allgaits":
        parser.error("--gait requires --task mini_cheetah_allgaits")
    if not playback:
        if args.viewer_port is not None and not 1 <= args.viewer_port <= 65535:
            parser.error("viewer_port must be between 1 and 65535")
        if args.viewer_fps is not None and (not math.isfinite(args.viewer_fps) or args.viewer_fps <= 0):
            parser.error("viewer_fps must be finite and positive")
    for name in ("num_envs", "max_iterations", "torch_threads", "steps"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if args.num_threads is not None and args.num_threads < 0:
        parser.error("num_threads must be nonnegative")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.coupling is not None and (not math.isfinite(args.coupling) or args.coupling < 0):
        parser.error("coupling must be finite and nonnegative")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning_rate must be finite and positive")
    if playback:
        if args.demo and (args.evaluate or args.checkpoint_path or args.resume):
            parser.error("demo cannot be combined with a checkpoint, resume or evaluation")
        if args.evaluate and args.velocity is not None:
            parser.error("evaluate uses the configured command suite")
        if args.velocity is not None and not all(math.isfinite(v) for v in args.velocity):
            parser.error("velocity must be finite")
        if not args.evaluate and (args.eval_seeds is not None or args.eval_output is not None):
            parser.error("eval_seeds and eval_output require evaluate")
        if args.evaluate and args.seed is not None and args.eval_seeds is not None:
            parser.error("choose seed or eval_seeds")
    try:
        args.rl_device = resolve_rl_device(args.rl_device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.torch_threads)
    return args


def update_cfg_from_args(env_cfg, train_cfg, args):
    for cfg in (env_cfg, train_cfg):
        if cfg is not None and args.seed is not None:
            cfg.seed = args.seed
    if env_cfg is not None:
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.num_threads is not None:
            env_cfg.sim.nthread = args.num_threads
        if args.observation is not None:
            env_cfg.env.observation = args.observation
        if args.coupling is not None:
            env_cfg.cpg.coupling_strength = args.coupling
        if args.randomize_geometry is not None:
            env_cfg.cpg.randomize_geometry = args.randomize_geometry
        if getattr(args, "gait", None) is not None:
            if args.gait != "random" and args.gait not in env_cfg.cpg.gait_contact_times:
                raise ValueError(
                    f"Unknown gait {args.gait!r}; choose random or {', '.join(env_cfg.cpg.gait_contact_times)}"
                )
            env_cfg.gait.name = args.gait
    if train_cfg is not None:
        # Explicit invocation controls visibility, including resumes of old headless runs.
        train_cfg.viewer.enabled = not args.headless
        for arg, field in (("viewer_host", "host"), ("viewer_port", "port"), ("viewer_fps", "max_fps")):
            value = getattr(args, arg, None)
            if value is not None:
                setattr(train_cfg.viewer, field, value)
        if args.learning_rate is not None:
            train_cfg.algorithm.learning_rate = args.learning_rate
        for key in ("max_iterations", "experiment_name", "run_name"):
            value = getattr(args, key)
            if value is not None:
                setattr(train_cfg.runner, key, value)
    return env_cfg, train_cfg
