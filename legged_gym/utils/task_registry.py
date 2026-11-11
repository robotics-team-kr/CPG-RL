# SPDX-License-Identifier: BSD-3-Clause
# Adapted from legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction, resolved run configurations, and PPO checkpoint loading."""

import json
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from .agent_factory import create_agent, override_learning_rate, policy_contract
from .helpers import class_to_dict, get_args, get_load_path, set_seed, update_cfg_from_args, update_class_from_dict


class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.tasks[name] = (task_class, env_cfg, train_cfg)

    def get_cfgs(self, name, args=None):
        if name not in self.tasks:
            raise ValueError(f"Unknown task {name!r}; available: {', '.join(self.tasks)}")
        _, env_cfg, train_cfg = deepcopy(self.tasks[name])
        if args is not None and (args.resume or args.checkpoint_path):
            if not args.checkpoint_path:
                experiment = args.experiment_name or train_cfg.runner.experiment_name
                root = Path(args.log_root) if args.log_root else Path.cwd() / "logs" / experiment
                args.checkpoint_path = get_load_path(root, args.load_run, args.checkpoint)
            path = Path(args.checkpoint_path).resolve().parent / "run_config.json"
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint requires its resolved configuration: {path}")
            snapshot = json.loads(path.read_text())
            if snapshot["task"] != name:
                raise ValueError("Checkpoint task does not match --task")
            update_class_from_dict(env_cfg, snapshot["env"])
            update_class_from_dict(train_cfg, snapshot["train"])
        if args is not None:
            update_cfg_from_args(env_cfg, train_cfg, args)
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None, train_cfg=None):
        args = get_args() if args is None else args
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name, args)
        else:
            env_cfg = deepcopy(env_cfg)
        set_seed(env_cfg.seed)
        env = self.tasks[name][0](cfg=env_cfg, headless=True)
        return env, env_cfg

    def make_alg_runner(self, env, name, args=None, train_cfg=None, log_root="default"):
        if name == "mini_cheetah_coop_allgaits":
            raise ValueError("Use python -m legged_gym.scripts.train_coop for the dual-rate task")
        args = get_args() if args is None else args
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name, args)
        train_cfg = deepcopy(train_cfg)
        train_cfg.seed = env.cfg.seed
        if log_root == "default":
            log_root = Path(args.log_root) if args.log_root else Path.cwd() / "logs" / train_cfg.runner.experiment_name
        log_dir = None
        if log_root is not None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            if train_cfg.runner.run_name:
                # Run names are labels, not paths outside the log directory.
                if Path(train_cfg.runner.run_name).name != train_cfg.runner.run_name:
                    raise ValueError("run_name must be a single path component")
                run_name += "_" + train_cfg.runner.run_name
            directory = Path(log_root) / run_name
            directory.mkdir(parents=True, exist_ok=False)
            log_dir = str(directory)
        runner = create_agent(env, train_cfg, log_dir, args.rl_device)
        if args.resume or args.checkpoint_path:
            if not args.checkpoint_path:
                raise ValueError("Resolve checkpoint through get_cfgs before creating the runner")
            runner.load(args.checkpoint_path, restore_env=not getattr(args, "play", False))
            if args.learning_rate is not None:
                override_learning_rate(runner.alg, args.learning_rate)
        if log_dir is not None:
            (directory / "run_config.json").write_text(
                json.dumps(
                    {
                        "task": name,
                        "env": class_to_dict(env.cfg),
                        "train": class_to_dict(train_cfg),
                        "rl_device": args.rl_device,
                        "versions": {key: version(key) for key in ("mujoco", "rsl-rl-lib", "torch")},
                        "contract": policy_contract(env),
                    },
                    indent=2,
                )
                + "\n"
            )
        return runner, train_cfg


task_registry = TaskRegistry()
