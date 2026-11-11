import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel

from legged_gym.envs import task_registry
from legged_gym.utils.agent_factory import policy_contract, runner_config
from legged_gym.utils.helpers import set_seed, update_class_from_dict


def find_checkpoint(root, task=None):
    candidates = [
        path for path in Path(root).rglob("model_*.pt")
        if path.stem.removeprefix("model_").isdigit() and (path.parent / "run_config.json").is_file()
    ]
    if task is not None:
        candidates = [
            path for path in candidates
            if json.loads((path.parent / "run_config.json").read_text()).get("task") == task
        ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint with run_config.json under {root}; use --checkpoint or --demo stand")
    return max(candidates, key=lambda path: (path.parent.name, int(path.stem.split("_")[-1])))


class ManualCommands:
    def _resample_commands(self, ids):
        self._next_command[ids] = np.inf
        self.commands[ids] = 0
        self.heading_envs[ids] = False
        self.standing_envs[ids] = False

    def _update_commands(self):
        pass


class Runtime:
    def __init__(self, checkpoint=None, *, demo=None, task=None, device="cpu", seed=42, gait=None):
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.demo, self.device = demo, device
        if checkpoint:
            snapshot = json.loads((self.checkpoint.parent / "run_config.json").read_text())
            if task is not None and task != snapshot["task"]:
                raise ValueError("Checkpoint task does not match --task")
            task = snapshot["task"]
        elif not demo:
            raise ValueError("A checkpoint or demo is required")
        self.task = task or "mini_cheetah_cpg"
        cfg, train_cfg = task_registry.get_cfgs(self.task)
        if checkpoint:
            update_class_from_dict(cfg, snapshot["env"])
            update_class_from_dict(train_cfg, snapshot["train"])
        if demo and self.task != "mini_cheetah_cpg":
            raise ValueError("Demos require --task mini_cheetah_cpg")
        if gait:
            if not hasattr(cfg, "gait") or gait not in (*cfg.cpg.gait_contact_times, "random"):
                raise ValueError("--gait requires an AllGaits task and a supported gait name")
            cfg.gait.name = gait
        cfg.seed = seed
        cfg.env.num_envs = 1
        cfg.sim.nthread = 1
        cfg.noise.add_noise = False
        for field in ("randomize_friction", "randomize_com", "randomize_encoder_bias", "push_robots"):
            setattr(cfg.domain_rand, field, False)
        cfg.commands.curriculum = cfg.commands.heading_command = False
        cfg.commands.standing_fraction = 0.0
        if demo == "stand":
            cfg.control.mode = "joint"
        set_seed(seed)
        env_type = type("ConsoleEnvironment", (ManualCommands, task_registry.tasks[self.task][0]), {})
        self.env = env_type(cfg=cfg, headless=True)
        try:
            self.env.auto_reset = False
            self.env.max_episode_length = np.iinfo(np.int64).max
            self.bounds = np.asarray(cfg.commands.limit_ranges, dtype=float)
            self.policy = None
            if checkpoint:
                saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
                contract = (saved.get("infos") or {}).get("legged_gym", {}).get("contract")
                if contract != policy_contract(self.env) or snapshot.get("contract") != contract:
                    raise ValueError("Checkpoint robot, asset, observation, CPG or control contract does not match")
                actor_cfg = runner_config(train_cfg)["actor"]
                actor_cfg.pop("class_name")
                self.policy = MLPModel(
                    self.env.get_observations(), {"actor": ["policy"]}, "actor", self.env.num_actions, **actor_cfg
                ).to(device)
                self.policy.load_state_dict(saved["actor_state_dict"], strict=True)
                self.policy.eval()
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self):
        env = self.env
        env.reset()
        env.sim.qpos[0, :2] = 0
        env.sim.qpos[0, 3:7] = [1, 0, 0, 0]
        env.sim.forward()
        env._refresh_state()
        self.command = np.zeros(3)
        self.failed = False
        self.reason = ""
        self.set_command(self.command)

    def set_command(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command must contain finite VX, VY and WZ")
        self.command = np.where(command == 0, 0, np.clip(command, self.bounds[:, 0], self.bounds[:, 1]))
        self.env.commands[:] = self.command
        self.env.compute_observations()

    @torch.inference_mode()
    def step(self):
        if self.failed:
            return
        if self.policy is not None:
            actions = self.policy(self.env.get_observations().to(self.device))
        else:
            actions = np.zeros((1, self.env.num_actions))
            if self.demo == "trot":
                actions[:, 4:8] = 2 * 1.5 / self.env.cfg.cpg.frequency_range[1] - 1
        self.env.step(actions)
        if self.env.terminated_buf[0]:
            self.failed = True
            self.reason = "tilt" if self.env._tilt_failure[0] else "body/thigh contact"

    def close(self):
        self.env.close()
