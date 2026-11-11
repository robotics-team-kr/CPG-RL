"""Strict frozen-base loading, configuration snapshots and reproducible run metadata."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.models import MLPModel

from legged_gym.envs.mini_cheetah.allgaits import MiniCheetahAllGaits
from legged_gym.envs.mini_cheetah.config import MiniCheetahAllGaitsCfg, MiniCheetahAllGaitsPPO
from legged_gym.utils.agent_factory import policy_contract, runner_config
from legged_gym.utils.helpers import update_class_from_dict


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_hash():
    """Identify the actual Python source, including uncommitted implementation files."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_base(path, num_envs=1, device="cpu"):
    """Validate the original environment contract before adding any research settings."""
    path = Path(path).resolve()
    config_path = path.parent / "run_config.json"
    snapshot = json.loads(config_path.read_text())
    if snapshot.get("task") != "mini_cheetah_allgaits":
        raise ValueError(
            "A mini_cheetah_allgaits checkpoint and its run_config.json are required; CPG 84D is incompatible"
        )
    cfg, train_cfg = MiniCheetahAllGaitsCfg(), MiniCheetahAllGaitsPPO()
    update_class_from_dict(cfg, snapshot["env"])
    update_class_from_dict(train_cfg, snapshot["train"])
    cfg.env.num_envs = num_envs
    data = torch.load(path, map_location="cpu", weights_only=False)
    env = MiniCheetahAllGaits(deepcopy(cfg))
    try:
        contract = policy_contract(env)
        stored = (data.get("infos") or {}).get("legged_gym", {}).get("contract")
        if stored != contract or snapshot.get("contract") != contract:
            raise ValueError("Base checkpoint/config/model contract mismatch")
        if (env.num_obs, env.num_actions) != (68, 8):
            raise ValueError("Expected full AllGaits observation/action dimensions 68/8")
        actor_cfg = runner_config(train_cfg)["actor"].copy()
        actor_cfg.pop("class_name")
        actor = MLPModel(env.get_observations(), {"actor": ["policy"]}, "actor", 8, **actor_cfg)
        actor.load_state_dict(data["actor_state_dict"], strict=True)
        # Tensor-only exported module includes frozen normalizer buffers.
        frozen = actor.as_jit().to(device).eval()
        frozen.requires_grad_(False)
        audit = dict(
            checkpoint=str(path),
            checkpoint_sha256=file_hash(path),
            config_sha256=file_hash(config_path),
            contract=contract,
            frozen_state_sha256=module_hash(frozen),
            code_commit=code_commit(),
            source_sha256=source_hash(),
            normalization={k: list(v.shape) for k, v in actor.state_dict().items() if "normalizer" in k},
            iteration=data.get("iter"),
            versions=snapshot.get("versions", {}),
        )
    finally:
        env.close()
    return frozen, cfg, audit


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
