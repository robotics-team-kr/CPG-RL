# SPDX-License-Identifier: BSD-3-Clause
"""PPO construction and strict checkpoint contracts, adapted from legged-gym-mujoco."""

from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from .helpers import class_to_dict


def policy_contract(env):
    """Fields that must match when a saved policy is applied to this task."""
    contract = {
        "version": 3,
        "robot": env.cfg.asset.file,
        "asset_sha256": env.sim.asset_digest,
        "joint_names": env.dof_names,
        "policy_obs": env.num_obs,
        "critic_obs": env.num_privileged_obs,
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.control.decimation,
        "action_scale": env.cfg.control.action_scale,
        "clip_actions": env.cfg.control.clip_actions,
        "default_joint_pos": env.default_dof_pos.tolist(),
        "kp": env.p_gains.tolist(),
        "kd": env.d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "armature": env.cfg.control.armature,
        "frictionloss": env.cfg.control.frictionloss,
        "foot_sites": env.cfg.asset.foot_sites,
        "solver_settings": {
            key: getattr(env.cfg.sim, key)
            for key in ("iterations", "cone", "ls_iterations", "tolerance", "ls_tolerance", "ccd_iterations")
        },
    }
    contract["control_mode"] = env.cfg.control.mode
    contract["observation_mode"] = env.cfg.env.observation
    contract["cpg"] = class_to_dict(env.cfg.cpg)
    contract["contact_settings"] = {
        key: getattr(env.cfg.asset, key)
        for key in ("self_collisions", "foot_contact_bodies", "foot_geoms", "foot_solimp", "reward_body")
        if hasattr(env.cfg.asset, key)
    }
    return contract


class TerminalPPO(PPO):
    """Bootstrap truncations from the actual terminal state, before automatic reset."""

    def process_env_step(self, obs, rewards, dones, extras):
        timeouts = extras.get("time_outs")
        if timeouts is not None and timeouts.any():
            terminal = extras["time_outs_obs"].to(self.device)
            with torch.no_grad():
                value = self.critic(terminal).squeeze(-1)
            rewards = rewards + self.gamma * value * timeouts.to(self.device)
        super().process_env_step(obs, rewards, dones, {k: v for k, v in extras.items() if k != "time_outs"})


def runner_config(cfg):
    distribution = getattr(cfg.policy, "action_distribution", "gaussian")
    if distribution == "gaussian":
        distribution_cfg = {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "std_type": "scalar",
            "init_std": cfg.policy.init_noise_std,
        }
    elif distribution == "beta":
        distribution_cfg = {"class_name": "rsl_rl.modules:BetaDistribution", "action_range": [-1.0, 1.0]}
    else:
        raise ValueError(f"Unknown action distribution: {distribution}")
    actor = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": cfg.policy.actor_hidden_dims,
        "activation": cfg.policy.activation,
        "obs_normalization": cfg.policy.obs_normalization,
        "distribution_cfg": distribution_cfg,
    }
    critic = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": cfg.policy.critic_hidden_dims,
        "activation": cfg.policy.activation,
        "obs_normalization": cfg.policy.obs_normalization,
    }
    return {
        "seed": cfg.seed,
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": actor,
        "critic": critic,
        "algorithm": {
            **class_to_dict(cfg.algorithm),
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "class_name": "legged_gym.utils.agent_factory:TerminalPPO",
        },
        "logger": "tensorboard",
        "check_for_nan": True,
    }


def override_learning_rate(algorithm, value):
    algorithm.learning_rate = value
    for group in algorithm.optimizer.param_groups:
        group["lr"] = value


class PPORunner(OnPolicyRunner):
    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["legged_gym"] = {
            "contract": policy_contract(self.env),
            "training_state": self.env.training_state_dict(),
            "next_iteration": self.current_learning_iteration + 1,
        }
        super().save(path, metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, restore_env=True):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state = (checkpoint.get("infos") or {}).get("legged_gym")
        if state is None or state.get("contract") != policy_contract(self.env):
            raise ValueError("Checkpoint robot, CPG, observation or control contract does not match")
        del checkpoint
        infos = super().load(path, load_cfg, strict, map_location or "cpu")
        if restore_env:
            self.current_learning_iteration = state["next_iteration"]
            self.env.load_training_state_dict(state["training_state"])
        return infos


def create_agent(env, cfg, log_dir=None, device="cpu"):
    config = runner_config(deepcopy(cfg))
    env.capture_terminal_observations = True
    return PPORunner(env, config, log_dir, device=device)
