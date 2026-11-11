"""Dual-rate PPO with per-environment macro draining and exact boundary resumes."""

import random
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch

from legged_gym.utils.coop_io import atomic_save, module_hash
from legged_gym.utils.helpers import class_to_dict

from .cooperative_actor_critic import Agent, SharedActor
from .multi_rate_rollout import DecisionBuffer


@dataclass
class PPOSettings:
    rollout_steps: int = 200
    hidden: tuple = (256, 256)
    gamma: float = float(np.exp(-0.01 / 2))
    lam_slow: float = 0.95
    lam_reflex: float = 0.95
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    lr_slow: float = 0.00015
    lr_reflex: float = 0.0003
    entropy: float = 0.001
    target_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_normalization: bool = True


class CooperativePPO:
    def __init__(self, env, base, audit, params=None, device="cpu", stage="cooperative_spatial"):
        self.env, self.base, self.audit = env, base, audit
        if not env.settings.enabled:
            raise ValueError("enabled=False is a legacy identity test mode; use stage=base for evaluation")
        self.params, self.device, self.stage = params or PPOSettings(), torch.device(device), stage
        p = self.params
        if min(p.rollout_steps, p.epochs, p.minibatches) < 1 or not 0 < p.gamma <= 1:
            raise ValueError("Invalid rollout/PPO settings")
        obs = env.observations_coop()
        critic_dim = obs["critic"].shape[1]
        self.shared = SharedActor().to(self.device) if stage == "monolithic" else None
        self.agents = {
            role: Agent(223 if self.shared is not None else dim, critic_dim, actions, p.hidden, role == "reflex").to(
                self.device
            )
            for role, dim, actions in (("slow", 102, 8), ("reflex", 121, 20))
        }
        self.train_roles = {"slow", "reflex"}
        if stage == "reflex_only":
            self.train_roles = {"reflex"}
        elif stage in ("slow_only", "sequential"):
            self.train_roles = {"slow"}
        elif stage in ("base", "independent_merge"):
            self.train_roles = set()
        if self.shared is not None:
            # A single autoregressive joint PPO objective on the R timeline.
            # S log-probability appears only at actual S decision ticks.
            self.train_roles = {"reflex"}
            for agent in self.agents.values():
                agent.actor = torch.nn.Identity()
                agent.log_std.requires_grad_(False)
        self.optimizers = {
            role: torch.optim.Adam(agent.parameters(), lr=getattr(p, "lr_" + role))
            for role, agent in self.agents.items()
        }
        self.shared_optimizer = torch.optim.Adam(self.shared.parameters(), lr=p.lr_reflex) if self.shared else None
        self.iteration, self.total_steps = 0, 0
        self.base_hash = module_hash(base)
        self.last_durations = {}

    def actor_obs(self, obs, role):
        return torch.cat((obs["slow"], obs["reflex"]), dim=-1) if self.shared else obs[role]

    def distribution(self, obs, role):
        if self.shared:
            return self.shared.distribution(obs, role)
        return self.agents[role].distribution(obs)

    @torch.no_grad()
    def actions(self, obs, role, ids, deterministic=False, critic=None):
        x = self.actor_obs(obs, role)[ids].to(self.device)
        state = (obs["critic"] if critic is None else critic)[ids].to(self.device)
        dist = self.distribution(x, role)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.agents[role].value(state)
        return x.cpu().numpy(), state.cpu().numpy(), action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def collect(self):
        p, n = self.params, self.env.num_envs
        buffers = {role: DecisionBuffer(n, p.gamma, getattr(p, "lam_" + role)) for role in self.agents}
        pending = {role: [None] * n for role in self.agents}
        obs = self.env.observations_coop()
        # Update only before collection; no statistic changes between sampling and PPO ratio evaluation.
        for role in self.train_roles:
            agent = self.agents[role]
            agent.obs_norm.update(self.actor_obs(obs, role).to(self.device))
            agent.critic_norm.update(obs["critic"].to(self.device))
        if self.shared:
            self.shared.norm.update(self.actor_obs(obs, "slow").to(self.device))
        episodes, ticks, actual_steps = [], 0, 0
        while True:
            draining = ticks >= p.rollout_steps
            active = (
                np.ones(n, dtype=bool)
                if not draining
                else np.array([any(pending[role][i] is not None for role in pending) for i in range(n)])
            )
            if not active.any():
                break
            pre_decision_critic = obs["critic"].clone()
            sampled_slow = {}
            for role in ("slow", "reflex"):
                countdown = getattr(self.env, role + "_countdown")
                ids = np.flatnonzero(active & (countdown == 0))
                if not len(ids):
                    continue
                # During drain, R may make decisions while a held S action finishes;
                # S never starts a new macro once its existing macro has finished.
                if draining and role == "slow":
                    continue
                x, state, action, logp, value = self.actions(obs, role, ids, critic=pre_decision_critic)
                for j, i in enumerate(ids):
                    if pending[role][i] is not None:
                        raise RuntimeError("Attempted to replace a pending decision")
                    pending[role][i] = dict(
                        obs=x[j],
                        critic=state[j],
                        action=action[j],
                        logp=logp[j],
                        value=value[j],
                        reward=0.0,
                        duration=0,
                    )
                    if self.shared and role == "slow":
                        sampled_slow[i] = (x[j], action[j], logp[j])
                    elif self.shared:
                        sx, sa, sl = sampled_slow.get(i, (np.zeros(223), np.zeros(8), 0.0))
                        pending[role][i].update(slow_obs=sx, slow_action=sa, slow_due=float(i in sampled_slow))
                        pending[role][i]["logp"] += sl
                getattr(self.env, "set_" + role)(action, ids)
                # Crucial: R sees the current S action, with no future physics observations.
                obs = self.env.observations_coop()
            base_actions = self.base(obs["base"].to(self.device)).cpu()
            next_obs, reward, done, info = self.env.step_coop(base_actions, active)
            reward, done, timeouts = reward.numpy(), done.numpy(), info["time_outs"].numpy()
            for role in pending:
                end = active & ((getattr(self.env, role + "_countdown") == 0) | done)
                # Value inputs are pre-next-S-decision states, including countdown and held past S.
                state = next_obs["critic"].clone()
                state[done] = info["terminal"]["critic"][done]
                next_values = self.agents[role].value(state.to(self.device)).cpu().numpy()
                for i in np.flatnonzero(active):
                    row = pending[role][i]
                    if row is None:
                        continue
                    row["reward"] += p.gamma ** row["duration"] * float(reward[i])
                    row["duration"] += 1
                    if end[i]:
                        row.update(
                            next_value=float(next_values[i]),
                            terminal=bool(done[i] and not timeouts[i]),
                            boundary=bool(done[i]),
                        )
                        buffers[role].add(i, row)
                        pending[role][i] = None
            episodes.extend(info["episodes"])
            actual_steps += int(active.sum())
            ticks += 1
            obs = next_obs
            if ticks > p.rollout_steps + self.env.settings.slow_interval + self.env.settings.reflex_interval:
                raise RuntimeError("Macro drain exceeded its bounded duration")
        self.total_steps += actual_steps
        self.last_durations = {
            role: [row["duration"] for trajectory in buf.trajectories for row in trajectory]
            for role, buf in buffers.items()
        }
        return buffers, episodes, actual_steps

    def update(self, buffers):
        p, stats = self.params, {}
        order = ("reflex", "slow") if self.iteration % 2 == 0 else ("slow", "reflex")
        for role in order:
            if role not in self.train_roles:
                continue
            data = buffers[role].tensors(self.device)
            agent, optimizer = self.agents[role], self.optimizers[role]
            if p.value_normalization:
                agent.normalize_returns(data["return"], optimizer)
            losses, kls, gradients = [], [], []
            stop = False
            for _ in range(p.epochs):
                indices = torch.randperm(len(data["obs"]), device=self.device)
                for batch in torch.tensor_split(indices, min(p.minibatches, len(indices))):
                    dist = self.distribution(data["obs"][batch], role)
                    logp = dist.log_prob(data["action"][batch]).sum(-1)
                    entropy = dist.entropy().sum(-1)
                    if self.shared:
                        slow_dist = self.shared.distribution(data["slow_obs"][batch], "slow")
                        logp = logp + data["slow_due"][batch] * slow_dist.log_prob(data["slow_action"][batch]).sum(-1)
                        entropy = entropy + data["slow_due"][batch] * slow_dist.entropy().sum(-1)
                    ratio = (logp - data["logp"][batch]).exp()
                    kl = ((ratio - 1) - (logp - data["logp"][batch])).mean()
                    if not torch.isfinite(kl) or not torch.isfinite(ratio).all():
                        raise FloatingPointError("Nonfinite PPO ratio")
                    kls.append(float(kl.detach()))
                    if float(kl.detach()) > p.target_kl * 1.5:
                        stop = True
                        break
                    advantage = data["advantage"][batch]
                    surrogate = -torch.minimum(
                        ratio * advantage, ratio.clamp(1 - p.clip, 1 + p.clip) * advantage
                    ).mean()
                    value_error = agent.value(data["critic"][batch]) - data["return"][batch]
                    value_loss = (value_error / agent.return_norm.var[0].sqrt()).square().mean()
                    loss = surrogate + 0.5 * value_loss - p.entropy * entropy.mean()
                    optimizer.zero_grad()
                    if self.shared_optimizer:
                        self.shared_optimizer.zero_grad()
                    loss.backward()
                    parameters = list(agent.parameters()) + (list(self.shared.parameters()) if self.shared else [])
                    norm = torch.nn.utils.clip_grad_norm_(parameters, p.max_grad_norm)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite PPO gradient")
                    optimizer.step()
                    if self.shared_optimizer:
                        self.shared_optimizer.step()
                    losses.append(float(loss.detach()))
                    gradients.append(float(norm))
                if stop:
                    break
            stats[role + "_loss"] = float(np.mean(losses)) if losses else 0.0
            stats[role + "_kl"] = float(np.mean(kls))
            stats[role + "_grad_norm"] = float(np.mean(gradients)) if gradients else 0.0
            stats[role + "_decisions"] = len(data["obs"])
        if module_hash(self.base) != self.base_hash:
            raise RuntimeError("Frozen base parameters/normalizer changed")
        self.iteration += 1
        return stats

    @torch.no_grad()
    def act_step(self, deterministic=True, active=None):
        obs = self.env.observations_coop()
        active = np.ones(self.env.num_envs, dtype=bool) if active is None else active
        for role in ("slow", "reflex"):
            ids = np.flatnonzero(active & (getattr(self.env, role + "_countdown") == 0))
            if len(ids):
                _, _, action, _, _ = self.actions(obs, role, ids, deterministic)
                getattr(self.env, "set_" + role)(action, ids)
                obs = self.env.observations_coop()
        return self.env.step_coop(self.base(obs["base"].to(self.device)).cpu(), active)

    def save(self, path):
        if self.env.slow_countdown.any() or self.env.reflex_countdown.any():
            raise RuntimeError("Save only at drained decision boundaries")
        atomic_save(
            dict(
                version=1,
                learner_device=str(self.device),
                runtime_versions={
                    "torch": str(torch.__version__),
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                },
                stage=self.stage,
                settings=asdict(self.env.settings),
                params=asdict(self.params),
                audit=self.audit,
                environment_config=class_to_dict(self.env.cfg),
                iteration=self.iteration,
                total_steps=self.total_steps,
                agents={k: v.state_dict() for k, v in self.agents.items()},
                optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
                shared=self.shared.state_dict() if self.shared else None,
                shared_optimizer=self.shared_optimizer.state_dict() if self.shared_optimizer else None,
                environment=self.env.state_dict(),
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                mps_rng=torch.mps.get_rng_state() if self.device.type == "mps" else None,
                cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            ),
            path,
        )

    def load(self, path, resume=True, roles=None):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["audit"]["checkpoint_sha256"] != self.audit["checkpoint_sha256"]:
            raise ValueError("Residual checkpoint uses a different frozen base")
        if resume and (
            state.get("learner_device") != str(self.device)
            or state["audit"].get("source_sha256") != self.audit.get("source_sha256")
            or state.get("runtime_versions")
            != {"torch": str(torch.__version__), "mujoco": mujoco.__version__, "numpy": np.__version__}
        ):
            raise ValueError(
                "Exact resume requires the same source, runtime versions and learner device; use warm-start otherwise"
            )
        if resume and (
            state["settings"] != asdict(self.env.settings)
            or state["params"] != asdict(self.params)
            or state["stage"] != self.stage
            or state["environment_config"] != class_to_dict(self.env.cfg)
        ):
            raise ValueError("Exact resume requires matching settings, stage and PPO configuration")
        for role in roles or self.agents:
            self.agents[role].load_state_dict(state["agents"][role], strict=True)
            if resume:
                self.optimizers[role].load_state_dict(state["optimizers"][role])
        if self.shared is not None:
            if state["shared"] is None:
                raise ValueError("Monolithic checkpoint required")
            self.shared.load_state_dict(state["shared"])
            if resume:
                self.shared_optimizer.load_state_dict(state["shared_optimizer"])
        if resume:
            self.env.load_state_dict(state["environment"])
            self.iteration, self.total_steps = state["iteration"], state["total_steps"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if self.device.type == "mps" and state.get("mps_rng") is not None:
                torch.mps.set_rng_state(state["mps_rng"])
            if self.device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
