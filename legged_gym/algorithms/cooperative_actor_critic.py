"""Heterogeneous Gaussian actors with frozen-per-rollout observation statistics."""

import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, outputs, hidden):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        if not len(x):
            return
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + len(x)
        delta = mean - self.mean
        new_var = (self.var * self.count + var * len(x) + delta.square() * self.count * len(x) / total) / total
        self.mean.add_(delta * len(x) / total)
        self.var.copy_(new_var.clamp_min(1e-6))
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / (self.var + 1e-6).sqrt()).clamp(-10, 10)


class Agent(nn.Module):
    def __init__(self, obs_dim, critic_dim, actions, hidden=(256, 256), reflex=False):
        super().__init__()
        self.obs_norm, self.critic_norm = RunningNorm(obs_dim), RunningNorm(critic_dim)
        self.return_norm = RunningNorm(1)
        self.actor, self.critic = mlp(obs_dim, actions, hidden), mlp(critic_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((actions,), -1.0))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        if reflex:
            with torch.no_grad():
                self.actor[-1].bias[:4] = -0.5

    def distribution(self, obs):
        return Normal(self.actor(self.obs_norm(obs)), self.log_std.clamp(-5, 1).exp())

    def value(self, state):
        normalized = self.critic(self.critic_norm(state)).squeeze(-1)
        return normalized * self.return_norm.var[0].sqrt() + self.return_norm.mean[0]

    @torch.no_grad()
    def normalize_returns(self, returns, optimizer):
        """PopArt output rescaling preserves unnormalized values when statistics change."""
        old_mean, old_std = self.return_norm.mean[0].clone(), self.return_norm.var[0].sqrt().clone()
        self.return_norm.update(returns[:, None])
        mean, std = self.return_norm.mean[0], self.return_norm.var[0].sqrt()
        layer = self.critic[-1]
        scale = old_std / std
        layer.weight.mul_(scale)
        layer.bias.copy_((old_std * layer.bias + old_mean - mean) / std)
        for parameter in (layer.weight, layer.bias):
            state = optimizer.state.get(parameter, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(scale)
                state["exp_avg_sq"].mul_(scale.square())

    @torch.no_grad()
    def act(self, obs, state, deterministic=False):
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(state)


class SharedActor(nn.Module):
    """One conditional network, evaluated only at each head's decision times.

    Union input = S102 + R121; the final role bit selects the head. Conditioning
    R on S makes the joint policy autoregressive while using a single trunk.
    """

    def __init__(self, hidden=(336, 336)):
        super().__init__()
        self.norm = RunningNorm(223)
        self.network = mlp(224, 28, hidden)
        self.log_std = nn.Parameter(torch.full((28,), -1.0))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[8:12] = -0.5

    def distribution(self, obs, role):
        flag = torch.full((len(obs), 1), float(role == "reflex"), device=obs.device)
        out = self.network(torch.cat((self.norm(obs), flag), dim=-1))
        sl = slice(0, 8) if role == "slow" else slice(8, 28)
        return Normal(out[:, sl], self.log_std[sl].clamp(-5, 1).exp())
