"""One transition per actual decision, with duration-aware GAE and separate masks."""

import numpy as np
import torch


def duration_gae(rewards, values, next_values, durations, terminated, boundaries, gamma, lam):
    rewards, values, next_values = map(np.asarray, (rewards, values, next_values))
    discount = gamma ** np.asarray(durations)
    deltas = rewards + discount * (1 - np.asarray(terminated, dtype=float)) * next_values - values
    result = np.zeros_like(deltas, dtype=float)
    carry = 0.0
    for i in reversed(range(len(result))):
        carry = deltas[i] + discount[i] * lam * (not boundaries[i]) * carry
        result[i] = carry
    return result, result + values


class DecisionBuffer:
    def __init__(self, n, gamma, lam):
        self.trajectories = [[] for _ in range(n)]
        self.gamma, self.lam = gamma, lam

    def add(self, env_id, record):
        self.trajectories[env_id].append(record)

    def tensors(self, device):
        rows = []
        for trajectory in self.trajectories:
            if not trajectory:
                continue
            advantages, returns = duration_gae(
                [r["reward"] for r in trajectory],
                [r["value"] for r in trajectory],
                [r["next_value"] for r in trajectory],
                [r["duration"] for r in trajectory],
                [r["terminal"] for r in trajectory],
                [r["boundary"] for r in trajectory],
                self.gamma,
                self.lam,
            )
            for row, adv, ret in zip(trajectory, advantages, returns):
                rows.append({**row, "advantage": adv, "return": ret})
        if not rows:
            raise ValueError("Empty decision buffer")
        result = {}
        keys = ["obs", "critic", "action", "logp", "value", "advantage", "return"]
        if "slow_due" in rows[0]:
            keys.extend(["slow_obs", "slow_action", "slow_due"])
        for key in keys:
            values = [row[key] for row in rows]
            result[key] = torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)
        result["advantage"] = (result["advantage"] - result["advantage"].mean()) / (
            result["advantage"].std(unbiased=False) + 1e-8
        )
        return result
