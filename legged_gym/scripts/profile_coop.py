"""Measure actual simulator-plus-learner throughput before choosing a research budget."""

import argparse
import json
import time

from legged_gym.algorithms.cooperative_ppo import PPOSettings
from legged_gym.envs.mini_cheetah.coop_config import CoopSettings
from legged_gym.scripts.train_coop import make_runner
from legged_gym.utils.helpers import resolve_rl_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--num-envs", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=50)
    args = parser.parse_args()
    if args.rollouts < 2:
        parser.error("Use at least two rollouts (the first is warm-up)")
    for count in args.num_envs:
        runner = make_runner(
            args.base_checkpoint,
            CoopSettings(fixed_start=True, terrain="irregular", randomize_terrain=True, gait_names=["walk", "trot"]),
            PPOSettings(rollout_steps=args.rollout_steps),
            "cooperative_spatial",
            count,
            40,
            resolve_rl_device(args.device),
        )
        records = []
        try:
            for _ in range(args.rollouts):
                start = time.perf_counter()
                buffers, _, steps = runner.collect()
                collected = time.perf_counter()
                runner.update(buffers)
                records.append(
                    {"steps": steps, "collect_s": collected - start, "update_s": time.perf_counter() - collected}
                )
            records = records[1:]
            print(
                json.dumps(
                    {
                        "num_envs": count,
                        "device": str(runner.device),
                        "env_steps_per_second": sum(r["steps"] for r in records)
                        / sum(r["collect_s"] + r["update_s"] for r in records),
                        "rollouts": records,
                    }
                ),
                flush=True,
            )
        finally:
            runner.env.close()


if __name__ == "__main__":
    main()
