# SPDX-License-Identifier: BSD-3-Clause
"""Train Mini Cheetah CPG or joint-space PPO."""

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    env, _ = task_registry.make_env(args.task, args, env_cfg, train_cfg)
    monitor = None
    try:
        runner, cfg = task_registry.make_alg_runner(env, args.task, args, train_cfg)
        print(f"MuJoCo CPU: {env.num_envs} envs / {env.sim.batch.nthread} workers; PPO: {runner.device}")
        print(f"Logs: {runner.logger.log_dir}")
        if cfg.viewer.enabled:
            from legged_gym.utils.training_monitor import TrainingMonitor

            monitor = TrainingMonitor(env, runner, cfg.viewer, task=args.task)
            print(f"Training monitor: {monitor.url}", flush=True)
        else:
            print("Viser monitor disabled (--headless). Remove --headless to enable the web viewer.", flush=True)
        runner.learn(cfg.runner.max_iterations, init_at_random_ep_len=not (args.resume or args.checkpoint_path))
    finally:
        if monitor is not None:
            monitor.close()
        env.close()


def main():
    train(get_args())


if __name__ == "__main__":
    main()
