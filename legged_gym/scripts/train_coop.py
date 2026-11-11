"""Train residual controllers on a strictly validated, frozen AllGaits base."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from legged_gym.algorithms.cooperative_ppo import CooperativePPO, PPOSettings
from legged_gym.envs.mini_cheetah.coop_allgaits import MiniCheetahCoopAllGaits
from legged_gym.envs.mini_cheetah.coop_config import CoopSettings
from legged_gym.utils.coop_io import code_commit, load_base
from legged_gym.utils.helpers import class_to_dict, resolve_rl_device, set_seed

STAGES = (
    "reflex_only",
    "slow_only",
    "sequential",
    "cooperative_spatial",
    "cooperative_phase",
    "cooperative_phase_optional",
    "no_message",
    "same_rate_100",
    "same_rate_20",
    "monolithic",
    "independent_merge",
    "base",
)


def stage_settings(settings, stage):
    settings.slow_enabled = stage not in ("base", "reflex_only")
    settings.reflex_enabled = stage not in ("base", "slow_only")
    if stage == "no_message":
        settings.messages = False
    if stage in ("cooperative_phase", "cooperative_phase_optional"):
        settings.phase_enabled = True
    if stage == "same_rate_100":
        settings.slow_interval = settings.reflex_interval = 1
    if stage == "same_rate_20":
        settings.slow_interval = settings.reflex_interval = 5
    settings.validate()
    return settings


def make_runner(
    base_path,
    settings,
    params,
    stage,
    num_envs=64,
    seed=40,
    device="cpu",
    episode_seconds=20,
    speed=None,
    physics_dt=None,
):
    set_seed(seed)
    base, cfg, audit = load_base(base_path, num_envs, device)
    cfg.seed = seed
    cfg.env.episode_length_s = episode_seconds
    cfg.gait.name = "random"
    cfg.gait.resampling_time = episode_seconds + 1  # primary experiments use a fixed gait per episode
    cfg.domain_rand.push_robots = False  # new path uses explicitly scheduled force pushes
    if speed is not None:
        cfg.commands.ranges = [[speed, speed], [0, 0], [0, 0]]
        cfg.commands.limit_ranges = cfg.commands.ranges
    if physics_dt is not None:
        decimation = round(0.01 / physics_dt)
        if abs(physics_dt * decimation - 0.01) > 1e-10:
            raise ValueError("physics_dt must divide the 10ms control interval")
        cfg.sim.dt, cfg.control.decimation = physics_dt, decimation
    env = MiniCheetahCoopAllGaits(cfg, stage_settings(settings, stage))
    runner = CooperativePPO(env, base, audit, params, device, stage)
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--config", help="JSON with settings and ppo objects; omitted fields use dataclass defaults")
    parser.add_argument("--stage", choices=STAGES, default="cooperative_spatial")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--episode-seconds", type=float, default=20)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--physics-dt", type=float)
    parser.add_argument("--terrain", choices=["flat", "raised", "lowered", "irregular", "friction"])
    parser.add_argument("--terrain-seed", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume")
    parser.add_argument("--reflex-checkpoint")
    parser.add_argument("--slow-checkpoint")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_every, args.torch_threads) < 1:
        parser.error("Counts must be positive")
    if args.stage in ("base", "independent_merge"):
        parser.error("Use eval_coop for methods without additional optimization")
    if args.resume and (args.reflex_checkpoint or args.slow_checkpoint):
        parser.error("Choose exact resume or branch warm-starts, not both")
    if args.stage == "sequential" and not (args.reflex_checkpoint or args.resume):
        parser.error("Sequential training requires --reflex-checkpoint")
    torch.set_num_threads(args.torch_threads)
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    if set(config) - {"settings", "ppo"}:
        parser.error("Config accepts only settings and ppo objects")
    settings = CoopSettings(**{"fixed_start": True, "gait_names": ["walk", "trot"], **config.get("settings", {})})
    params = PPOSettings(**config.get("ppo", {}))
    for field in ("terrain", "terrain_seed"):
        if getattr(args, field) is not None:
            setattr(settings, field, getattr(args, field))
    if args.rollout_steps is not None:
        params.rollout_steps = args.rollout_steps
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; use a new directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    runner = make_runner(
        args.base_checkpoint,
        settings,
        params,
        args.stage,
        args.num_envs,
        args.seed,
        resolve_rl_device(args.device),
        args.episode_seconds,
        args.speed,
        args.physics_dt,
    )
    try:
        if args.resume:
            runner.load(args.resume)
        else:
            for role in ("slow", "reflex"):
                path = getattr(args, role + "_checkpoint")
                if path:
                    runner.load(path, resume=False, roles=[role])
        resolved = dict(
            settings=asdict(settings),
            ppo=asdict(params),
            stage=args.stage,
            seed=args.seed,
            environment=class_to_dict(runner.env.cfg),
            audit=runner.audit,
            device=str(runner.device),
            warm_starts={role: getattr(args, role + "_checkpoint") for role in ("slow", "reflex")},
            code_commit=code_commit(),
            actor_observations="ideal simulated velocity/contact; optional corruption",
            hardware_validated=False,
            actor_parameter_counts=(
                {"joint": sum(p.numel() for p in runner.shared.parameters())}
                if runner.shared
                else {
                    role: sum(p.numel() for p in agent.actor.parameters()) + agent.log_std.numel()
                    for role, agent in runner.agents.items()
                }
            ),
        )
        (out / "config_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
        (out / "base_checkpoint_sha256.txt").write_text(runner.audit["checkpoint_sha256"] + "\n")
        (out / "code_commit.txt").write_text(code_commit() + "\n")
        (out / "observation_schema.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "base": 68,
                    "critic": 147,
                    "leg_order": ["FR", "FL", "HR", "HL"],
                    "actor_slip": "zero; no oracle foot velocity",
                    "slow": {
                        "dim": 102,
                        "fields": [
                            "base68",
                            "gait9",
                            "events_ema8",
                            "slip_zero4",
                            "gate_ema4",
                            "foot_magnitude_ema1",
                            "previous_slow8",
                        ],
                    },
                    "reflex": {
                        "dim": 121,
                        "fields": [
                            "base68",
                            "current_slow8",
                            "contact_mismatch4",
                            "events8",
                            "contact_age4",
                            "slip_zero4",
                            "previous_gate_xyz16",
                            "gait9",
                        ],
                    },
                },
                indent=2,
            )
            + "\n"
        )
        stats_path, episodes_path = out / "policy_stats.csv", out / "episode_metrics.csv"
        for _ in range(args.iterations):
            started = time.perf_counter()
            buffers, episodes, steps = runner.collect()
            stats = runner.update(buffers)
            stats.update(
                iteration=runner.iteration,
                total_env_steps=runner.total_steps,
                env_steps=steps,
                env_steps_per_second=steps / (time.perf_counter() - started),
                mean_slow_duration=sum(runner.last_durations["slow"]) / len(runner.last_durations["slow"]),
            )
            append_rows(stats_path, [stats])
            append_rows(episodes_path, [{"iteration": runner.iteration, **row} for row in episodes])
            print(json.dumps(stats), flush=True)
            if runner.iteration % args.save_every == 0:
                runner.save(out / "checkpoints" / f"model_{runner.iteration}.pt")
        runner.save(out / "checkpoints" / "last.pt")
    finally:
        runner.env.close()


def append_rows(path, rows):
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
