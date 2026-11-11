"""Paired, deterministic first-episode evaluation from explicit JSON manifests."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from legged_gym.algorithms.cooperative_ppo import PPOSettings
from legged_gym.envs.mini_cheetah.coop_config import CoopSettings
from legged_gym.scripts.train_coop import append_rows, make_runner
from legged_gym.utils.coop_io import code_commit, file_hash, source_hash
from legged_gym.utils.helpers import resolve_rl_device


def evaluate(models, scenarios, out, device="cpu", save_traces=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be a new/empty directory")
    out.mkdir(parents=True, exist_ok=True)
    cases = scenarios["scenarios"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Scenario IDs must be unique")
    if scenarios.get("split") not in ("validation", "test", "smoke"):
        raise ValueError("Explicit validation/test/smoke split required")
    training_seeds = set(scenarios.get("training_terrain_seeds", []))
    if scenarios["split"] != "smoke" and any(case["terrain_seed"] in training_seeds for case in cases):
        raise ValueError("Held-out terrain seeds overlap training manifest")
    results, provenance = [], []
    for model in models["models"]:
        checkpoint = model.get("checkpoint")
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu") if checkpoint else None
        stage = model.get("stage", saved["stage"] if saved else "base")
        if stage not in ("base", "independent_merge") and not checkpoint:
            raise ValueError("Learned methods require a residual checkpoint")
        if stage == "independent_merge" and not all(model.get(role + "_checkpoint") for role in ("slow", "reflex")):
            raise ValueError("Independent merge requires both independently trained branch checkpoints")
        architecture = saved
        if stage == "independent_merge":
            architecture = torch.load(model["slow_checkpoint"], weights_only=False, map_location="cpu")
            reflex_source = torch.load(model["reflex_checkpoint"], weights_only=False, map_location="cpu")
            if architecture["params"]["hidden"] != reflex_source["params"]["hidden"]:
                raise ValueError("Independent branches must use the same configured hidden dimensions")
            for key in ("slow_scale", "foot_scale", "phase_limit", "filter_tau", "slow_interval", "reflex_interval"):
                if architecture["settings"][key] != reflex_source["settings"][key]:
                    raise ValueError(f"Independent branch action contracts differ: {key}")
        base_path = model.get("base_checkpoint", models.get("base_checkpoint"))
        paths = [p for p in (base_path, checkpoint, model.get("slow_checkpoint"), model.get("reflex_checkpoint")) if p]
        provenance.append({"model": model, "hashes": {p: file_hash(p) for p in paths}})
        for case in cases:
            # Shared scenarios define evaluation physics/sensors for every method.
            options = asdict(CoopSettings())
            if architecture:
                policy_fields = (
                    "slow_interval",
                    "reflex_interval",
                    "phase_enabled",
                    "spatial_enabled",
                    "slow_scale",
                    "foot_scale",
                    "phase_limit",
                    "filter_tau",
                    "summary_tau",
                    "messages",
                    "slow_to_reflex",
                    "reflex_to_slow",
                )
                options.update({k: architecture["settings"][k] for k in policy_fields if k in architecture["settings"]})
            options.update(scenarios.get("settings", {}))
            fields = set(asdict(CoopSettings()))
            unknown = set(case) - fields - {"id", "seed", "gait", "speed", "episode_seconds", "physics_dt"}
            if unknown:
                raise ValueError(f"Unknown scenario fields: {unknown}")
            options.update({k: v for k, v in case.items() if k in fields})
            options.update(fixed_start=True, gait_names=[case["gait"]], randomize_terrain=False)
            params = PPOSettings(**architecture["params"]) if architecture else PPOSettings()
            runner = make_runner(
                base_path,
                CoopSettings(**options),
                params,
                stage,
                1,
                case["seed"],
                device,
                case.get("episode_seconds", 20),
                case.get("speed", 0.4),
                case.get("physics_dt"),
            )
            trace = []
            try:
                if checkpoint:
                    runner.load(checkpoint, resume=False)
                for role in ("slow", "reflex"):
                    if model.get(role + "_checkpoint"):
                        runner.load(model[role + "_checkpoint"], resume=False, roles=[role])
                max_steps = runner.env.max_episode_length + 1
                for _ in range(max_steps):
                    _, _, done, info = runner.act_step()
                    if save_traces:
                        trace.append(info["trace"][0].copy())
                    if done[0]:
                        row = dict(
                            method=model["name"],
                            training_seed=model.get("training_seed", 0),
                            scenario=case["id"],
                            scenario_seed=case["seed"],
                            terrain=options["terrain"],
                            split=scenarios["split"],
                            **info["episodes"][0],
                        )
                        results.append(row)
                        append_rows(out / "metrics_per_episode.csv", [row])
                        print(json.dumps(row), flush=True)
                        break
                else:
                    raise RuntimeError("Episode did not terminate by its configured time limit")
                if trace:
                    index = len(results) - 1
                    np.savez_compressed(
                        out / f"trace_{index:05d}.npz",
                        data=np.array(trace),
                        fields=np.array(
                            [
                                "time1",
                                "position3",
                                "velocity3",
                                "command3",
                                "gate4",
                                "slow8",
                                "foot12",
                                "phase4",
                                "contact4",
                                "observed_contact4",
                                "expected_contact4",
                                "early4",
                                "late4",
                                "phase_rate4",
                                "positive_energy1",
                            ]
                        ),
                    )
            finally:
                runner.env.close()
    (out / "eval_manifest.json").write_text(
        json.dumps(
            dict(
                models=provenance,
                scenarios=scenarios,
                code_commit=code_commit(),
                source_sha256=source_hash(),
                device=str(device),
                hardware_validated=False,
            ),
            indent=2,
        )
        + "\n"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", required=True, help="JSON: base_checkpoint and models[{name,stage,checkpoint,training_seed}]"
    )
    parser.add_argument("--scenario-set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(
        json.loads(Path(args.models).read_text()),
        json.loads(Path(args.scenario_set).read_text()),
        args.out,
        resolve_rl_device(args.device),
        args.traces,
    )


if __name__ == "__main__":
    main()
