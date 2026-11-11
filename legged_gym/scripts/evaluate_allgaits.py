"""Compare explicit AllGaits checkpoints under one flat-ground validation protocol."""

import argparse
import contextlib
import csv
import json
from pathlib import Path

from legged_gym.scripts.play import play
from legged_gym.utils.coop_io import file_hash, source_hash
from legged_gym.utils.helpers import get_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--gaits", nargs="+", default=["walk", "trot"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = [Path(p).resolve() for p in args.checkpoints]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate checkpoint paths")
    for path in paths:
        if not path.is_file() or not (path.parent / "run_config.json").is_file():
            parser.error(f"Checkpoint and run_config.json are required: {path}")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new/empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    for index, path in enumerate(paths):
        snapshot = json.loads((path.parent / "run_config.json").read_text())
        if snapshot["task"] != "mini_cheetah_allgaits":
            parser.error(f"Not an AllGaits checkpoint: {path}")
        label = f"{index:02d}_{path.parent.parent.name}_{path.stem}"
        profile = snapshot["train"].get("baseline_profile", "legacy")
        provenance.append(
            dict(
                label=label,
                checkpoint=str(path),
                checkpoint_sha256=file_hash(path),
                config_sha256=file_hash(path.parent / "run_config.json"),
                profile=profile,
            )
        )
        for gait in args.gaits:
            directory = out / label / gait
            directory.mkdir(parents=True, exist_ok=True)
            cli = [
                "--task",
                "mini_cheetah_allgaits",
                "--checkpoint_path",
                str(path),
                "--gait",
                gait,
                "--evaluate",
                "--headless",
                "--rl_device",
                "cpu",
                "--num_envs",
                str(args.num_envs),
                "--steps",
                str(args.steps),
                "--eval_seeds",
                *map(str, args.seeds),
                "--eval_output",
                str(directory),
            ]
            with (directory / "console.log").open("w") as stream, contextlib.redirect_stdout(stream):
                report = play(get_args(cli, playback=True))
            for row in report["summary"]:
                if row["seed"] != "all":
                    continue
                rows.append(dict(label=label, profile=profile, checkpoint=str(path), gait=gait, **row))
                print(
                    f"{label} {gait} {row['scenario']}: failure={row['failure_rate']:.1%}, "
                    f"vx MAE={row['mae_x']}, yaw MAE={row['mae_yaw']}, "
                    f"clip={row['action_clipped_fraction']}, lateral={row['mean_max_lateral_deviation_m']:.3f} m",
                    flush=True,
                )
            with (out / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                checkpoints=provenance,
                gaits=args.gaits,
                seeds=args.seeds,
                num_envs=args.num_envs,
                tracking_steps=args.steps,
                source_sha256=source_hash(),
                split="validation",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Comparison: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
