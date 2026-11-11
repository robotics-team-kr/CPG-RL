"""Seed-level summaries and paired hierarchical bootstrap; episodes are not training replicates."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("completed", "failure", "progress", "energy_pos", "cot", "tracking_rmse", "intervention")


def numeric(value):
    return 1.0 if value == "True" else 0.0 if value == "False" else float(value)


def paired_bootstrap(rows, reference, method, metric, samples=2000, seed=123):
    grouped = defaultdict(dict)
    for row in rows:
        if row["method"] in (reference, method):
            grouped[(row["method"], row["training_seed"])][row["scenario"]] = numeric(row[metric])
    reference_seeds = {key[1] for key in grouped if key[0] == reference}
    method_seeds = {key[1] for key in grouped if key[0] == method}
    seeds = sorted(method_seeds if len(reference_seeds) == 1 else reference_seeds & method_seeds)
    deltas = []
    for s in seeds:
        reference_seed = next(iter(reference_seeds)) if len(reference_seeds) == 1 else s
        a, b = grouped[(reference, reference_seed)], grouped[(method, s)]
        pairs = [(b[k] - a[k]) for k in sorted(a.keys() & b.keys()) if np.isfinite(a[k]) and np.isfinite(b[k])]
        if pairs:
            deltas.append(np.array(pairs))
    if not deltas:
        return dict(delta=None, lower=None, upper=None, training_seeds=0, paired_episodes=0)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        selected = rng.integers(len(deltas), size=len(deltas))
        bootstrap.append(np.mean([rng.choice(deltas[i], size=len(deltas[i]), replace=True).mean() for i in selected]))
    return dict(
        delta=float(np.mean([d.mean() for d in deltas])),
        lower=float(np.quantile(bootstrap, 0.025)),
        upper=float(np.quantile(bootstrap, 0.975)),
        training_seeds=len(deltas),
        paired_episodes=sum(len(d) for d in deltas),
    )


def analyze(path, reference, out, samples=2000, plots=False):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No evaluation rows")
    identities = [(r["method"], r["training_seed"], r["scenario"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate method/seed/scenario rows")
    methods = sorted({r["method"] for r in rows})
    if reference not in methods:
        raise ValueError("Reference method missing")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in methods:
        for metric in METRICS:
            by_seed = defaultdict(list)
            for row in rows:
                if row["method"] == method and np.isfinite(numeric(row[metric])):
                    by_seed[row["training_seed"]].append(numeric(row[metric]))
            means = [np.mean(values) for values in by_seed.values()]
            summary.append(
                dict(
                    method=method,
                    metric=metric,
                    mean=float(np.mean(means)) if means else None,
                    seed_means={s: float(np.mean(v)) for s, v in by_seed.items()},
                    training_seeds=len(means),
                    valid_episodes=sum(map(len, by_seed.values())),
                )
            )
    comparisons = [
        dict(
            reference=reference,
            method=method,
            metric=metric,
            **paired_bootstrap(rows, reference, method, metric, samples),
        )
        for method in methods
        if method != reference
        for metric in METRICS
    ]
    interaction_rows = []
    required = ("base", "reflex_only", "slow_only", "cooperative_spatial")
    if all(method in methods for method in required):
        lookup = {(r["method"], r["training_seed"], r["scenario"]): numeric(r["completed"]) for r in rows}
        seeds_by_method = {m: {r["training_seed"] for r in rows if r["method"] == m} for m in required}
        synthetic = []
        for row in rows:
            if row["method"] != "cooperative_spatial":
                continue
            values = []
            for method in required:
                seeds = seeds_by_method[method]
                seed = next(iter(seeds)) if method == "base" and len(seeds) == 1 else row["training_seed"]
                key = (method, seed, row["scenario"])
                if key not in lookup:
                    break
                values.append(lookup[key])
            if len(values) == 4:
                contrast = values[3] - values[2] - values[1] + values[0]
                for name, value in (("zero", 0), ("interaction", contrast)):
                    synthetic.append(
                        dict(
                            method=name,
                            training_seed=row["training_seed"],
                            scenario=row["scenario"],
                            completed=str(value),
                        )
                    )
        interaction_rows.append(
            dict(
                contrast="S_interaction_completion",
                **paired_bootstrap(synthetic, "zero", "interaction", "completed", samples),
            )
        )
    for control, label in (
        ("independent_merge", "G_joint_training_completion"),
        ("no_message", "G_communication_completion"),
    ):
        if control in methods and "cooperative_spatial" in methods:
            interaction_rows.append(
                dict(contrast=label, **paired_bootstrap(rows, control, "cooperative_spatial", "completed", samples))
            )
    with (output / "synergy_interaction.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["contrast", "delta", "lower", "upper", "training_seeds", "paired_episodes"]
        )
        writer.writeheader()
        writer.writerows(interaction_rows)
    report = dict(
        summary=summary,
        comparisons=comparisons,
        caveat="COT pairs condition on joint completion. Report completion rates alongside COT. "
        "One-seed intervals reflect only scenario variability, not training uncertainty.",
    )
    (output / "bootstrap_ci.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "bootstrap_ci.csv").open("w", newline="") as f:
        if comparisons:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    for field in ("gait", "terrain"):
        groups = defaultdict(list)
        for row in rows:
            groups[(row["method"], row["training_seed"], row[field])].append(row)
        with (output / f"metrics_by_{field}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "training_seed", field, "episodes", *METRICS])
            writer.writeheader()
            for (method, seed, group), values in groups.items():
                entry = dict(method=method, training_seed=seed, **{field: group}, episodes=len(values))
                for metric in METRICS:
                    valid = [numeric(r[metric]) for r in values if np.isfinite(numeric(r[metric]))]
                    entry[metric] = float(np.mean(valid)) if valid else ""
                writer.writerow(entry)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric in ("completed", "cot", "tracking_rmse"):
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, method in enumerate(methods):
                values = next(r["seed_means"] for r in summary if r["method"] == method and r["metric"] == metric)
                ax.scatter([i] * len(values), list(values.values()))
            ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
            ax.set_ylabel(metric + " (one dot per training seed)")
            fig.tight_layout()
            fig.savefig(output / f"{metric}.png", dpi=200)
            plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    analyze(args.episodes, args.reference, args.out, args.samples, args.plots)


if __name__ == "__main__":
    main()
