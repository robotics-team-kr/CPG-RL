"""Audit a trained AllGaits checkpoint without starting training."""

import argparse
import json
from pathlib import Path

from legged_gym.utils.coop_io import load_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="mini_cheetah_allgaits", choices=["mini_cheetah_allgaits"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    _, _, audit = load_base(args.checkpoint)
    value = json.dumps(audit, indent=2)
    if args.output:
        Path(args.output).write_text(value + "\n")
    print(value)


if __name__ == "__main__":
    main()
