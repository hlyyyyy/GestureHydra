from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesturehydra.cli import ensure_paths_exist, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GestureHYDRA training entrypoint.")
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/data/streamer.yaml",
            "configs/model/gesturehydra_base.yaml",
            "configs/train/base.yaml",
        ],
        help="Path to a YAML config file. Can be passed multiple times.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_paths_exist(args.config)
    run_train(args.config)


if __name__ == "__main__":
    main()
