from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesturehydra.cli import ensure_paths_exist, run_prepare_streamer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the local Streamer dataset.")
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/data/streamer.yaml"],
        help="Path to a YAML config file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory containing the raw dataset export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_paths_exist(args.config)
    run_prepare_streamer(args.config, args.source_dir)


if __name__ == "__main__":
    main()
