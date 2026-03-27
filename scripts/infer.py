from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesturehydra.cli import ensure_paths_exist, run_infer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GestureHYDRA inference entrypoint.")
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/data/streamer.yaml",
            "configs/model/gesturehydra_base.yaml",
            "configs/inference/base.yaml",
        ],
        help="Path to a YAML config file. Can be passed multiple times.",
    )
    parser.add_argument("--input-audio", default=None, help="Path to input audio.")
    parser.add_argument("--transcript", default=None, help="Optional transcript text.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_paths_exist(args.config)
    run_infer(args.config, args.input_audio, args.transcript)


if __name__ == "__main__":
    main()
