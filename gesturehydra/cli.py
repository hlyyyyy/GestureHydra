from __future__ import annotations

import json
from pathlib import Path

from gesturehydra.config import load_config


def _print_banner(title: str, config_paths: list[str]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Loaded config files:")
    for path in config_paths:
        print(f"- {path}")
    print()


def run_train(config_paths: list[str]) -> None:
    config = load_config(config_paths)
    _print_banner("GestureHYDRA Training Entrypoint Placeholder", config_paths)
    print(json.dumps(config, indent=2))
    print()
    print("TODO: connect dataset loaders, model construction, trainer, and logging.")


def run_infer(config_paths: list[str], input_audio: str | None, transcript: str | None) -> None:
    config = load_config(config_paths)
    _print_banner("GestureHYDRA Inference Entrypoint Placeholder", config_paths)
    print(json.dumps(config, indent=2))
    print()
    print(f"Input audio   : {input_audio or '<not provided>'}")
    print(f"Transcript    : {transcript or '<not provided>'}")
    print("TODO: connect checkpoint loading, retrieval, generation, and rendering.")


def run_prepare_streamer(config_paths: list[str], source_dir: str | None) -> None:
    config = load_config(config_paths)
    _print_banner("GestureHYDRA Data Preparation Placeholder", config_paths)
    print(json.dumps(config, indent=2))
    print()
    print(f"Source directory: {source_dir or '<not provided>'}")
    print("TODO: implement dataset parsing, alignment, filtering, and metadata export.")


def ensure_paths_exist(paths: list[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        print("Warning: some config files do not exist yet:")
        for path in missing:
            print(f"- {path}")
