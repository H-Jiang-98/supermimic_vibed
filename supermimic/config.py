"""Shared runtime settings; relative paths are resolved against the config file."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from supermimic.training_config import validate_training

PATH_KEYS = ("conda_env", "motion_dir", "robot_description_dir", "robot_usd_dir", "ground_dir", "logs_dir")


# Match quoted strings first so comment markers within paths/URLs remain untouched.
_JSON_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*.*?\*/', re.DOTALL)


def strip_comments(text: str) -> str:
    def replace(match):
        token = match.group()
        if token.startswith('"'):
            return token
        return "".join("\n" if char == "\n" else " " for char in token)
    return _JSON_TOKENS.sub(replace, text)


def load_config(path: str | Path | None = None) -> dict:
    source = Path(path or os.environ.get("SUPERMIMIC_CONFIG", REPO / "config.json")).expanduser().resolve()
    with source.open() as stream:
        settings = json.loads(strip_comments(stream.read()))
    if not isinstance(settings, dict):
        raise ValueError(f"{source}: expected a JSON object")
    settings.setdefault("logs_dir", "logs")
    required = set(PATH_KEYS)
    unknown = settings.keys() - required - {"training"}
    if unknown:
        raise ValueError(f"{source}: unknown settings: {', '.join(sorted(unknown))}")
    for key in required:
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"{source}: {key} must be a non-empty string")
    settings["training"] = validate_training(settings.get("training", {}))
    for key in PATH_KEYS:
        value = Path(os.path.expandvars(settings[key])).expanduser()
        settings[key] = str((source.parent / value).resolve())
    settings["config_file"] = str(source)
    return settings


def configure_cli() -> dict:
    """Select paths before importing simulator modules that construct asset configs."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config")
    args, _ = parser.parse_known_args()
    settings = load_config(args.config)
    os.environ["SUPERMIMIC_CONFIG"] = settings["config_file"]
    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", choices=PATH_KEYS)
    parser.add_argument("--config")
    args = parser.parse_args()
    print(load_config(args.config)[args.key])
