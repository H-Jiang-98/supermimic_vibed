"""Training defaults and validation, independent of simulator dependencies."""

import argparse
import math

OBS_DIMS = {"paper": 93, "so3": 96, "se3": 99, "se23": 102}

TRAIN_DEFAULTS = {
    "obs_mode": "paper",
    "num_envs": 4096,
    "iterations": 320_000,
    "buffer_steps": 1000,
    "utd": 0.4,
    "horizon": 32,
    "goal_frames": 10,
    "hist": 10,
    "wm_window": 8,
    "wm_batches": 10,
    "wm_batch": 2048,
    "pi_batch": 4096,
    "lr": 2e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "action_noise": 0.1,
    "critic_epochs": 5,
    "critic_minibatches": 4,
    "no_critic": False,
    "warmup_steps": 64,
    "wm_only_iters": 500,
    "lr_decay_iters": 250_000,
    "ckpt_every": 2000,
    "log_every": 50,
    "seed": 0,
    "amp": "bf16",
}


def validate_training(overrides):
    if not isinstance(overrides, dict):
        raise ValueError("training must be an object")
    unknown = overrides.keys() - TRAIN_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown training settings: {', '.join(sorted(unknown))}")
    values = TRAIN_DEFAULTS | overrides
    for key, value in values.items():
        expected = type(TRAIN_DEFAULTS[key])
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"training.{key} must be {expected.__name__}")
        if expected is int:
            minimum = 0 if key in ("seed", "warmup_steps", "wm_only_iters") else 1
            if value < minimum:
                raise ValueError(f"training.{key} must be >= {minimum}")
    for key in ("utd", "lr"):
        if values[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if not 0 <= values["gamma"] < 1:
        raise ValueError("training.gamma must be in [0, 1)")
    if not 0 <= values["lam"] <= 1:
        raise ValueError("training.lam must be in [0, 1]")
    if values["action_noise"] < 0:
        raise ValueError("training.action_noise must be nonnegative")
    if values["obs_mode"] not in OBS_DIMS:
        raise ValueError(f"training.obs_mode must be one of {tuple(OBS_DIMS)}")
    if values["amp"] not in ("none", "bf16"):
        raise ValueError("training.amp must be 'none' or 'bf16'")
    if values["buffer_steps"] < values["wm_window"]:
        raise ValueError("training.buffer_steps must be >= wm_window")
    if not values["no_critic"] and values["critic_minibatches"] > values["horizon"] * values["pi_batch"]:
        raise ValueError("training.critic_minibatches must be <= horizon * pi_batch")
    return values


def add_training_args(parser: argparse.ArgumentParser, overrides=None):
    defaults = validate_training(overrides or {})
    for key, default in TRAIN_DEFAULTS.items():
        if key == "no_critic":
            group = parser.add_mutually_exclusive_group()
            group.add_argument("--no_critic", action="store_true", help="Disable critic bootstrap/updates")
            group.add_argument("--use_critic", dest="no_critic", action="store_false", help="Override no_critic=true")
        else:
            options = {"type": type(default), "help": f"Code default: {default}"}
            if key == "obs_mode":
                options["choices"] = tuple(OBS_DIMS)
            if key == "amp":
                options["choices"] = ("none", "bf16")
            parser.add_argument(f"--{key}", **options)
    parser.set_defaults(**defaults)
