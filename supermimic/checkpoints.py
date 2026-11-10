"""Checkpoint selection independent of training's save interval."""

from pathlib import Path


def checkpoint_iteration(path):
    return int(Path(path).stem.removeprefix("ckpt_"))


def pending_checkpoints(paths, done, every=1):
    if every < 1:
        raise ValueError("every must be >= 1")
    ordered = sorted(paths, key=checkpoint_iteration)
    # Count all checkpoints so restarting the watcher preserves the selection.
    return [path for index, path in enumerate(ordered, 1)
            if index % every == 0 and checkpoint_iteration(path) not in done]


def checkpoint_obs_mode(checkpoint, expected_mode=None):
    """Validate the checkpoint's observation mode and policy input dimension."""
    from .training_config import OBS_DIMS

    mode = checkpoint["args"]["obs_mode"]
    if not isinstance(mode, str) or mode not in OBS_DIMS:
        raise ValueError(f"Unsupported checkpoint obs_mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"Observation mode mismatch: checkpoint={mode}, configured={expected_mode}")
    mean = checkpoint["pi"]["obs_norm.mean"]
    if tuple(mean.shape) != (OBS_DIMS[mode],):
        raise ValueError(f"Checkpoint observation dimension does not match obs_mode={mode}")
    return mode
