"""Checkpoint selection independent of training's save interval."""

from pathlib import Path

WM_FORMAT_VERSION = 2


def checkpoint_world_model(checkpoint, *, dt, num_substeps):
    """Reject incompatible training bundles before loading any weights/optimizers."""
    if checkpoint.get("wm_format_version") != WM_FORMAT_VERSION:
        raise ValueError("Incompatible world-model checkpoint: gencoord v2 is required; start a new training run")
    expected = {"dt": dt, "num_substeps": num_substeps}
    if checkpoint.get("wm_config") != expected:
        raise ValueError(f"World-model integration configuration mismatch: expected {expected}")
    optimizers = checkpoint.get("opt_wm")
    if not isinstance(optimizers, dict) or set(optimizers) != {"adam", "muon"}:
        raise ValueError("World-model resume requires Adam and Muon states from last.pt")


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
