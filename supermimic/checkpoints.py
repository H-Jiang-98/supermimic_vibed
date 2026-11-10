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
