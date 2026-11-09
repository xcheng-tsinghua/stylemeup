"""Bounded background checkpoint writes with independent CPU snapshots."""
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = cpu_snapshot(item)
        return result
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


class AsyncCheckpointWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Finish the final write before exiting, including on training errors.
        self.executor.shutdown(wait=True)

    def submit(self, checkpoint):
        if self.pending is not None and not self.pending.done():
            logging.warning('Skipping checkpoint: previous save is still running.')
            return False
        try:
            # Snapshot before training resumes; state_dict tensors share live storage.
            snapshot = cpu_snapshot(checkpoint)
            self.pending = self.executor.submit(self._save, snapshot)
            return True
        except Exception:
            logging.warning('Skipping checkpoint: snapshot or submission failed.', exc_info=True)
            return False

    def _save(self, checkpoint):
        temporary = self.path.with_name(self.path.name + '.tmp')
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(self.path)
        except Exception:
            logging.warning('Checkpoint save failed; training continues: %s', self.path, exc_info=True)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.warning('Could not remove temporary checkpoint: %s', temporary, exc_info=True)
