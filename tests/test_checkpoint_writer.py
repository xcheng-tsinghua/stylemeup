import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from checkpoint_writer import AsyncCheckpointWriter


class CheckpointWriterTests(unittest.TestCase):
    def test_background_snapshot_busy_skip_and_exit_flush(self):
        entered, release = threading.Event(), threading.Event()
        original_save = torch.save

        def slow_save(checkpoint, path):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test save timed out')
            original_save(checkpoint, path)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            tensor = torch.tensor([1.])
            with patch('checkpoint_writer.torch.save', side_effect=slow_save):
                with AsyncCheckpointWriter(path) as writer:
                    try:
                        self.assertTrue(writer.submit({'optimizer': {0: [tensor]}, 'epoch': 0}))
                        self.assertTrue(entered.wait(5))
                        tensor.add_(10)
                        with self.assertLogs(level='WARNING'):
                            self.assertFalse(writer.submit({'epoch': 1}))
                    finally:
                        release.set()
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['optimizer'][0][0].item(), 1.)
            self.assertFalse(path.with_name('last.pt.tmp').exists())

    def test_write_and_replace_failure_preserve_old_checkpoint_and_retry(self):
        for stage in ('write', 'replace'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'last.pt'
                torch.save({'epoch': 0}, path)
                target = 'checkpoint_writer.torch.save' if stage == 'write' else 'checkpoint_writer.Path.replace'
                with AsyncCheckpointWriter(path) as writer:
                    with self.assertLogs(level='WARNING'), patch(target, side_effect=OSError('disk failure')):
                        self.assertTrue(writer.submit({'epoch': 1}))
                        writer.pending.result(timeout=5)
                    self.assertEqual(torch.load(path, weights_only=False)['epoch'], 0)
                    self.assertFalse(path.with_name('last.pt.tmp').exists())
                    self.assertTrue(writer.submit({'epoch': 2}))
                self.assertEqual(torch.load(path, weights_only=False)['epoch'], 2)

    def test_snapshot_failure_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, AsyncCheckpointWriter(Path(tmp) / 'last.pt') as writer:
            with self.assertLogs(level='WARNING'), patch('checkpoint_writer.cpu_snapshot', side_effect=RuntimeError('copy failed')):
                self.assertFalse(writer.submit({'epoch': 0}))
            self.assertTrue(writer.submit({'epoch': 1}))


if __name__ == '__main__':
    unittest.main()
