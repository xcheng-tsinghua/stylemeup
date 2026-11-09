import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import warnings
import torch
import train


class TrainingEntryTests(unittest.TestCase):
    def test_boolean_and_validation(self):
        for flags, expected in [([], False), (['--resume'], True), (['--resume', 'True'], True), (['--resume', 'False'], False)]:
            self.assertEqual(train.parse_args(['--dataset', 'chair'] + flags).resume, expected)
        with self.assertRaises(SystemExit):
            train.parse_args(['--dataset', 'chair', '--batch_size', '0'])

    def test_resume_identity_and_metrics(self):
        sdk = Mock()
        sdk.init.return_value.id = 'original'
        tracker = train.SwanLabTracker(sdk, 'project')
        record = dict(epoch=3, top1=.25, top5=.75, loss=1.2)
        tracker.start(argparse.Namespace(output='runs/test'),
                      {'swanlab': {'id': 'original', 'project': 'project'}, 'record': record})
        self.assertEqual(sdk.init.call_args.kwargs['resume'], 'must')
        self.assertEqual(sdk.init.call_args.kwargs['id'], 'original')
        sdk.log.assert_called_once_with({'Acc@1': 25., 'Acc@5': 75., 'loss': 1.2, 'epoch': 3}, step=3)
        tracker.finish()
        sdk.finish.assert_called_once()

    def test_checkpoint_fallback_and_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = Mock()
            engine.Objective.side_effect = lambda: torch.nn.Linear(2, 2)
            engine.Regulariser.side_effect = lambda: torch.nn.Linear(2, 2)
            model, reg = engine.Objective(), engine.Regulariser()
            optimizer = torch.optim.Adam(list(model.parameters()) + list(reg.parameters()))
            checkpoint = dict(config={'category': 'chair', 'root': str(root)}, epoch=2,
                              model=model.state_dict(), regulariser=reg.state_dict(), optimizer=optimizer.state_dict(),
                              python_rng=None, numpy_rng=None, torch_rng=None, cuda_rng=[],
                              swanlab={'project': 'project', 'id': 'run'})
            good = root / 'good.pt'
            torch.save(checkpoint, good)
            (root / 'broken.pt').write_bytes(b'broken')
            args = argparse.Namespace(output=root, dataset='chair', root=root, swanlab_project='project')
            with warnings.catch_warnings(record=True):
                self.assertEqual(train.find_checkpoint(args, torch, engine), good)
                args.dataset = 'shoe'
                self.assertIsNone(train.find_checkpoint(args, torch, engine))


if __name__ == '__main__':
    unittest.main()
