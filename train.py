"""Convenient QMUL training with automatic checkpoints and SwanLab tracking."""
import argparse
import os
from pathlib import Path
import sys
import uuid
import warnings

ROOT = Path(__file__).resolve().parent


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected True or False')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, default=50, help='Total target epochs, including resumed epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of meta-training episodes per batch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--swanlab_api_key', default='IV17QULap1Ekc1OF7mN88', help='Defaults to SWANLAB_API_KEY or existing login')
    parser.add_argument('--swanlab_project', default='StyleMeUp')
    parser.add_argument('--dataset', choices=['shoe', 'chair'], required=True)
    parser.add_argument('--resume', type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--root', type=Path, default=Path(r'/opt/data/private/data_set/sketch_retrieval/qmul_v2_fit'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs', help='Checkpoint search root; new runs get a unique subdirectory')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fast_lr', type=float, default=.005)
    parser.add_argument('--eval_batch', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=0, help='Smoke test only; 0 means all batches')
    args = parser.parse_args(argv)
    if min(args.epoch, args.batch_size, args.eval_batch) < 1 or args.max_steps < 0:
        parser.error('epoch and batch sizes must be positive; max_steps must be nonnegative')
    if args.lr <= 0 or args.fast_lr <= 0:
        parser.error('learning rates must be positive')
    return args


def find_checkpoint(args, torch, engine):
    # Only load local, trusted training checkpoints (they include Python RNG state).
    candidates = sorted(args.output.rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            config = checkpoint['config']
            if config['category'] != args.dataset:
                continue
            if Path(config['root']).resolve() != args.root.resolve():
                continue
            identity = checkpoint.get('swanlab')
            if identity and identity['project'] != args.swanlab_project:
                continue
            for key in ('model', 'regulariser', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                if key not in checkpoint:
                    raise ValueError(f'missing {key}')
            if not isinstance(checkpoint['epoch'], int) or checkpoint['epoch'] < 0:
                raise ValueError('invalid epoch')
            model, reg = engine.Objective(), engine.Regulariser()
            model.load_state_dict(checkpoint['model'])
            reg.load_state_dict(checkpoint['regulariser'])
            optimizer = torch.optim.Adam(list(model.parameters()) + list(reg.parameters()))
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Resuming {path} after epoch {checkpoint["epoch"] + 1}', flush=True)
            return path
        except Exception as exc:
            warnings.warn(f'Skipping unsuitable checkpoint {path}: {exc}')
    warnings.warn('No suitable checkpoint found; starting training from scratch.')
    return None


class SwanLabTracker:
    def __init__(self, swanlab, project):
        self.swanlab = swanlab
        self.project = project
        self.identity = None
        self.started = False

    def start(self, args, checkpoint):
        previous = checkpoint.get('swanlab') if checkpoint else None
        options = dict(project=self.project, config=vars(args), mode='online',
                       log_dir=str(Path(args.output) / 'swanlog'))
        if previous:
            options.update(id=previous['id'], resume='must')
        else:
            if checkpoint:
                warnings.warn('Legacy checkpoint has no SwanLab ID; creating a new experiment.')
            options.update(name=Path(args.output).name, resume='never')
        run = self.swanlab.init(**options)
        self.started = True
        self.identity = dict(id=run.id, project=self.project)
        # Replay the last successful checkpoint on restart; asynchronous saves
        # may lag cloud logging. SwanLab ignores already recorded steps.
        if previous and checkpoint.get('record'):
            self.log(checkpoint['record'])

    def log(self, record):
        self.swanlab.log({'Acc@1': record['top1'] * 100,
                          'Acc@5': record['top5'] * 100,
                          'loss': record['loss'], 'epoch': record['epoch']},
                         step=record['epoch'])

    def finish(self):
        if self.started:
            self.swanlab.finish()


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT / 'Code'))
    import torch
    import run_qmul as engine
    try:
        import swanlab
    except ImportError as exc:
        raise SystemExit('Install training dependencies: pip install -r requirements-qmul.txt') from exc
    checkpoint = find_checkpoint(args, torch, engine) if args.resume else None
    output = checkpoint.parent if checkpoint else args.output / f'{args.dataset}-{uuid.uuid4().hex[:12]}'
    if args.swanlab_api_key:
        swanlab.login(api_key=args.swanlab_api_key, save=False)
    tracker = SwanLabTracker(swanlab, args.swanlab_project)
    command = ['--root', str(args.root.resolve()), '--category', args.dataset,
               '--output', str(output.resolve()), '--epochs', str(args.epoch),
               '--meta-batch', str(args.batch_size), '--seed', str(args.seed),
               '--lr', str(args.lr), '--fast-lr', str(args.fast_lr),
               '--eval-batch', str(args.eval_batch), '--max-steps', str(args.max_steps)]
    if checkpoint:
        command += ['--resume', str(checkpoint.resolve())]
    try:
        engine.main(command, tracker=tracker)
    finally:
        tracker.finish()


if __name__ == '__main__':
    main()

