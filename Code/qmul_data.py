"""Read the local QMUL stroke-3 export without changing the source dataset."""
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


def image_tensor(image):
    image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1


def raster(path):
    points = np.loadtxt(path, ndmin=2)
    if points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'Invalid stroke-3 data: {path}')
    canvas = Image.new('L', (256, 256), 0)
    draw = ImageDraw.Draw(canvas)
    for i, (x, y, _) in enumerate(points):
        if i and points[i - 1, 2] != 1:
            draw.line([tuple(points[i - 1, :2]), (x, y)], fill=255, width=1)
        else:
            draw.point((x, y), fill=255)
    # Match the original white-stroke / black-background convention.
    return canvas.filter(ImageFilter.MaxFilter(3))


class QMUL(Dataset):
    def __init__(self, root, category, split):
        self.root = Path(root) / category
        self.split = split
        self.sketches = sorted((self.root / 'sketch_s3' / split).rglob('*.txt'))
        self.groups = defaultdict(list)
        for path in self.sketches:
            self.groups[path.stem.rsplit('_', 1)[0]].append(path)
        self.ids = sorted(self.groups)
        self.photos = {p.stem: p for p in (self.root / 'photo' / split).rglob('*.png')}
        assert self.ids and set(self.ids) <= self.photos.keys()
        other = 'test' if split == 'train' else 'train'
        other_ids = {p.stem.rsplit('_', 1)[0] for p in (self.root / 'sketch_s3' / other).rglob('*.txt')}
        assert not set(self.ids) & other_ids, 'Train/test instance leakage'
        self.episodes = [i for i in self.ids if all(any(p.stem == f'{i}_{k}' for p in self.groups[i]) for k in (1, 2, 3))]

    def __len__(self):
        return len(self.episodes) if self.split == 'train' else len(self.sketches)

    def __getitem__(self, index):
        instance = self.episodes[index]
        negative = random.choice(self.ids)
        while negative == instance:
            negative = random.choice(self.ids)
        paths = {p.stem: p for p in self.groups[instance]}
        images = [Image.open(self.photos[instance]), Image.open(self.photos[negative])]
        images += [raster(paths[f'{instance}_{k}']) for k in (1, 2, 3)]
        if random.random() < .5:
            images = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in images]
        return torch.stack([image_tensor(im) for im in images])

    def audit(self):
        return dict(split=self.split, instances=len(self.ids), sketches=len(self.sketches),
                    three_style_episodes=len(self.episodes), photo_files=len(self.photos),
                    excluded_gallery_photos=len(self.photos) - len(self.ids))
