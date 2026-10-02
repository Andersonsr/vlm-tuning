import argparse
import os
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from model.encoders import (
    crop_transform, SPECTRAL_OPS, _nlm, IMAGENET_MEAN, IMAGENET_STD,
)


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def list_images(paths):
    images = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                images += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            images.append(path)
    return images


def center_crop(image, crop_size):
    w, h = image.size
    left = w // 2 - crop_size // 2
    top = h // 2 - crop_size // 2
    return TF.to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


def denormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def make_grid(tiles, ncols, tile_size):
    # tiles: list of (label, [3, H, W] tensor in [0, 1])
    label_h = 20
    nrows = (len(tiles) + ncols - 1) // ncols
    grid = Image.new('RGB', (ncols * tile_size, nrows * (tile_size + label_h)), 'white')
    draw = ImageDraw.Draw(grid)
    for i, (label, x) in enumerate(tiles):
        r, c = divmod(i, ncols)
        tile = TF.to_pil_image(x).resize((tile_size, tile_size))
        grid.paste(tile, (c * tile_size, r * (tile_size + label_h) + label_h))
        draw.text((c * tile_size + 4, r * (tile_size + label_h) + 4), label, fill='black')
    return grid


def save_examples(path, out_dir, crop_size, n_random, max_shift, tile_size, save_tiles):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    if w < crop_size or h < crop_size:
        print(f'skipping {path}: size ({w}, {h}) smaller than crop size {crop_size}')
        return

    name = os.path.splitext(os.path.basename(path))[0]
    image_dir = os.path.join(out_dir, name)
    os.makedirs(image_dir, exist_ok=True)

    original = center_crop(image, crop_size)

    # every geometric variant: 4 rotations x (no flip, horizontal flip)
    geometric = []
    for flip in (False, True):
        flipped = TF.hflip(original) if flip else original
        for k in range(4):
            label = f'rot{k * 90}' + (' + hflip' if flip else '')
            geometric.append((label, torch.rot90(flipped, k, dims=(1, 2))))

    # each spectral op applied to the original center crop
    spectral = [('original', original), ('nlm', _nlm(original))]
    spectral += [(op_name, op(original)) for op_name, op in SPECTRAL_OPS.items()]

    # full training pipeline (random shift + geometric + spectral), as returned by crop_transform
    pipeline = [('original', original)]
    for i in range(n_random):
        x = crop_transform([path], crop_size, 16, max_shift, geometric=True, spectral=True)
        pipeline.append((f'pipeline {i}', denormalize(x)))

    ncols = 4
    for group_name, tiles in [('geometric', geometric), ('spectral', spectral), ('pipeline', pipeline)]:
        make_grid(tiles, ncols, tile_size).save(os.path.join(image_dir, f'{group_name}_grid.png'))
        if save_tiles:
            for label, x in tiles:
                TF.to_pil_image(x).save(os.path.join(image_dir, f'{group_name}_{label.replace(" ", "_")}.png'))

    print(f'saved examples for {path} to {image_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save examples of the geometric and spectral augmentations used by crop_transform')
    parser.add_argument('--images', type=str, nargs='+', required=True, help='image files and/or directories')
    parser.add_argument('--num_images', type=int, default=4, help='number of images randomly sampled from --images')
    parser.add_argument('--output_dir', type=str, default='augmentation_examples')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--num_random', type=int, default=7, help='random pipeline samples per image')
    parser.add_argument('--random_shift', type=float, default=0.2, help='max shift used in the pipeline examples')
    parser.add_argument('--tile_size', type=int, default=256, help='size of each tile in the grids')
    parser.add_argument('--save_tiles', action='store_true', default=False, help='also save every example as a separate image')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    images = list_images(args.images)
    if len(images) > args.num_images:
        images = random.sample(images, args.num_images)

    for path in images:
        save_examples(path, args.output_dir, args.crop_size, args.num_random, args.random_shift, args.tile_size, args.save_tiles)
