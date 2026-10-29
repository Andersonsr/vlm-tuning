"""Download Conceptual Captions 3M (CC3M) to a local folder.

Two sources are supported:
  - hf   (default): pre-packaged WebDataset mirror on the Hugging Face Hub
                    (pixparse/cc3m-wds). Images are already included, so nothing
                    is lost to dead links. ~2.9M train / ~13k val samples;
                    576 train shards (~280GB) + 16 validation shards.
  - urls          : the official Google TSVs (caption + image URL), with images
                    fetched by img2dataset. Many URLs are dead nowadays, so expect
                    noticeably fewer samples than the original 3.3M.

Usage:
    python dataset/download_cc3m.py --out D:\\datasets\\cc3m
    python dataset/download_cc3m.py --out D:\\datasets\\cc3m --splits validation
    python dataset/download_cc3m.py --out D:\\datasets\\cc3m --source urls --processes 8

Requirements:
    hf  : pip install huggingface_hub
    urls: pip install img2dataset
"""
import argparse
import os
import urllib.request

from tqdm import tqdm

HF_REPO = 'pixparse/cc3m-wds'
TSV_URLS = {
    'train': 'https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv',
    'validation': 'https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv',
}


def download_hf(out_dir, splits, workers):
    from huggingface_hub import snapshot_download

    # Shards are named cc3m-train-XXXX.tar / cc3m-validation-XXXX.tar
    patterns = [f'*{split}*' for split in splits] + ['*.json', '*.md']
    snapshot_download(
        repo_id=HF_REPO,
        repo_type='dataset',
        local_dir=out_dir,
        allow_patterns=patterns,
        max_workers=workers,
    )


def fetch_tsv(url, path):
    if os.path.exists(path):
        print(f'{path} already exists, skipping')
        return

    tmp_path = path + '.part'
    with urllib.request.urlopen(url) as response, open(tmp_path, 'wb') as f:
        total = int(response.headers.get('Content-Length', 0)) or None
        with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(path)) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp_path, path)


def add_header(src, dst):
    # The official TSVs have no header; img2dataset needs column names
    if os.path.exists(dst):
        return
    with open(src, 'r', encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        fout.write('caption\turl\n')
        for line in fin:
            fout.write(line)


def download_urls(out_dir, splits, processes, threads, image_size):
    from img2dataset import download

    meta_dir = os.path.join(out_dir, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)

    for split in splits:
        raw_path = os.path.join(meta_dir, f'{split}_raw.tsv')
        tsv_path = os.path.join(meta_dir, f'{split}.tsv')
        fetch_tsv(TSV_URLS[split], raw_path)
        add_header(raw_path, tsv_path)

        download(
            url_list=tsv_path,
            input_format='tsv',
            url_col='url',
            caption_col='caption',
            output_format='webdataset',
            output_folder=os.path.join(out_dir, split),
            processes_count=processes,
            thread_count=threads,
            image_size=image_size,
            resize_mode='keep_ratio',
            resize_only_if_bigger=True,
            skip_reencode=True,
            enable_wandb=False,
            incremental_mode='incremental',  # re-running resumes unfinished shards
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the CC3M dataset')
    parser.add_argument('--out', required=True, help='Destination folder')
    parser.add_argument('--source', choices=['hf', 'urls'], default='hf')
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--workers', type=int, default=8, help='[hf] parallel file downloads')
    parser.add_argument('--processes', type=int, default=4, help='[urls] img2dataset processes')
    parser.add_argument('--threads', type=int, default=32, help='[urls] img2dataset threads per process')
    parser.add_argument('--image-size', type=int, default=512, help='[urls] max image side after resizing')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.source == 'hf':
        download_hf(args.out, args.splits, args.workers)
    else:
        download_urls(args.out, args.splits, args.processes, args.threads, args.image_size)

    print(f'CC3M ({", ".join(args.splits)}) saved to {args.out}')
