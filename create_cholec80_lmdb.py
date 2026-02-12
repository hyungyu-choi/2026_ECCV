"""
Convert Cholec80 training frames to LMDB for PL-Stitch pretraining.

Expected input structure:
    <frames_root>/
        01/
            frame_000000.jpg   (or any .jpg/.png naming)
            frame_000001.jpg
            ...
        02/
            ...
        ...
        40/
            ...

LMDB key format:  video{NN}_{frame_index}   (e.g. video01_0, video01_1, ...)
LMDB value:       JPEG-encoded bytes (so cv2.imdecode works in the data loader)

Usage:
    python create_cholec80_lmdb.py \
        --frames-root /path/to/cholec80/frames/extract_1fps/trainin_set \
        --lmdb-path   /path/to/output/cholec80_pretrain.lmdb \
        --image-size 224 \
        --map-size 5e11
"""

import os
import argparse
import glob
import cv2
import lmdb
import numpy as np
from tqdm import tqdm
from natsort import natsorted  # pip install natsort; fallback to sorted if not available


def natural_sort(file_list):
    """Sort filenames naturally (e.g. frame_2 before frame_10)."""
    try:
        from natsort import natsorted
        return natsorted(file_list)
    except ImportError:
        return sorted(file_list)


def create_cholec80_lmdb(frames_root, lmdb_path, image_size=224, map_size=5e11):
    """
    Reads video folders (01..40) under frames_root, encodes each frame as
    JPEG, and writes it into an LMDB with key = 'video{NN}_{idx}'.
    """
    # Discover video folders
    video_dirs = sorted([
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ])

    if not video_dirs:
        raise RuntimeError(f"No sub-folders found under {frames_root}")

    print(f"Found {len(video_dirs)} video folders: {video_dirs[:5]} ... {video_dirs[-1]}")

    env = lmdb.open(lmdb_path, map_size=int(map_size))
    total_frames = 0

    with env.begin(write=True) as txn:
        for vdir in video_dirs:
            video_path = os.path.join(frames_root, vdir)

            # Collect all image files
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            frame_files = []
            for ext in exts:
                frame_files.extend(glob.glob(os.path.join(video_path, ext)))

            frame_files = natural_sort(frame_files)

            if len(frame_files) == 0:
                print(f"  [WARN] No images in {video_path}, skipping.")
                continue

            # Video name used in the key (e.g. "video01")
            video_name = f"video{vdir}"

            for idx, fpath in enumerate(tqdm(
                frame_files,
                desc=f"  {video_name} ({len(frame_files)} frames)",
                leave=False,
            )):
                img = cv2.imread(fpath)
                if img is None:
                    print(f"  [WARN] Cannot read {fpath}, skipping.")
                    continue

                # Resize
                img = cv2.resize(img, (image_size, image_size))

                # Encode as JPEG bytes (this is what cv2.imdecode expects later)
                success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    print(f"  [WARN] Failed to encode {fpath}, skipping.")
                    continue

                key = f"{video_name}_{idx}".encode('utf-8')
                txn.put(key, buf.tobytes())
                total_frames += 1

    env.close()
    print(f"\nDone! Wrote {total_frames} frames to {lmdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Cholec80 training frames to LMDB for PL-Stitch."
    )
    parser.add_argument(
        '--frames-root', type=str, required=True,
        help="Root folder containing video sub-folders (01, 02, ..., 40)."
    )
    parser.add_argument(
        '--lmdb-path', type=str, required=True,
        help="Output path for the LMDB database."
    )
    parser.add_argument(
        '--image-size', type=int, default=224,
        help="Resize frames to this square size (default: 224)."
    )
    parser.add_argument(
        '--map-size', type=float, default=5e11,
        help="LMDB max map size in bytes (default: 500GB)."
    )
    args = parser.parse_args()

    create_cholec80_lmdb(
        frames_root=args.frames_root,
        lmdb_path=args.lmdb_path,
        image_size=args.image_size,
        map_size=args.map_size,
    )