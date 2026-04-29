#!/usr/bin/env python3
"""
bootstrap_vlad.py — Rebuild the DINOv2 VLAD vocabulary from real images.

The default dinov2_server.py bootstraps the VLAD codebook from synthetic
random-color tiles. This script replaces that with real EuRoC frames, giving
KMeans clusters that reflect actual visual variation in the deployment domain.

Usage:
    python3 experiments/bootstrap_vlad.py \
        --img-dirs /home/ros/datasets/EuRoC/MH_01_easy/mav0/cam0/data \
                   /tmp/euroc/MH_02_easy/mav0/cam0/data \
        --n-images 500 --clusters 32 --dim 128 \
        --cache-dir /tmp/dinov2_cache
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

# Import from dinov2_server (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dinov2_server import DINOv2PatchExtractor, VLADAggregator, PREPROCESS
from PIL import Image


def collect_image_paths(img_dirs: list[str]) -> list[str]:
    paths = []
    for d in img_dirs:
        for f in os.listdir(d):
            if f.endswith('.png'):
                paths.append(os.path.join(d, f))
    return paths


def extract_patches(extractor: DINOv2PatchExtractor, img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')
    batch = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        feats = extractor(batch)[0].numpy()   # (n_patches, d)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dirs', nargs='+', required=True,
                        help='Directories containing .png image files (EuRoC cam0/data/)')
    parser.add_argument('--n-images', type=int, default=500,
                        help='Number of images to sample for vocabulary building')
    parser.add_argument('--clusters', type=int, default=32,
                        help='Number of VLAD clusters (k)')
    parser.add_argument('--dim', type=int, default=128,
                        help='PCA output dimension (0 = skip PCA)')
    parser.add_argument('--cache-dir', default='/tmp/dinov2_cache',
                        help='Directory to write vlad_k<k>.pkl and pca_d<dim>.pkl')
    parser.add_argument('--model', default='dinov2_vitb14')
    parser.add_argument('--layer', type=int, default=9)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    vlad_path = os.path.join(args.cache_dir, f'vlad_k{args.clusters}.pkl')
    pca_path  = os.path.join(args.cache_dir, f'pca_d{args.dim}.pkl')

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Collect and sample image paths
    all_paths = collect_image_paths(args.img_dirs)
    if not all_paths:
        print(f"ERROR: no .png files found in {args.img_dirs}")
        sys.exit(1)
    n = min(args.n_images, len(all_paths))
    sampled = random.sample(all_paths, n)
    print(f"Sampled {n}/{len(all_paths)} images from {len(args.img_dirs)} dir(s)")

    # Load DINOv2
    extractor = DINOv2PatchExtractor(model_name=args.model,
                                     layer=args.layer, device=args.device)
    desc_dim = extractor.embed_dim()

    # ── Step 1: collect patch descriptors ─────────────────────────────────────
    print(f"Extracting patch descriptors from {n} images …")
    all_patches = []
    for i, path in enumerate(sampled):
        patches = extract_patches(extractor, path)
        all_patches.append(patches)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")
    all_patches_np = np.concatenate(all_patches, axis=0)
    print(f"Total patches: {all_patches_np.shape[0]}  dim={all_patches_np.shape[1]}")

    # ── Step 2: fit VLAD codebook ─────────────────────────────────────────────
    vlad = VLADAggregator(num_clusters=args.clusters, desc_dim=desc_dim,
                          cache_path=vlad_path)
    vlad.fit(all_patches_np)   # saves vlad_k<k>.pkl

    # ── Step 3: fit PCA on VLAD vectors ───────────────────────────────────────
    if args.dim > 0:
        print(f"Building {n} VLAD vectors for PCA fitting …")
        vlad_vecs = []
        for patches in all_patches:
            vlad_vecs.append(vlad.generate(patches))
        mat = np.stack(vlad_vecs)
        print(f"VLAD matrix: {mat.shape}")
        n_comp = min(args.dim, mat.shape[1], mat.shape[0] - 1)
        print(f"Fitting PCA ({n_comp} components, whiten=True) …")
        pca_model = PCA(n_components=n_comp, whiten=True)
        pca_model.fit(mat)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_model, f)
        size_mb = os.path.getsize(pca_path) / 1e6
        print(f"PCA saved → {pca_path}  ({size_mb:.1f} MB)")

    print(f"\nDone. Vocabulary files in {args.cache_dir}/")
    print(f"  {vlad_path}")
    if args.dim > 0:
        print(f"  {pca_path}")


if __name__ == '__main__':
    main()
