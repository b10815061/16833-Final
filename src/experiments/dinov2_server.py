#!/usr/bin/env python3
"""
Self-contained DINOv2 ZMQ descriptor server for cslam experiments.

Implements the same ZMQ protocol as cslam/anyloc_server.py but without
the /opt/anyloc dependency — loads DINOv2 directly via torch.hub.

Feature extraction: hooks a DINOv2 transformer block to get patch tokens.
Aggregation: VLAD with sklearn KMeans codebook + optional PCA reduction.

ZMQ protocol (matches cslam/vpr/anyloc.py client):
    Request  frames: [shape_bytes (3×int32: H,W,C), img_bytes (H×W×C uint8)]
    Response frames: [status_byte (b'0' ok | b'1' error), payload]
                     payload = float32 descriptor on success, UTF-8 msg on error

Usage:
    python3 dinov2_server.py [--port 5555] [--device cpu] [--dim 128]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import zmq
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="[DINOv2Server %(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("dinov2_server")


# ── DINOv2 feature extractor ──────────────────────────────────────────────────

class DINOv2PatchExtractor:
    """Extracts patch tokens from a DINOv2 transformer block via a forward hook."""

    def __init__(self, model_name: str = "dinov2_vitb14", layer: int = 9,
                 device: str = "cpu"):
        log.info(f"Loading {model_name} (layer={layer}) on {device} …")
        self.device = device
        self.layer = layer
        self._features: torch.Tensor | None = None

        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name,
            verbose=False, trust_repo=True,
        )
        self.model.eval().to(device)

        # Register hook on the specified transformer block output
        block = self.model.blocks[layer]
        self._hook = block.register_forward_hook(self._hook_fn)
        log.info(f"{model_name} ready, patch_dim={self.model.embed_dim}")
        log.info(f"Running on device: {device.upper()} {'(CUDA available)' if torch.cuda.is_available() else '(CUDA not available)'}")

    def _hook_fn(self, module, input, output):
        # output: (batch, 1 + n_patches, d)  — first token is [CLS]
        self._features = output[:, 1:, :].detach()  # drop CLS, keep patches

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: (B, 3, H, W) preprocessed image tensor on self.device
        Returns:
            (B, n_patches, d) patch feature tensor on CPU
        """
        self._features = None
        self.model(batch.to(self.device))
        feats = self._features.cpu()   # (B, n_patches, d)
        # L2-normalise each patch descriptor
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    def embed_dim(self) -> int:
        return self.model.embed_dim

    def __del__(self):
        try:
            self._hook.remove()
        except Exception:
            pass


# ── VLAD aggregation ──────────────────────────────────────────────────────────

class VLADAggregator:
    """VLAD aggregation over patch descriptors using a KMeans codebook."""

    def __init__(self, num_clusters: int = 32, desc_dim: int = 768,
                 cache_path: str = "/tmp/vlad_codebook.pkl"):
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.cache_path = cache_path
        self.kmeans: MiniBatchKMeans | None = None
        self.vlad_dim = num_clusters * desc_dim

    def fit(self, patch_descs: np.ndarray) -> None:
        """Fit KMeans codebook from (N, d) patch descriptors."""
        log.info(f"Fitting VLAD KMeans (k={self.num_clusters}) on "
                 f"{patch_descs.shape[0]} vectors …")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters, random_state=42,
            batch_size=2048, max_iter=300,
        )
        self.kmeans.fit(patch_descs)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.kmeans, f)
        log.info(f"Codebook saved → {self.cache_path}")

    def load(self) -> bool:
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.kmeans = pickle.load(f)
            log.info(f"Codebook loaded ← {self.cache_path}")
            return True
        return False

    def generate(self, patch_descs: np.ndarray) -> np.ndarray:
        """
        Args:
            patch_descs: (n_patches, d) float32
        Returns:
            (num_clusters * d,) float32 VLAD vector, L2-normalised
        """
        assert self.kmeans is not None, "Call fit() or load() first"
        assignments = self.kmeans.predict(patch_descs)
        centers = self.kmeans.cluster_centers_
        vlad = np.zeros((self.num_clusters, self.desc_dim), dtype=np.float32)
        for k in range(self.num_clusters):
            mask = assignments == k
            if mask.any():
                residuals = patch_descs[mask] - centers[k]
                vlad[k] = residuals.sum(axis=0)
        # Intra-normalise per cluster, then global L2
        norms = np.linalg.norm(vlad, axis=1, keepdims=True) + 1e-8
        vlad = vlad / norms
        vlad = vlad.flatten()
        vlad = vlad / (np.linalg.norm(vlad) + 1e-8)
        return vlad.astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────

IMG_SIZE: int = 224  # overridden in main() to 518 when device=cuda
PREPROCESS: T.Compose  # set in main() after device is resolved


def _make_preprocess(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _collect_patch_descs(extractor: DINOv2PatchExtractor,
                         n_images: int = 300) -> np.ndarray:
    """Generate synthetic images and collect patch descriptors for VLAD fitting."""
    log.info(f"Collecting patch features from {n_images} synthetic images …")
    rng = np.random.default_rng(42)
    all_patches = []
    for _ in range(n_images):
        colour = rng.integers(0, 256, 3, dtype=np.uint8)
        tile = np.full((IMG_SIZE, IMG_SIZE, 3), colour, dtype=np.uint8)
        noise = rng.integers(-30, 30, tile.shape, dtype=np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(tile)
        batch = PREPROCESS(pil).unsqueeze(0)
        feats = extractor(batch)       # (1, n_patches, d)
        all_patches.append(feats[0].numpy())
    return np.concatenate(all_patches, axis=0)


def _bootstrap_pca(vlad: VLADAggregator, extractor: DINOv2PatchExtractor,
                   reduce_dim: int, cache_path: str,
                   n_images: int = 300) -> PCA | None:
    if reduce_dim <= 0:
        return None
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            pca = pickle.load(f)
        log.info(f"PCA loaded ← {cache_path} ({pca.n_components_} dims)")
        return pca
    log.info(f"Fitting PCA ({reduce_dim} dims) from synthetic VLAD vectors …")
    rng = np.random.default_rng(99)
    vlad_vecs = []
    for _ in range(n_images):
        colour = rng.integers(0, 256, 3, dtype=np.uint8)
        tile = np.full((IMG_SIZE, IMG_SIZE, 3), colour, dtype=np.uint8)
        noise = rng.integers(-25, 26, tile.shape, dtype=np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(tile)
        batch = PREPROCESS(pil).unsqueeze(0)
        feats = extractor(batch)[0].numpy()
        vlad_vecs.append(vlad.generate(feats))
    mat = np.stack(vlad_vecs)
    n_comp = min(reduce_dim, mat.shape[1], mat.shape[0] - 1)
    pca_model = PCA(n_components=n_comp, whiten=True)
    pca_model.fit(mat)
    with open(cache_path, "wb") as f:
        pickle.dump(pca_model, f)
    log.info(f"PCA fitted and cached → {cache_path} ({n_comp} dims)")
    return pca_model


# ── Descriptor extraction ─────────────────────────────────────────────────────

def extract_descriptor(img_np: np.ndarray,
                       extractor: DINOv2PatchExtractor,
                       vlad: VLADAggregator,
                       pca: PCA | None) -> np.ndarray:
    pil = Image.fromarray(img_np).convert("RGB")
    batch = PREPROCESS(pil).unsqueeze(0)
    patch_feats = extractor(batch)[0].numpy()  # (n_patches, d)
    desc = vlad.generate(patch_feats)
    if pca is not None:
        desc = pca.transform(desc[np.newaxis])[0].astype(np.float32)
    return desc


# ── ZMQ server ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 ZMQ descriptor server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ZMQ_PORT", 5555)))
    parser.add_argument("--device", default=os.environ.get("ANYLOC_DEVICE", "cpu"))
    parser.add_argument("--model", default=os.environ.get("ANYLOC_MODEL", "dinov2_vitb14"))
    parser.add_argument("--layer", type=int, default=int(os.environ.get("ANYLOC_LAYER", 9)))
    parser.add_argument("--clusters", type=int, default=int(os.environ.get("ANYLOC_NUM_CLUSTERS", 32)))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("ANYLOC_REDUCE_DIM", 128)),
                        help="PCA output dimension (0 = no PCA)")
    parser.add_argument("--cache-dir", default=os.environ.get("ANYLOC_CACHE_DIR", "/tmp/dinov2_cache"))
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Pick image size: 518 for CUDA (native DINOv2 patch grid), 224 for CPU
    global IMG_SIZE, PREPROCESS
    IMG_SIZE = 518 if args.device.startswith("cuda") else 224
    PREPROCESS = _make_preprocess(IMG_SIZE)
    log.info(f"IMG_SIZE={IMG_SIZE} (device={args.device})")

    # Include img_size in cache keys so CPU/CUDA caches don't collide
    vlad_cache = os.path.join(args.cache_dir,
                              f"vlad_k{args.clusters}_s{IMG_SIZE}.pkl")
    pca_cache  = os.path.join(args.cache_dir,
                              f"pca_d{args.dim}_s{IMG_SIZE}.pkl")

    extractor = DINOv2PatchExtractor(args.model, args.layer, args.device)
    desc_dim  = extractor.embed_dim()

    vlad = VLADAggregator(args.clusters, desc_dim, vlad_cache)
    if not vlad.load():
        patches = _collect_patch_descs(extractor)
        vlad.fit(patches)

    pca = _bootstrap_pca(vlad, extractor, args.dim, pca_cache)

    # Determine final descriptor dimension
    test_patches = np.random.randn(10, desc_dim).astype(np.float32)
    test_patches /= np.linalg.norm(test_patches, axis=1, keepdims=True)
    test_vlad = vlad.generate(test_patches)
    if pca is not None:
        final_dim = pca.transform(test_vlad[np.newaxis])[0].shape[0]
    else:
        final_dim = len(test_vlad)
    log.info(f"Final descriptor dimension: {final_dim}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    log.info(f"Listening on tcp://*:{args.port}")

    req_count = 0
    while True:
        frames = sock.recv_multipart()
        if len(frames) != 2:
            sock.send_multipart([b"1", f"Expected 2 frames, got {len(frames)}".encode()])
            continue

        shape_bytes, img_bytes = frames
        if len(shape_bytes) != 12:
            sock.send_multipart([b"1", b"shape_bytes must be 12 bytes"])
            continue

        h, w, c = np.frombuffer(shape_bytes, dtype=np.int32)
        if c != 3 or h <= 0 or w <= 0 or len(img_bytes) != int(h * w * c):
            sock.send_multipart([b"1", f"Bad image dims h={h} w={w} c={c}".encode()])
            continue

        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c).copy()

        t0 = time.perf_counter()
        try:
            desc = extract_descriptor(img_np, extractor, vlad, pca)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            req_count += 1
            if req_count % 20 == 0:
                log.info(f"Processed {req_count} requests; last {elapsed_ms:.1f} ms")
            sock.send_multipart([b"0", desc.tobytes()])
        except Exception as exc:
            log.error(f"Error: {exc}", exc_info=True)
            sock.send_multipart([b"1", str(exc).encode()])


if __name__ == "__main__":
    main()
