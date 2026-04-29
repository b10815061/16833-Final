#!/usr/bin/env python3
"""
AnyLoc descriptor server for cslam — ZMQ REP socket.

Loads DINOv2, aggregates patch features with VLAD, optionally reduces
dimensionality with PCA, and serves compact float32 descriptors to any
number of robot agents over a plain ZMQ socket.

This is the *backend* counterpart to cslam/vpr/anyloc.py (the robot-side
client).  Run this once on the backend machine; all robots connect to it.

Usage:
    python anyloc_server.py [--config /path/to/anyloc.yaml]

Environment variable overrides (highest priority):
    ZMQ_PORT            TCP port to bind (default 5555)
    ANYLOC_MODEL        DINOv2 model name, e.g. dinov2_vitb14 (default)
    ANYLOC_LAYER        Transformer layer to extract features from (default 9)
    ANYLOC_FACET        Feature type: key | query | value | token (default value)
    ANYLOC_DEVICE       cuda | cpu (default cpu)
    ANYLOC_NUM_CLUSTERS VLAD codebook size (default 32)
    ANYLOC_REDUCE_DIM   PCA output dimension; 0 = no PCA (default 128)
    ANYLOC_CACHE_DIR    Directory for VLAD/PCA caches (default /tmp/anyloc_cache)
    ANYLOC_CONFIG       Path to YAML config file

ZMQ protocol:
    Request  frames: [shape_bytes (3×int32: H,W,C), img_bytes (H×W×C uint8)]
    Response frames: [status_byte (b'0' ok | b'1' error), payload]
                     payload = float32 descriptor on success, UTF-8 msg on error
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import yaml
import zmq
from PIL import Image

# AnyLoc demo utilities (DinoV2ExtractFeatures, VLAD)
sys.path.insert(0, "/opt/anyloc/demo")
try:
    from utilities import DinoV2ExtractFeatures, VLAD
except ImportError as exc:
    raise SystemExit(
        "Cannot import AnyLoc utilities from /opt/anyloc/demo. "
        "Make sure the AnyLoc demo is installed at that path."
    ) from exc

try:
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="[AnyLoc %(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("anyloc_server")


# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _cfg(env_key: str, yaml_val, default):
    """Priority: env var → yaml value → built-in default."""
    val = os.environ.get(env_key)
    if val is not None:
        return val
    if yaml_val is not None:
        return yaml_val
    return default


def _load_config(config_path: str) -> dict:
    raw = _load_yaml(config_path)
    cfg = {}
    cfg["model"]        = str(_cfg("ANYLOC_MODEL",        raw.get("model", {}).get("backbone"),         "dinov2_vitb14"))
    cfg["layer"]        = int(_cfg("ANYLOC_LAYER",        raw.get("model", {}).get("layer"),             9))
    cfg["facet"]        = str(_cfg("ANYLOC_FACET",        raw.get("model", {}).get("facet"),             "value"))
    cfg["device"]       = str(_cfg("ANYLOC_DEVICE",       raw.get("server", {}).get("device"),           "cpu"))
    cfg["num_clusters"] = int(_cfg("ANYLOC_NUM_CLUSTERS", raw.get("descriptor", {}).get("num_clusters"), 32))
    cfg["reduce_dim"]   = int(_cfg("ANYLOC_REDUCE_DIM",   raw.get("descriptor", {}).get("reduce_dim"),   128))
    cfg["use_vlad"]     = str(_cfg("ANYLOC_USE_VLAD",     raw.get("descriptor", {}).get("use_vlad"),     True)).lower() not in ("0", "false", "no")
    cfg["zmq_port"]     = int(_cfg("ZMQ_PORT",            raw.get("server", {}).get("zmq_port"),         5555))
    cfg["cache_dir"]    = str(_cfg("ANYLOC_CACHE_DIR",    raw.get("server", {}).get("cache_dir"),        "/tmp/anyloc_cache"))
    return cfg


# ── Pre-processing ─────────────────────────────────────────────────────────────

def _make_preprocess(device: str, img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_extractor(cfg: dict) -> DinoV2ExtractFeatures:
    assert cfg["facet"] in ("key", "query", "value", "token"), \
        f"Unknown facet: {cfg['facet']}"
    log.info(f"Loading {cfg['model']} layer={cfg['layer']} "
             f"facet={cfg['facet']} device={cfg['device']} …")
    extractor = DinoV2ExtractFeatures(
        dino_model=cfg["model"],
        layer=cfg["layer"],
        facet=cfg["facet"],
        use_cls=False,
        norm_descs=True,
        device=cfg["device"],
    )
    log.info("DINOv2 model loaded.")
    return extractor


# ── VLAD vocabulary ────────────────────────────────────────────────────────────

def _synthetic_patch_features(
    extractor: DinoV2ExtractFeatures,
    preprocess: T.Compose,
    device: str,
    img_size: int,
    n_images: int = 200,
) -> torch.Tensor:
    """Bootstrap VLAD vocabulary from random solid-colour synthetic images."""
    log.info(f"Building VLAD vocabulary from {n_images} synthetic images …")
    rng = np.random.default_rng(42)
    all_descs = []
    for _ in range(n_images):
        colour = rng.integers(0, 256, 3, dtype=np.uint8)
        tile = np.full((img_size, img_size, 3), colour, dtype=np.uint8)
        noise = rng.integers(-30, 30, tile.shape, dtype=np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(tile)
        batch = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = extractor(batch)
        all_descs.append(feats.squeeze(0).cpu())
    return torch.cat(all_descs, dim=0)


def _load_vlad(
    extractor: DinoV2ExtractFeatures,
    preprocess: T.Compose,
    cfg: dict,
) -> VLAD:
    vlad = VLAD(
        num_clusters=cfg["num_clusters"],
        desc_dim=None,
        intra_norm=True,
        norm_descs=True,
        dist_mode="cosine",
        vlad_mode="hard",
        cache_dir=cfg["cache_dir"],
    )
    if vlad.can_use_cache_vlad():
        log.info(f"Loading VLAD vocabulary from cache: {cfg['cache_dir']}")
        vlad.fit(None)
    else:
        img_size = 224 if cfg["device"] == "cpu" else 518
        train_descs = _synthetic_patch_features(
            extractor, preprocess, cfg["device"], img_size)
        vlad.fit(train_descs)
        log.info(f"VLAD vocabulary cached to {cfg['cache_dir']}")
    return vlad


# ── PCA ────────────────────────────────────────────────────────────────────────

def _pca_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "pca.pkl")


def _load_pca(cache_dir: str) -> Optional[SklearnPCA]:
    path = _pca_cache_path(cache_dir)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_pca(pca_obj: SklearnPCA, cache_dir: str) -> None:
    with open(_pca_cache_path(cache_dir), "wb") as f:
        pickle.dump(pca_obj, f)


def _fit_pca(
    extractor: DinoV2ExtractFeatures,
    vlad: VLAD,
    preprocess: T.Compose,
    cfg: dict,
) -> Optional[SklearnPCA]:
    """Fit PCA on synthetic VLAD vectors; cache result."""
    if not _HAS_SKLEARN or cfg["reduce_dim"] <= 0:
        return None

    cached = _load_pca(cfg["cache_dir"])
    if cached is not None:
        log.info(f"PCA loaded from cache → {cached.n_components_} dims")
        return cached

    log.info("Fitting PCA from synthetic VLAD vectors …")
    img_size = 224 if cfg["device"] == "cpu" else 518
    rng = np.random.default_rng(99)
    vlad_vecs = []
    for _ in range(200):
        colour = rng.integers(0, 256, 3, dtype=np.uint8)
        tile = np.full((img_size, img_size, 3), colour, dtype=np.uint8)
        noise = rng.integers(-25, 26, tile.shape, dtype=np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(tile)
        batch = preprocess(pil).unsqueeze(0).to(cfg["device"])
        with torch.no_grad():
            feats = extractor(batch)
        patch_descs = feats.squeeze(0).cpu()
        vlad_vec = vlad.generate(patch_descs)
        vlad_vecs.append(vlad_vec.numpy())

    mat = np.stack(vlad_vecs, axis=0)
    n_comp = min(cfg["reduce_dim"], mat.shape[1], mat.shape[0] - 1)
    pca = SklearnPCA(n_components=n_comp, whiten=True)
    pca.fit(mat)
    _save_pca(pca, cfg["cache_dir"])
    log.info(f"PCA fitted and cached: {mat.shape} → {n_comp} dims")
    return pca


# ── Descriptor extraction ──────────────────────────────────────────────────────

def extract_descriptor(
    img_np: np.ndarray,
    extractor: DinoV2ExtractFeatures,
    preprocess: T.Compose,
    vlad: Optional[VLAD],
    pca: Optional[SklearnPCA],
    cfg: dict,
) -> np.ndarray:
    """
    img_np: H×W×3 uint8
    Returns: 1-D float32 descriptor.
    """
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 image, got shape {img_np.shape}")

    pil = Image.fromarray(img_np).convert("RGB")
    batch = preprocess(pil).unsqueeze(0).to(cfg["device"])

    with torch.no_grad():
        feats = extractor(batch)    # 1 × n_patches × d

    patch_descs = feats.squeeze(0).cpu()  # n_patches × d

    if cfg["use_vlad"] and vlad is not None:
        vlad_vec = vlad.generate(patch_descs)
        desc = vlad_vec.numpy().astype(np.float32)
        if pca is not None:
            desc = pca.transform(desc[np.newaxis])[0].astype(np.float32)
    else:
        desc = patch_descs.mean(dim=0).numpy().astype(np.float32)

    return desc


# ── ZMQ helpers ────────────────────────────────────────────────────────────────

def _send_ok(sock: zmq.Socket, payload: bytes) -> None:
    sock.send_multipart([b"0", payload])


def _send_err(sock: zmq.Socket, message: str) -> None:
    sock.send_multipart([b"1", message.encode()])


# ── Main server loop ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AnyLoc ZMQ descriptor server")
    parser.add_argument("--config", default=os.environ.get(
        "ANYLOC_CONFIG", "/workspace/config/anyloc.yaml"),
        help="Path to anyloc YAML config file")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    os.makedirs(cfg["cache_dir"], exist_ok=True)

    log.info(f"Config: {cfg}")

    # DINOv2 expects image sizes that are multiples of patch size 14.
    # 518 = 37×14 (full-res). 224 = 16×14 (CPU-friendly).
    img_size = 224 if cfg["device"] == "cpu" else 518
    preprocess = _make_preprocess(cfg["device"], img_size)

    extractor = _load_extractor(cfg)

    vlad: Optional[VLAD] = None
    pca: Optional[SklearnPCA] = None

    if cfg["use_vlad"]:
        vlad = _load_vlad(extractor, preprocess, cfg)
        if cfg["reduce_dim"] > 0:
            pca = _fit_pca(extractor, vlad, preprocess, cfg)
    else:
        log.info("VLAD disabled — returning mean-pooled patch descriptors.")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{cfg['zmq_port']}")
    log.info(f"Listening on tcp://*:{cfg['zmq_port']}")

    req_count = 0
    while True:
        frames = sock.recv_multipart()
        if len(frames) != 2:
            _send_err(sock, f"Expected 2 frames [shape, image], got {len(frames)}")
            continue

        shape_bytes, img_bytes = frames

        if len(shape_bytes) != 12:  # 3 × int32
            _send_err(sock, f"shape_bytes must be 12 bytes, got {len(shape_bytes)}")
            continue

        h, w, c = np.frombuffer(shape_bytes, dtype=np.int32)

        if c != 3:
            _send_err(sock, f"Expected 3-channel image, got c={c}")
            continue
        if h <= 0 or w <= 0 or h > 4096 or w > 4096:
            _send_err(sock, f"Implausible image dimensions h={h} w={w}")
            continue
        if len(img_bytes) != int(h) * int(w) * int(c):
            _send_err(sock, f"Buffer length {len(img_bytes)} != {h}×{w}×{c}={h*w*c}")
            continue

        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c).copy()

        t0 = time.perf_counter()
        try:
            desc = extract_descriptor(img_np, extractor, preprocess, vlad, pca, cfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            req_count += 1
            if req_count % 10 == 0:
                log.info(f"Processed {req_count} requests; "
                         f"last {elapsed_ms:.1f} ms, desc_dim={len(desc)}")
            _send_ok(sock, desc.tobytes())
        except Exception as exc:
            log.error(f"Error processing request: {exc}", exc_info=True)
            _send_err(sock, str(exc))


if __name__ == "__main__":
    main()
