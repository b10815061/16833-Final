#!/usr/bin/env python3
"""
CosPlace descriptor server for cslam — ZMQ REP socket.

Loads a ResNet18 CosPlace model once and serves compact float32 descriptors
to any number of robot agents over a plain ZMQ socket.  This is the server-
side counterpart to cslam/vpr/cosplace_server_client.py.

Running inference in a dedicated process (rather than inside each robot's LCD
node) eliminates the CPU contention that prevents RTAB-Map odometry from
recovering on difficult sequences during multi-agent experiments.

Usage:
    python cosplace_server.py [--checkpoint PATH] [--backbone resnet18]
                               [--dim 64] [--crop-size 376]
                               [--port 5556] [--device auto]

Environment overrides (higher priority than CLI flags):
    COSPLACE_PORT      TCP port to bind (default 5556)
    COSPLACE_DEVICE    cuda | cpu | auto (default auto)

ZMQ protocol (identical to anyloc_server.py):
    Request  frames: [shape_bytes (3×int32 H,W,C), img_bytes (H×W×C uint8)]
    Response frames: [b'0', float32_descriptor_bytes]   # success
                     [b'1', error_utf8_bytes]            # failure
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import zmq
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[CosPlace %(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("cosplace_server")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)


# ── Device helpers ─────────────────────────────────────────────────────────────

def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(checkpoint: str, backbone: str, dim: int, device: str):
    try:
        from cslam.vpr.cosplace_utils.network import GeoLocalizationNet
    except ImportError as exc:
        raise SystemExit(
            "Cannot import cslam package.  Source the workspace before running:\n"
            "  source /home/ros/ws/install/setup.bash\n"
            f"  (original error: {exc})"
        ) from exc

    log.info(f"Loading {backbone}  dim={dim}  device={device}")
    log.info(f"Checkpoint: {checkpoint}")

    model = GeoLocalizationNet(backbone, dim, node=None)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    log.info("Model ready.")
    return model


def _make_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def _find_default_checkpoint() -> str:
    """Try to locate the default checkpoint via ament_index (requires sourced ws)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory("cslam")
        return os.path.join(pkg, "models", "resnet18_64_imagenet.pth")
    except Exception:
        return ""


# ── ZMQ serve loop ─────────────────────────────────────────────────────────────

def serve(model, transform, device: str, port: int) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    log.info(f"Bound to tcp://*:{port}  —  waiting for requests")

    while True:
        try:
            frames = socket.recv_multipart()
            shape_bytes, img_bytes = frames[0], frames[1]

            h, w, c = np.frombuffer(shape_bytes, dtype=np.int32)
            img_np  = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c)

            pil    = Image.fromarray(img_np)
            tensor = transform(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                desc = model(tensor)

            desc_np = desc[0].cpu().numpy().astype(np.float32)
            socket.send_multipart([b'0', desc_np.tobytes()])

        except Exception as exc:
            log.warning(f"Request failed: {exc}")
            try:
                socket.send_multipart([b'1', str(exc).encode()])
            except Exception:
                pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CosPlace ZMQ descriptor server for cslam multi-robot SLAM")
    parser.add_argument(
        "--checkpoint", default="",
        help="Path to .pth checkpoint file (default: auto-locate via ament_index)")
    parser.add_argument(
        "--backbone", default="resnet18",
        choices=["resnet18", "resnet50", "resnet101", "resnet152", "vgg16"],
        help="Backbone architecture (must match checkpoint)")
    parser.add_argument(
        "--dim", type=int, default=64,
        help="Descriptor output dimension (must match checkpoint, default 64)")
    parser.add_argument(
        "--crop-size", type=int, default=376,
        help="Centre-crop size before Resize(224) (default 376, matches tum_baseline.yaml)")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("COSPLACE_PORT", "5556")),
        help="TCP port to bind (default 5556; env COSPLACE_PORT overrides)")
    parser.add_argument(
        "--device",
        default=os.environ.get("COSPLACE_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default auto: CUDA if available, else CPU; "
             "env COSPLACE_DEVICE overrides)")
    args = parser.parse_args()

    device     = _resolve_device(args.device)
    checkpoint = args.checkpoint or _find_default_checkpoint()

    if not checkpoint or not os.path.isfile(checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {checkpoint!r}\n"
            "Pass --checkpoint /path/to/model.pth"
        )

    model     = _load_model(checkpoint, args.backbone, args.dim, device)
    transform = _make_transform(args.crop_size)
    serve(model, transform, device, args.port)


if __name__ == "__main__":
    main()
