"""CosPlace ZMQ client — drop-in VPR backend for cslam.

Instead of running ResNet18 locally inside the LCD node process (which
competes with RTAB-Map odometry for CPU time), this backend forwards raw
keyframe images to a standalone cosplace_server.py over ZMQ and receives
back the float32 descriptor.

The wire protocol is identical to anyloc.py / anyloc_server.py, so the
same server-probe and error-handling logic applies.

Start the server before launching the experiment:
    python cslam/cslam/cosplace_server.py --port 5556

Select this backend in the config YAML:
    frontend:
      global_descriptor_technique: "cosplace_server"
      cosplace_server:
        server_address: "tcp://localhost:5556"
        timeout_ms: 5000
"""

import numpy as np
import zmq


class CosPlaceServerClient:
    """VPR backend that offloads CosPlace inference to a remote server.

    Robots stay lightweight: the only local work is encoding the image into
    bytes and decoding the returned descriptor.  All ResNet18 computation
    happens on the backend server (in a separate process, optionally on GPU).
    """

    def __init__(self, params, node):
        self.params = params
        self.node = node

        server_addr = self.params.get(
            'frontend.cosplace_server.server_address', 'tcp://localhost:5556')
        timeout_ms = int(self.params.get(
            'frontend.cosplace_server.timeout_ms', 5000))

        self._context = zmq.Context()
        self._socket  = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(server_addr)

        self.node.get_logger().info(
            f'CosPlaceServerClient: connecting to {server_addr} '
            f'(timeout={timeout_ms} ms)')

        self._descriptor_dim = self._probe_server()
        self.node.get_logger().info(
            f'CosPlaceServerClient: server ready, '
            f'descriptor_dim={self._descriptor_dim}')

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _probe_server(self) -> int:
        """Send a synthetic 16×16 image; return the descriptor length."""
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            desc = self._send_image(dummy)
            return desc.shape[0]
        except Exception as exc:
            self.node.get_logger().warn(
                f'CosPlaceServerClient: probe failed ({exc}). '
                'Defaulting descriptor_dim=64.')
            return 64

    def _send_image(self, img_np: np.ndarray) -> np.ndarray:
        """Send an H×W×3 uint8 image and return the server's float32 descriptor."""
        h, w, c    = img_np.shape
        shape_bytes = np.array([h, w, c], dtype=np.int32).tobytes()
        img_bytes   = np.ascontiguousarray(img_np).tobytes()
        self._socket.send_multipart([shape_bytes, img_bytes])

        frames = self._socket.recv_multipart()
        status, payload = frames[0], frames[1]
        if status == b'0':
            return np.frombuffer(payload, dtype=np.float32).copy()
        raise RuntimeError(f'CosPlace server error: {payload.decode()}')

    # ── Public interface (mirrors cosplace.py / anyloc.py) ───────────────────

    def compute_embedding(self, keyframe: np.ndarray) -> np.ndarray:
        """Forward a keyframe to the CosPlace server and return its descriptor.

        Args:
            keyframe: H×W×C uint8 numpy array (BGR or RGB, consistent with
                      how the camera driver delivers images — the server
                      applies the same preprocessing as cosplace.py).

        Returns:
            1-D float32 descriptor vector of length ``self._descriptor_dim``.
            Returns a zero vector on network error so the SLAM system can
            continue running; the missed descriptor simply won't produce a
            loop-closure candidate.
        """
        try:
            return self._send_image(keyframe)
        except zmq.error.Again:
            self.node.get_logger().warn(
                'CosPlaceServerClient: ZMQ timeout — server may be overloaded. '
                'Returning zero descriptor for this keyframe.')
            return np.zeros(self._descriptor_dim, dtype=np.float32)
        except Exception as exc:
            self.node.get_logger().error(
                f'CosPlaceServerClient: compute_embedding failed: {exc}')
            return np.zeros(self._descriptor_dim, dtype=np.float32)
