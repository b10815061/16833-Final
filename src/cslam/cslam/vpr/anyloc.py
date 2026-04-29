"""AnyLoc ZMQ client — drop-in VPR backend for cslam.

Instead of running DINOv2 locally on each resource-constrained robot, this
backend forwards the raw keyframe image to a centralized AnyLoc server over
ZMQ.  The server runs DINOv2 + VLAD + PCA and returns a compact float32
descriptor.  This implements Option B: the robot sends raw images and the
backend handles all heavy inference.

ZMQ protocol (must match anyloc_server.py):
    Request  frames: [shape_bytes (3×int32 H,W,C), img_bytes (H×W×C uint8)]
    Response frames: [status (b'0' ok | b'1' error), payload]
                     payload = float32 descriptor on success, UTF-8 msg on error
"""

import numpy as np
import zmq


class AnyLocVPR:
    """VPR backend that offloads DINOv2 inference to a remote AnyLoc server.

    Robots stay lightweight: the only local work is encoding the image into
    bytes and decoding the returned descriptor.  All DINOv2, VLAD, and PCA
    computation happens on the backend server.
    """

    def __init__(self, params, node):
        self.params = params
        self.node = node

        server_addr = self.params.get('frontend.anyloc.server_address',
                                      'tcp://localhost:5555')
        timeout_ms = int(self.params.get('frontend.anyloc.timeout_ms', 5000))

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(server_addr)

        self.node.get_logger().info(
            f'AnyLocVPR: connecting to {server_addr} (timeout={timeout_ms} ms)')

        # Send a tiny probe image to confirm the server is reachable and to
        # learn the descriptor dimension without hard-coding it here.
        self._descriptor_dim = self._probe_server()
        self.node.get_logger().info(
            f'AnyLocVPR: server ready, descriptor_dim={self._descriptor_dim}')

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _probe_server(self) -> int:
        """Send a synthetic 16×16 image; return the descriptor length."""
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            desc = self._send_image(dummy)
            return desc.shape[0]
        except Exception as exc:
            self.node.get_logger().warn(
                f'AnyLocVPR: probe failed ({exc}). Defaulting descriptor_dim=128.')
            return 128

    def _send_image(self, img_np: np.ndarray) -> np.ndarray:
        """Send an H×W×3 uint8 image and return the server's float32 descriptor."""
        h, w, c = img_np.shape
        shape_bytes = np.array([h, w, c], dtype=np.int32).tobytes()
        img_bytes = np.ascontiguousarray(img_np).tobytes()
        self._socket.send_multipart([shape_bytes, img_bytes])

        frames = self._socket.recv_multipart()
        status, payload = frames[0], frames[1]
        if status == b'0':
            return np.frombuffer(payload, dtype=np.float32).copy()
        raise RuntimeError(f'AnyLoc server error: {payload.decode()}')

    # ── Public interface (mirrors cosplace.py / netvlad.py) ──────────────────

    def compute_embedding(self, keyframe: np.ndarray) -> np.ndarray:
        """Forward a keyframe to the AnyLoc server and return its descriptor.

        Args:
            keyframe: H×W×C uint8 numpy array from cv_bridge (BGR or RGB
                      depending on the camera driver — the server treats
                      pixel values as-is, consistent with how other cslam
                      VPR backends handle the image).

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
                'AnyLocVPR: ZMQ timeout — server may be overloaded. '
                'Returning zero descriptor for this keyframe.')
            return np.zeros(self._descriptor_dim, dtype=np.float32)
        except Exception as exc:
            self.node.get_logger().error(
                f'AnyLocVPR: compute_embedding failed: {exc}')
            return np.zeros(self._descriptor_dim, dtype=np.float32)
