"""SAM3 text-segmentation client for the tutorial's length-prefixed TCP protocol."""

from __future__ import annotations

import io
import json
import os
import socket
import struct
import numpy as np
from PIL import Image


SAM3_HOST = os.environ.get("SAM3_HOST", "127.0.0.1")
SAM3_PORT = int(os.environ.get("SAM3_PORT", "28317"))


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(struct.pack("<I", len(payload)) + payload)


def _receive_frame(connection: socket.socket) -> bytes:
    header = bytearray()
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("SAM3 closed the connection while sending a frame header")
        header.extend(chunk)
    size = struct.unpack("<I", header)[0]
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("SAM3 closed the connection while sending a frame")
        payload.extend(chunk)
    return bytes(payload)


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("SAM3 input must be an HxWx3 uint8 RGB image")
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


def call_sam3_text(host: str, port: int, image: bytes, texts: list[str],
                   confidence: float = 0.25, timeout: float = 60.0):
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("SAM3 requires at least one non-empty text prompt")
    request = {"type": "infer", "model": "sam3", "mode": "text",
               "image_bytes": len(image), "text": texts, "conf": confidence}
    arrays = {}
    done = None
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        _send_frame(connection, json.dumps(request).encode("utf-8"))
        _send_frame(connection, image)
        while True:
            message = json.loads(_receive_frame(connection).decode("utf-8"))
            if message["type"] == "array":
                raw = _receive_frame(connection)
                dtype = np.dtype(message["dtype"])
                expected = int(np.prod(message["shape"]))
                values = np.frombuffer(raw, dtype=dtype)
                if values.size != expected:
                    raise ValueError(f"SAM3 array {message['name']} has the wrong size")
                arrays[message["name"]] = values.reshape(message["shape"]).copy()
            elif message["type"] == "done":
                done = message
                break
            elif message["type"] == "error":
                raise RuntimeError(message.get("message", "SAM3 inference failed"))
            else:
                continue
    if done is None:
        raise RuntimeError("SAM3 returned no done message")
    for name in ("masks", "boxes", "scores"):
        if name not in arrays:
            raise ValueError(f"SAM3 response omitted {name}")
    return arrays, done


class SAM3TextSegmenter:
    """Produce one best boolean mask per requested object description."""

    def __init__(self, texts: list[str] | None = None, host: str = SAM3_HOST,
                 port: int = SAM3_PORT, confidence: float = 0.25,
                 timeout: float = 60.0, jpeg_quality: int = 95):
        self.texts = list(texts or [])
        self.host = host
        self.port = port
        self.confidence = confidence
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self.last_response = None

    def detect(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        image = encode_rgb_jpeg(rgb, self.jpeg_quality)
        responses = {}
        observations = {}
        for prompt in self.texts:
            arrays, done = call_sam3_text(self.host, self.port, image, [prompt],
                                           self.confidence, self.timeout)
            responses[prompt] = {"arrays": arrays, "done": done}
            masks = arrays["masks"].astype(bool)
            scores = np.asarray(arrays["scores"], dtype=float).reshape(-1)
            if len(masks):
                observations[prompt] = masks[int(np.argmax(scores))].copy()
        self.last_response = responses
        return observations
