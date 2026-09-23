import json
import socket
import struct
import unittest
from unittest.mock import patch

import numpy as np

from sam3_client import SAM3TextSegmenter, call_sam3_text, encode_rgb_jpeg


def frame(payload):
    return struct.pack("<I", len(payload)) + payload


class FakeConnection:
    def __init__(self, incoming):
        self.incoming = bytearray(incoming)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def settimeout(self, timeout):
        del timeout

    def sendall(self, payload):
        self.sent.append(payload)

    def recv(self, size):
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result


class Sam3ClientTests(unittest.TestCase):
    def test_call_uses_tutorial_framing_and_reads_arrays(self):
        masks = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
        boxes = np.array([[0, 0, 2, 2]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        incoming = b"".join([
            frame(json.dumps({"type": "status", "stage": "received"}).encode()),
            frame(json.dumps({"type": "array", "name": "masks", "shape": [1, 2, 2],
                              "dtype": "uint8", "bytes": masks.nbytes}).encode()), frame(masks.tobytes()),
            frame(json.dumps({"type": "array", "name": "boxes", "shape": [1, 4],
                              "dtype": "float32", "bytes": boxes.nbytes}).encode()), frame(boxes.tobytes()),
            frame(json.dumps({"type": "array", "name": "scores", "shape": [1],
                              "dtype": "float32", "bytes": scores.nbytes}).encode()), frame(scores.tobytes()),
            frame(json.dumps({"type": "done", "k": 1, "prompts": ["orange block"]}).encode()),
        ])
        connection = FakeConnection(incoming)
        with patch("sam3_client.socket.create_connection", return_value=connection):
            arrays, done = call_sam3_text("127.0.0.1", 28317, b"jpeg", ["orange block"])
        request = json.loads(connection.sent[0][4:])
        self.assertEqual(request["model"], "sam3")
        self.assertEqual(request["mode"], "text")
        self.assertEqual(request["image_bytes"], 4)
        self.assertEqual(connection.sent[1][4:], b"jpeg")
        np.testing.assert_array_equal(arrays["masks"], masks)
        self.assertEqual(done["k"], 1)

    def test_segmenter_returns_mask_by_prompt_and_encodes_rgb(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertTrue(encode_rgb_jpeg(image).startswith(b"\xff\xd8"))
        segmenter = SAM3TextSegmenter(["orange block"])
        arrays = {"masks": np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
                  "boxes": np.zeros((1, 4), dtype=np.float32),
                  "scores": np.array([0.9], dtype=np.float32)}
        with patch("sam3_client.call_sam3_text", return_value=(arrays, {"prompts": ["orange block"]})):
            masks = segmenter.detect(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(set(masks), {"orange block"})
        self.assertEqual(masks["orange block"].dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
