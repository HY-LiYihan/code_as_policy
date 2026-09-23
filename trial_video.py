"""Record the actual MuJoCo trajectory from a fixed third-person camera."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import mujoco


class TrialVideo:
    def __init__(self, backend, path: Path, fps: int = 10, width: int = 640, height: int = 480):
        self.backend = backend
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = 0
        self.start_time = backend.data.time
        self._original_step = backend._step
        self._next_frame_time = backend.data.time + 1 / fps
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0, 0.78]
        camera.distance = 1.55
        camera.azimuth = 135
        camera.elevation = -28
        self.camera = camera
        self.renderer = mujoco.Renderer(backend.model, height=height, width=width)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self._capture()
        except BaseException:
            self.close()
            raise
        backend._step = self._step

    def _capture(self):
        self.renderer.update_scene(self.backend.data, camera=self.camera)
        self.encoder.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def _step(self, steps: int = 1, *, pace: bool = True):
        remaining = steps
        while remaining:
            until_frame = max(1, math.ceil(
                (self._next_frame_time - self.backend.data.time - 1e-9)
                / self.backend.model.opt.timestep))
            chunk = min(remaining, until_frame)
            self._original_step(chunk, pace=pace)
            remaining -= chunk
            if self.backend.data.time + 1e-9 >= self._next_frame_time:
                self._capture()
                self._next_frame_time += 1 / self.fps

    def close(self):
        self.backend._step = self._original_step
        try:
            self.simulation_seconds = self.backend.data.time - self.start_time
            while self.frames < self.fps * 2:
                self._capture()
            self.encoder.stdin.close()
            error = self.encoder.stderr.read().decode("utf-8", errors="replace")
            if self.encoder.wait() != 0:
                raise RuntimeError(f"Video encoding failed: {error}")
        finally:
            self.renderer.close()
            if self.encoder.poll() is None:
                self.encoder.kill()
                self.encoder.wait()
