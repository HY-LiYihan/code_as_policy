import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrialVideoTests(unittest.TestCase):
    def test_third_person_video_contains_actual_motion(self):
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("mjpython and ffmpeg are required for offscreen video")
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "trial.mp4"
            subprocess.run([str(mjpython), "-c", "\n".join([
                "from pathlib import Path",
                "from piper_demo import MuJoCoTabletop, SCENE",
                "from trial_video import TrialVideo",
                "tabletop = MuJoCoTabletop(SCENE)",
                f"video = TrialVideo(tabletop.backend, Path({str(video)!r}))",
                "tabletop.robot.gripper(0.08)",
                "tabletop.robot.step(350)",
                "assert video.frames >= 7",
                "video.close()",
                "tabletop.disconnect()",
            ])], check=True, capture_output=True, text=True)
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video)],
                                   check=True, capture_output=True, text=True)
            self.assertGreaterEqual(int(probe.stdout.strip()), 7)
            frames = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "framemd5", "-"],
                                    check=True, capture_output=True, text=True)
            hashes = {line.split(",")[-1].strip() for line in frames.stdout.splitlines()
                      if line and not line.startswith("#")}
            self.assertGreater(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
