import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from repro_metadata import backend_metadata, build_manifest, wilson_interval


class ReproductionMetadataTests(unittest.TestCase):
    def test_archived_official_files_are_present(self):
        root = Path(__file__).parents[1]
        self.assertTrue((root / "original" / "README.original.md").is_file())
        notebooks = list((root / "original" / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 8)

    def test_legacy_results_are_indexed(self):
        root = Path(__file__).parents[1]
        index = root / "results" / "archive" / "legacy-runs-20260923" / "INDEX.md"
        self.assertTrue(index.is_file())
        self.assertIn("paper-50-aligned", index.read_text())

    def test_backend_metadata_is_clean_and_frozen(self):
        metadata = backend_metadata()
        self.assertEqual(metadata["repository"], "https://github.com/HY-LiYihan/robot_control")
        self.assertEqual(len(metadata["commit"]), 40)
        self.assertIn(metadata["branch"], {"main", "detached"})
        pinned = subprocess.check_output(
            ["git", "ls-files", "--stage", "--", "robot_control"],
            cwd=Path(__file__).parents[1], text=True,
        ).split()
        self.assertEqual(pinned[0], "160000")
        self.assertEqual(metadata["commit"], pinned[1])

    def test_manifest_rejects_stale_workspace_calibration(self):
        backend = backend_metadata()
        scene_protocol = {"workspace_calibration": {"backend_commit": "0" * 40}}
        with self.assertRaisesRegex(ValueError, "Workspace calibration backend differs"):
            build_manifest(
                suite="paper", protocol="paper-sim", model=None, seed=0, trials=1,
                visual_sam3=False, require_downward_ik=True, blocks_only=False,
                prompt_commit=None, prompt_sha256={}, scene_protocol=scene_protocol,
            )
        scene_protocol["workspace_calibration"]["backend_commit"] = backend["commit"]
        manifest = build_manifest(
            suite="paper", protocol="paper-sim", model=None, seed=0, trials=1,
            visual_sam3=False, require_downward_ik=True, blocks_only=False,
            prompt_commit=None, prompt_sha256={}, scene_protocol=scene_protocol,
        )
        self.assertEqual(manifest["backend"]["commit"], backend["commit"])

    def test_wilson_interval(self):
        interval = wilson_interval(10, 22)
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], 10 / 22)
        self.assertGreaterEqual(interval[1], 10 / 22)


if __name__ == "__main__":
    unittest.main()
