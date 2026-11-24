"""Focused storage staging tests use tiny synthetic files, never H3 weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


from docker import stage_models


class StageModelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.mount = self.base / "volume"
        self.data_root = self.mount / "imagine-h3-turbo"
        self.remote_root = self.data_root / "models"
        self.local_root = self.base / "worker" / "ComfyUI" / "models"
        self.manifest_path = self.base / "manifest.json"
        self.receipt_dir = self.data_root / ".verification-receipts"
        self.mount.mkdir()
        self.remote_root.mkdir(parents=True)
        self.blobs = {
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": b"tiny diffusion model fixture",
            "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": b"tiny turbo lora fixture",
        }
        self.items = []
        for index, (relative, content) in enumerate(self.blobs.items()):
            source = self.remote_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(content)
            self.items.append({
                "id": "fl2va_model" if index == 0 else "fl2v_turbo_lora_8step",
                "path": relative,
                "repo": "Comfy-Org/MiniMax-H3",
                "revision": "a" * 40,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "bytes_approx": len(content),
            })
        self.manifest_path.write_text(json.dumps({"models": self.items[:1], "optional_models": self.items[1:]}))
        self.root_patch = patch.object(stage_models, "ROOT", self.base / "worker")
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.scratch_patch = patch.object(stage_models, "MIN_SCRATCH_BYTES", 0)
        self.scratch_patch.start()
        self.addCleanup(self.scratch_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, mode="direct", verify="full"):
        return stage_models.prepare(
            storage_mode=mode,
            verification_mode=verify,
            mount=self.mount,
            remote_root=self.remote_root,
            local_root=self.local_root,
            manifest_path=self.manifest_path,
            receipt_path=self.receipt_dir,
            mount_check=lambda _: None,
        )

    def test_direct_full_creates_symlinks_without_copying_or_receipt(self):
        result = self.prepare("direct", "full")
        self.assertEqual(result["bytes_copied"], 0)
        self.assertEqual(result["files_full"], len(self.blobs))
        self.assertEqual(result["verification_mode"], "full")
        for relative in self.blobs:
            dest = self.local_root / relative
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(), (self.remote_root / relative).resolve())
        self.assertFalse(self.receipt_dir.exists())

    def test_direct_cached_writes_immutable_receipts_and_reuses_on_second_run(self):
        with patch.object(stage_models.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            first = self.prepare("direct", "cached")
            self.assertEqual(first["files_full"], len(self.blobs))
            self.assertEqual(first["receipts_written"], len(self.blobs))
            receipt_paths = sorted(self.receipt_dir.glob("*.json"))
            before = {path: path.read_bytes() for path in receipt_paths}
            second = self.prepare("direct", "cached")
        self.assertEqual(second["files_cached"], len(self.blobs))
        self.assertEqual(second["bytes_hashed"], 0)
        self.assertEqual(second["verification_mode"], "cached")
        self.assertEqual(before, {path: path.read_bytes() for path in receipt_paths})

    def test_malformed_receipt_causes_full_hash_fallback(self):
        self.receipt_dir.mkdir()
        manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        (self.receipt_dir / f"{manifest_sha}-broken.json").write_text("[]")
        result = self.prepare("direct", "cached")
        self.assertEqual(result["files_full"], len(self.blobs))
        self.assertEqual(result["files_cached"], 0)

    def test_size_mode_always_rejects_wrong_size(self):
        item = self.items[0]
        source = self.remote_root / item["path"]
        source.write_bytes(b"short")
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.prepare("direct", "size")
        self.assertFalse((self.local_root / item["path"]).exists())

    def test_direct_missing_model_does_not_fallback_to_copy_or_download(self):
        (self.remote_root / self.items[0]["path"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unsafe Global Volume model"):
            self.prepare("direct", "full")
        self.assertFalse(any(self.local_root.rglob("*.safetensors")) if self.local_root.exists() else False)

    def test_local_copy_hashes_once_and_size_checks_existing_files(self):
        first = self.prepare("local", "full")
        self.assertEqual(first["bytes_copied"], sum(map(len, self.blobs.values())))
        self.assertEqual(first["bytes_hashed"], first["bytes_copied"])
        self.assertTrue(all(not (self.local_root / relative).is_symlink() for relative in self.blobs))
        second = self.prepare("local", "size")
        self.assertEqual(second["bytes_copied"], 0)
        self.assertEqual(second["files_size_only"], len(self.blobs))
        self.assertEqual(second["verification_mode"], "size")

    def test_local_symlinked_parent_cannot_redirect_model_writes(self):
        local_models = self.local_root
        local_models.mkdir(parents=True)
        (local_models / "diffusion_models").symlink_to(self.remote_root / "diffusion_models")
        with self.assertRaisesRegex(RuntimeError, "escapes ComfyUI models root"):
            stage_models._safe_local_dest(local_models, Path(self.items[0]["path"]))


if __name__ == "__main__":
    unittest.main()
