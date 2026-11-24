import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from docker import model_downloader, stage_models


class ModelDownloaderTests(unittest.TestCase):
    def test_download_is_pinned_verified_and_reused(self):
        payload = b"small model fixture"
        item = {
            "id": "fixture", "path": "vae/fixture.safetensors",
            "repo": "fixture/fixture", "revision": "a" * 40,
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fetch(request, scratch, force):
                calls.append((request["revision"], force))
                dest = scratch / request["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                return dest

            options = dict(download_file=fetch, scratch_reserve_bytes=0)
            first = model_downloader.download_pinned_models([item], root, **options)
            second = model_downloader.download_pinned_models([item], root, **options)
            self.assertEqual(calls, [("a" * 40, False)])
            self.assertEqual(first["bytes_downloaded"], len(payload))
            self.assertEqual(second["files_reused"], 1)
            self.assertEqual((root / item["path"]).read_bytes(), payload)

            (root / item["path"]).unlink()
            with patch.object(model_downloader, "_hub_download", side_effect=lambda request, scratch, force: fetch({**request, "revision": "b" * 40}, scratch, force)):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    model_downloader.download_pinned_models(
                        [{**item, "sha256": "0" * 64}], root,
                        scratch_reserve_bytes=0, sleep_fn=lambda seconds: None,
                    )
            self.assertFalse((root / item["path"]).exists())

    def test_startup_uses_download_without_mount(self):
        report = {"files_downloaded": 5, "files_reused": 0, "bytes_downloaded": 42}
        with patch.dict("os.environ", {"IMAGINE_DOWNLOAD_MODELS": "1", "IMAGINE_MODEL_STORAGE_MODE": "download"}):
            with patch.object(stage_models.subprocess, "run", side_effect=AssertionError("must not check a mount")):
                if hasattr(stage_models, "prepare"):
                    with patch.object(stage_models, "download_pinned_models", return_value=report) as fetch:
                        result = stage_models.prepare()
                    self.assertEqual(result["verification_mode"], "full")
                else:
                    with patch.object(stage_models, "DOWNLOAD", True), patch.object(stage_models, "STORAGE_MODE", "download"):
                        with patch.object(model_downloader, "download_pinned_models", return_value=report) as fetch:
                            stage_models.main()
                items, _root = fetch.call_args.args
                self.assertEqual(len(items), 5)
                self.assertEqual(sum(item["bytes"] for item in items), 42030035159)


if __name__ == "__main__":
    unittest.main()
