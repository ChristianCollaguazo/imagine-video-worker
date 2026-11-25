import asyncio
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app import main


class LoraUploadTests(unittest.TestCase):
    def test_filename_requires_a_safe_safetensors_basename(self):
        self.assertEqual(main._safe_lora_filename("style.safetensors"), "style.safetensors")
        for filename in (None, "", ".hidden.safetensors", "../style.safetensors", "a/b.safetensors", "a\\b.safetensors", "style.SAFETENSORS"):
            with self.subTest(filename=filename), self.assertRaises(HTTPException):
                main._safe_lora_filename(filename)

    def test_streams_and_atomically_publishes_without_overwriting(self):
        with TemporaryDirectory() as directory:
            comfy_root = Path(directory)
            lora_dir = comfy_root / "models" / "loras"
            lora_dir.mkdir(parents=True)
            content = b"safetensors fixture"
            upload = UploadFile(filename="style.safetensors", file=io.BytesIO(content))
            with patch.object(main, "COMFY_ROOT", comfy_root), patch.object(main, "MAX_LORA_UPLOAD_BYTES", 1024):
                size = asyncio.run(main._store_lora(upload, "style.safetensors"))
                self.assertEqual(size, len(content))
                self.assertEqual((lora_dir / "style.safetensors").read_bytes(), content)
                self.assertEqual(main._available_lora_names(), ["style.safetensors"])
                duplicate = UploadFile(filename="style.safetensors", file=io.BytesIO(b"replacement"))
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(main._store_lora(duplicate, "style.safetensors"))
                self.assertEqual(raised.exception.status_code, 409)
                self.assertEqual((lora_dir / "style.safetensors").read_bytes(), content)
                self.assertEqual(list(lora_dir.iterdir()), [lora_dir / "style.safetensors"])

    def test_rejects_oversized_or_empty_files_and_cleans_temporary_file(self):
        with TemporaryDirectory() as directory:
            comfy_root = Path(directory)
            lora_dir = comfy_root / "models" / "loras"
            lora_dir.mkdir(parents=True)
            with patch.object(main, "COMFY_ROOT", comfy_root), patch.object(main, "MAX_LORA_UPLOAD_BYTES", 3):
                for payload, expected in ((b"four", 413), (b"", 422)):
                    upload = UploadFile(filename="style.safetensors", file=io.BytesIO(payload))
                    with self.subTest(expected=expected), self.assertRaises(HTTPException) as raised:
                        asyncio.run(main._store_lora(upload, "style.safetensors"))
                    self.assertEqual(raised.exception.status_code, expected)
                    self.assertEqual(list(lora_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
