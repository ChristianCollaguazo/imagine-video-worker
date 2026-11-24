from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.lora_catalog import available_lora_names, resolve_lora_name, resolve_lora_stack
from app.workflow import build_workflow, canvas_for_image, sampling_steps


class GenerationOptionsTests(unittest.TestCase):
    def test_steps_accept_one_through_fifty_including_four(self):
        for value in (1, 4, 8, 12, 20, 50):
            self.assertEqual(sampling_steps(value), value)
        for value in (0, 51, -1):
            with self.assertRaises(ValueError):
                sampling_steps(value)

    def test_supported_resolutions_are_short_edge_targets(self):
        for resolution in (480, 576, 768):
            width, height = canvas_for_image(1600, 900, resolution)
            self.assertEqual(width % 32, 0)
            self.assertEqual(height % 32, 0)
            self.assertLessEqual(width, round((resolution * 1344 / 768) / 32) * 32)
            self.assertLess(abs(width / height - 16 / 9), 0.04)
        self.assertEqual(canvas_for_image(1600, 900, 768), (1344, 768))
        with self.assertRaises(ValueError):
            canvas_for_image(1600, 900, 720)

    def test_extreme_aspect_ratio_scales_both_sides_within_h3_native_area(self):
        width, height = canvas_for_image(5000, 1000, 768)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertLessEqual(width * height, 1344 * 768)
        self.assertLess(abs(width / height - 5.0), 0.1)

    def test_workflow_applies_requested_lora_steps_and_resolution(self):
        workflow = build_workflow(
            "a person walking", "input.png", 10, 1, 16, 9, "job",
            steps=4, resolution=480, lora_name="custom.safetensors", lora_strength=0.65,
            audio=False,
        )
        self.assertEqual(workflow["7"]["inputs"]["steps"], 4)
        self.assertEqual((workflow["6"]["inputs"]["width"], workflow["6"]["inputs"]["height"]), canvas_for_image(16, 9, 480))
        self.assertEqual(workflow["16"]["inputs"]["lora_name"], "custom.safetensors")
        self.assertEqual(workflow["16"]["inputs"]["strength_model"], 0.65)
        self.assertNotIn("13", workflow)

    def test_lora_catalog_is_rescanned_and_only_returns_basenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            loras = root / "models" / "loras"
            loras.mkdir(parents=True)
            (loras / "one.safetensors").write_bytes(b"fixture")
            (loras / "ignore.txt").write_bytes(b"fixture")
            self.assertEqual(available_lora_names(loras), ["one.safetensors"])
            (loras / "two.safetensors").write_bytes(b"fixture")
            self.assertEqual(available_lora_names(loras), ["one.safetensors", "two.safetensors"])

    def test_lora_selection_defaults_and_rejects_traversal(self):
        default_lora = "default.safetensors"
        catalog = [default_lora, "other.safetensors"]
        self.assertEqual(resolve_lora_name("turbo", None, catalog, default_lora), default_lora)
        self.assertIsNone(resolve_lora_name("base", None, catalog, default_lora))
        self.assertEqual(resolve_lora_name("base", "other.safetensors", catalog, default_lora), "other.safetensors")
        for name in ("../other.safetensors", "x/other.safetensors", "\\other.safetensors"):
            with self.assertRaises(ValueError):
                resolve_lora_name("base", name, catalog, default_lora)

    def test_lora_stack_preserves_order_and_validates_entries(self):
        names = ["one.safetensors", "two.safetensors"]
        stack = [
            {"name": names[0], "strength": 0.35},
            {"name": names[1], "strength": 1.25},
        ]
        self.assertEqual(resolve_lora_stack("base", stack, names, "default.safetensors"), stack)
        self.assertEqual(resolve_lora_stack("turbo", None, names + ["default.safetensors"], "default.safetensors"), [{"name": "default.safetensors", "strength": 1.0}])
        self.assertEqual(resolve_lora_stack("turbo", [], names, "default.safetensors"), [])
        for invalid in (
            [{"name": names[0], "strength": 1}, {"name": names[0], "strength": 1}],
            [{"name": "../one.safetensors", "strength": 1}],
            [{"name": names[0], "strength": 2.01}],
            [{"name": names[0], "strength": True}],
            [{"name": names[0], "strength": 1, "extra": "x"}],
        ):
            with self.assertRaises(ValueError):
                resolve_lora_stack("base", invalid, names, "default.safetensors")

    def test_workflow_chains_multiple_loras_in_requested_order(self):
        workflow = build_workflow(
            "x" * 10000, "input.png", 10, 1, 16, 9, "job", steps=8,
            lora_stack=[{"name": "one.safetensors", "strength": 0.25}, {"name": "two.safetensors", "strength": 1.5}],
        )
        self.assertEqual(workflow["6"]["inputs"]["prompt"], "x" * 10000)
        self.assertEqual(workflow["16"]["inputs"], {"model": ["1", 0], "lora_name": "one.safetensors", "strength_model": 0.25})
        self.assertEqual(workflow["17"]["inputs"], {"model": ["16", 0], "lora_name": "two.safetensors", "strength_model": 1.5})
        self.assertEqual(workflow["7"]["inputs"]["model"], ["17", 0])
        self.assertEqual(workflow["10"]["inputs"]["model"], ["17", 0])


if __name__ == "__main__":
    unittest.main()
