"""检查简化分支的独立输出、实体有效性与失败时保留原结果。"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import scene_creator as scene


class SimplificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".scene_validation_", dir=scene.ASSET_DIR)
        self.folder = Path(self.temporary.name).resolve()
        # 清理前确认递归删除的临时目录确实位于本次指定的 Assets 中。
        if self.folder.parent != scene.ASSET_DIR.resolve():
            raise AssertionError("临时目录不在预期的 Assets 下")
        self.addCleanup(self.temporary.cleanup)
        self.input = self.folder / "input.obj"
        trimesh.creation.box(extents=[0.3, 0.3, 0.6]).export(self.input)
        self.original_input = self.input.read_bytes()
        self.output = self.folder / "scene.obj"
        self.raw = self.folder / "raw.obj"

    def create(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return scene.create_scene(obj_path=self.input, output_path=self.output,
                                      raw_output_path=self.raw, obj_up_axis="Z", **kwargs)

    def test_enabled_keeps_original_outputs_and_return_value(self):
        with patch.object(scene, "simplify_mesh_manifold", side_effect=AssertionError("默认不得简化")):
            self.assertEqual(self.create(), self.output)
        simplified_path = self.output.with_name("scene_simplified.obj")
        self.assertFalse(simplified_path.exists())
        original, raw = self.output.read_bytes(), self.raw.read_bytes()
        self.assertEqual(self.create(simplify=True), self.output)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(self.raw.read_bytes(), raw)
        self.assertEqual(self.input.read_bytes(), self.original_input)
        simplified = trimesh.load(simplified_path, force="mesh")
        self.assertTrue(simplified.is_volume)
        self.assertEqual(len(simplified.split()), 1)

    def test_custom_path_and_axis(self):
        for axis in ("Y", "Z"):
            path = self.folder / axis / "custom.obj"
            self.create(simplify=True, simplified_output_path=path, up_axis=axis)
            simplified = trimesh.load(path, force="mesh")
            original = trimesh.load(self.output, force="mesh")
            self.assertTrue(simplified.is_volume)
            np.testing.assert_allclose(simplified.bounds, original.bounds, atol=0.001)
        y_mesh = trimesh.load(self.folder / "Y" / "custom.obj", force="mesh")
        z_mesh = trimesh.load(self.folder / "Z" / "custom.obj", force="mesh")
        z_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        np.testing.assert_allclose(y_mesh.vertices, z_mesh.vertices, atol=1e-7)

    def test_conflicts_rejected_before_writes(self):
        for target in (self.input, self.output, self.raw, self.folder / "subdir" / ".." / "scene.obj"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "不能相互覆盖"):
                self.create(simplify=True, simplified_output_path=target)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.raw.exists())
            self.assertEqual(self.input.read_bytes(), self.original_input)

    def test_invalid_options_rejected_before_writes(self):
        options = [dict(simplify=True, simplify_tolerance=value) for value in (0, -1, np.nan, np.inf)]
        options += [dict(simplified_output_path=self.folder / "copy.obj"),
                    dict(simplify=True, simplified_output_path=self.folder / "copy.ply")]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.create(**kwargs)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.raw.exists())

    def test_failed_branch_preserves_original_and_prior_simplified_file(self):
        simplified_path = self.folder / "scene_simplified.obj"
        simplified_path.write_bytes(b"prior simplified output")
        self.create()
        original = self.output.read_bytes()
        with patch.object(scene, "simplify_mesh_manifold", side_effect=RuntimeError("forced failure")):
            with self.assertRaisesRegex(ValueError, "原布尔结果已保留"):
                self.create(simplify=True)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(simplified_path.read_bytes(), b"prior simplified output")
        self.assertEqual(self.input.read_bytes(), self.original_input)

    def test_manifold_copy_and_tolerance_floor(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        result = scene.simplify_mesh_manifold(mesh, 0.001)
        self.assertTrue(result.is_volume)
        self.assertLessEqual(len(result.faces), len(faces))
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        with self.assertRaisesRegex(ValueError, "实际最小容差"):
            scene.simplify_mesh_manifold(mesh, 1e-20)

    def test_tolerance_floor_failure_keeps_original(self):
        with self.assertRaisesRegex(ValueError, "实际最小容差.*原布尔结果已保留"):
            self.create(simplify=True, simplify_tolerance=1e-20)
        self.assertTrue(trimesh.load(self.output, force="mesh").is_volume)
        self.assertTrue(self.raw.is_file())
        self.assertFalse(self.folder.joinpath("scene_simplified.obj").exists())


if __name__ == "__main__":
    unittest.main()
