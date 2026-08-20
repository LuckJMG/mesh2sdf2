import inspect
from pathlib import Path

import numpy as np
import pytest
import trimesh

import mesh2sdf
import mesh2sdf.core


REPO_ROOT = Path(__file__).resolve().parent.parent
PLANE_OBJ = REPO_ROOT / "example" / "data" / "plane.obj"


def _watertight_cube(half: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned cube centered at origin, consistent winding (trimesh.is_watertight)."""
    s = float(half)
    v = np.array(
        [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [1, 5, 6], [1, 6, 2],
            [0, 3, 7], [0, 7, 4],
        ],
        dtype=np.uint32,
    )
    return v, f


@pytest.fixture
def cube_norm() -> tuple[np.ndarray, np.ndarray]:
    """Watertight cube with vertices already in [-1, 1]."""
    v, f = _watertight_cube(half=0.5)
    mesh = trimesh.Trimesh(v, f)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    return v, f


class TestModuleAPI:
    def test_drop_in_import_name(self):
        import mesh2sdf as ms
        assert ms is mesh2sdf

    def test_compute_exposed(self):
        assert callable(mesh2sdf.compute)

    def test_core_version_exposed(self):
        v = mesh2sdf.core.__version__
        assert isinstance(v, str) and v


class TestComputeSignature:
    def test_default_arg_values(self):
        sig = inspect.signature(mesh2sdf.compute)
        params = sig.parameters
        assert set(params) == {
            "vertices",
            "faces",
            "size",
            "fix",
            "level",
            "return_mesh",
        }
        assert params["size"].default == 128
        assert params["fix"].default is False
        assert params["level"].default == 0.015
        assert params["return_mesh"].default is False


class TestOutputContract:
    def test_sdf_shape_default(self, cube_norm):
        v, f = cube_norm
        sdf = mesh2sdf.compute(v, f)
        assert sdf.shape == (128, 128, 128)

    def test_sdf_shape_custom_size(self, cube_norm):
        v, f = cube_norm
        for size in (32, 64, 96):
            sdf = mesh2sdf.compute(v, f, size=size)
            assert sdf.shape == (size, size, size)

    def test_sdf_dtype_float32(self, cube_norm):
        v, f = cube_norm
        sdf = mesh2sdf.compute(v, f, size=32)
        assert sdf.dtype == np.float32

    def test_return_mesh_paths(self, cube_norm):
        v, f = cube_norm
        only = mesh2sdf.compute(v, f, size=32)
        assert isinstance(only, np.ndarray)

        pair = mesh2sdf.compute(v, f, size=32, return_mesh=True)
        assert isinstance(pair, tuple) and len(pair) == 2
        sdf, mesh = pair
        assert isinstance(sdf, np.ndarray)
        assert isinstance(mesh, trimesh.Trimesh)


class TestResolution:
    def test_finer_grid_more_cells(self, cube_norm):
        v, f = cube_norm
        sdf32 = mesh2sdf.compute(v, f, size=32)
        sdf64 = mesh2sdf.compute(v, f, size=64)
        assert sdf64.size == 8 * sdf32.size

    def test_sign_invariant_to_size(self, cube_norm):
        v, f = cube_norm
        for size in (32, 64, 128):
            sdf = mesh2sdf.compute(v, f, size=size)
            mid = size // 2
            assert sdf[mid, mid, mid] < 0


class TestAlgorithmCorrectness:
    def test_center_negative(self, cube_norm):
        v, f = cube_norm
        sdf = mesh2sdf.compute(v, f, size=64)
        mid = 32
        assert sdf[mid, mid, mid] < 0

    def test_corner_positive(self, cube_norm):
        v, f = cube_norm
        sdf = mesh2sdf.compute(v, f, size=64)
        assert sdf[0, 0, 0] > 0
        assert sdf[63, 63, 63] > 0

    def test_face_near_zero(self, cube_norm):
        v, f = cube_norm
        size = 64
        sdf = mesh2sdf.compute(v, f, size=size)
        # cube face x=+0.5 lies at grid x = (0.5+1)/2 * 64 = 48
        assert abs(sdf[48, 32, 32]) < 0.1
        assert abs(sdf[32, 48, 32]) < 0.1
        assert abs(sdf[32, 32, 48]) < 0.1

    def test_vertices_project_to_zero(self, cube_norm):
        v, f = cube_norm
        size = 64
        sdf = mesh2sdf.compute(v, f, size=size)
        for vert in v:
            ix = int(round((vert[0] + 1.0) * size / 2.0))
            iy = int(round((vert[1] + 1.0) * size / 2.0))
            iz = int(round((vert[2] + 1.0) * size / 2.0))
            ix = max(0, min(size - 1, ix))
            iy = max(0, min(size - 1, iy))
            iz = max(0, min(size - 1, iz))
            assert abs(sdf[ix, iy, iz]) < 0.1


class TestFixPath:
    def test_fix_false_passthrough(self, cube_norm):
        v, f = cube_norm
        sdf, mesh = mesh2sdf.compute(v, f, size=32, fix=False, return_mesh=True)
        # fix=False path returns the input verts/faces wrapped in a Trimesh
        assert mesh.vertices.shape == v.shape
        assert mesh.faces.shape == f.shape
        np.testing.assert_array_equal(mesh.vertices, v)
        np.testing.assert_array_equal(mesh.faces, f)

    def test_fix_true_mesh_in_box(self, cube_norm):
        v, f = cube_norm
        _, mesh = mesh2sdf.compute(v, f, size=64, fix=True, return_mesh=True)
        assert mesh.vertices.min() >= -1.0
        assert mesh.vertices.max() <= 1.0

    def test_fix_true_mesh_watertight(self, cube_norm):
        v, f = cube_norm
        _, mesh = mesh2sdf.compute(v, f, size=64, fix=True, return_mesh=True)
        assert mesh.is_watertight

    def test_fix_true_inside_negative(self, cube_norm):
        v, f = cube_norm
        sdf = mesh2sdf.compute(v, f, size=64, fix=True)
        mid = 32
        assert sdf[mid, mid, mid] < 0


class TestExamplePlane:
    def test_plane_obj_exists(self):
        assert PLANE_OBJ.is_file()

    @pytest.fixture
    def plane_workspace(self, tmp_path):
        out_dir = tmp_path / "plane_out"
        out_dir.mkdir()
        return out_dir

    def test_plane_end_to_end(self, plane_workspace):
        mesh = trimesh.load(PLANE_OBJ, force="mesh")
        assert mesh.vertices.shape[0] > 0
        assert mesh.faces.shape[0] > 0

        # match example/test.py normalization (mesh_scale=0.8, 10% padding)
        verts = mesh.vertices
        bbmin = verts.min(0)
        bbmax = verts.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.8 / (bbmax - bbmin).max()
        norm = (verts - center) * scale

        size = 64
        sdf, fixed = mesh2sdf.compute(
            norm, mesh.faces, size, fix=True, level=2 / size, return_mesh=True
        )

        assert sdf.shape == (size, size, size)
        assert sdf.dtype == np.float32
        assert isinstance(fixed, trimesh.Trimesh)
        assert fixed.is_watertight

    def test_plane_outputs_persist(self, plane_workspace):
        mesh = trimesh.load(PLANE_OBJ, force="mesh")
        verts = mesh.vertices
        bbmin = verts.min(0)
        bbmax = verts.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.8 / (bbmax - bbmin).max()
        norm = (verts - center) * scale

        size = 32
        sdf, fixed = mesh2sdf.compute(
            norm, mesh.faces, size, fix=True, level=2 / size, return_mesh=True
        )

        npy = plane_workspace / "plane.npy"
        obj = plane_workspace / "plane.fixed.obj"
        np.save(npy, sdf)
        fixed.vertices = fixed.vertices / scale + center
        fixed.export(obj)

        assert npy.is_file()
        assert obj.is_file()
        loaded = np.load(npy)
        assert loaded.shape == (size, size, size)
        # tearDown is automatic via tmp_path
