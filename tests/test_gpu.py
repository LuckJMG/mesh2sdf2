"""Bitwise equivalence of the CUDA path (mesh2sdf/gpu.cu) and the CPU path.

Skipped when CuPy or a CUDA device is missing. size=33 checks a grid that is
not a multiple of the 32x8 thread block. icosphere4 crosses the parallel init
threshold on the CPU side.
"""

import numpy as np
import pytest
import trimesh

import mesh2sdf
from tests.conftest import PLANE_OBJ, normalize

cp = pytest.importorskip("cupy")
if not cp.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

MESHES = {
    "cube": lambda: trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
    "icosphere4": lambda: trimesh.creation.icosphere(subdivisions=4),
    "plane": lambda: trimesh.load(PLANE_OBJ, force="mesh"),
}


@pytest.mark.parametrize("size", [33, 96])
@pytest.mark.parametrize("name", MESHES)
def test_gpu_bitwise_matches_cpu(name, size):
    v, f, _, _ = normalize(MESHES[name]())
    cpu = mesh2sdf.compute(v, f, size=size)
    gpu = mesh2sdf.compute(v, f, size=size, device="cuda")
    assert gpu.dtype == np.float32
    assert np.array_equal(cpu, gpu)


def test_gpu_fix_matches_cpu(plane_norm):
    v, f, _, _ = plane_norm
    cpu = mesh2sdf.compute(v, f, size=64, fix=True, level=2 / 64)
    gpu = mesh2sdf.compute(v, f, size=64, fix=True, level=2 / 64, device="cuda")
    assert np.array_equal(cpu, gpu)


def test_gpu_empty_mesh_matches_cpu():
    v = np.zeros((0, 3), dtype=np.float32)
    f = np.zeros((0, 3), dtype=np.uint32)
    assert np.array_equal(mesh2sdf.compute(v, f, size=8), mesh2sdf.compute(v, f, size=8, device="cuda"))
