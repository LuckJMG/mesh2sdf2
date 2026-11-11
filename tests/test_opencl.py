"""Bitwise equivalence of the OpenCL sweep (mesh2sdf/sweep.cl) and the CPU path.

Runs on the device that PYOPENCL_CTX selects, or on the first one. The dev
dependencies install pocl, an OpenCL runtime for the CPU, so the test also
runs on machines without a GPU. size=8 and 33 check grids that are smaller
than or not a multiple of the 32 x 32 relayout tile.
"""

import numpy as np
import pytest
import trimesh

import mesh2sdf
from tests.conftest import PLANE_OBJ, normalize

cl = pytest.importorskip("pyopencl")
try:
    from mesh2sdf import opencl  # noqa: F401
except (cl.Error, RuntimeError) as e:
    pytest.skip(f"no usable OpenCL device: {e}", allow_module_level=True)

MESHES = {
    "cube": lambda: trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
    "icosphere4": lambda: trimesh.creation.icosphere(subdivisions=4),
    "plane": lambda: trimesh.load(PLANE_OBJ, force="mesh"),
}


@pytest.mark.parametrize("size", [8, 33, 96])
@pytest.mark.parametrize("name", MESHES)
def test_opencl_bitwise_matches_cpu(name, size):
    v, f, _, _ = normalize(MESHES[name]())
    cpu = mesh2sdf.compute(v, f, size=size)
    sdf = mesh2sdf.compute(v, f, size=size, device="opencl")
    assert sdf.dtype == np.float32
    assert np.array_equal(cpu, sdf)


def test_opencl_fix_matches_cpu(plane_norm):
    v, f, _, _ = plane_norm
    cpu = mesh2sdf.compute(v, f, size=64, fix=True, level=2 / 64)
    sdf = mesh2sdf.compute(v, f, size=64, fix=True, level=2 / 64, device="opencl")
    assert np.array_equal(cpu, sdf)


def test_opencl_empty_mesh_matches_cpu():
    v = np.zeros((0, 3), dtype=np.float32)
    f = np.zeros((0, 3), dtype=np.uint32)
    assert np.array_equal(mesh2sdf.compute(v, f, size=8), mesh2sdf.compute(v, f, size=8, device="opencl"))
