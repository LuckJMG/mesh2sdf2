import numpy as np
import pytest
import trimesh

import mesh2sdf


@pytest.mark.parametrize("size", [64, 128])
def test_e2e_plane_fix(benchmark, plane_norm, size):
    """End-to-end with fix=True on plane: includes marching-cubes + re-sweep."""
    v, f = plane_norm[:2]
    sdf = benchmark(mesh2sdf.compute, v, f, size, fix=True, level=2.0 / size)
    assert sdf.shape == (size, size, size)
    assert sdf.dtype == np.float32


@pytest.mark.parametrize("size", [64, 128])
def test_e2e_plane_no_fix(benchmark, plane_norm, size):
    """Control: fix=False isolates the C++ fast-sweep cost."""
    v, f = plane_norm[:2]
    sdf = benchmark(mesh2sdf.compute, v, f, size, fix=False)
    assert sdf.shape == (size, size, size)
    assert sdf.dtype == np.float32


@pytest.mark.parametrize("size", [64, 128])
def test_e2e_icosphere_fix(benchmark, size):
    """End-to-end fix=True on an already-watertight mesh: marching-cubes still runs."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    v = mesh.vertices.astype(np.float32, copy=False)
    f = mesh.faces
    sdf = benchmark(mesh2sdf.compute, v, f, size, fix=True, level=2.0 / size)
    assert sdf.shape == (size, size, size)
    assert sdf.dtype == np.float32
