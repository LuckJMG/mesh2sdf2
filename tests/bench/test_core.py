import mesh2sdf.core
import numpy as np
import pytest
import trimesh


@pytest.mark.parametrize("size", [32, 64, 128])
def test_core_cube_size(benchmark, cube_norm, size):
    """Watertight cube: tiny deterministic baseline across resolutions."""
    v, f = cube_norm
    sdf = benchmark(mesh2sdf.core.compute, v, f, size)
    assert sdf.shape == (size, size, size)
    assert sdf.dtype == np.float32


@pytest.mark.parametrize("size", [64, 128])
def test_core_plane_size(benchmark, plane_norm, size):
    """Plane mesh: grid-resolution sweep on a real non-watertight mesh."""
    v, f = plane_norm[:2]
    sdf = benchmark(mesh2sdf.core.compute, v, f, size)
    assert sdf.shape == (size, size, size)
    assert sdf.dtype == np.float32


@pytest.mark.parametrize("subdivisions", [2, 3, 4])
def test_core_complexity(benchmark, subdivisions):
    """Icosphere: mesh-complexity sweep at fixed size=128."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    v = mesh.vertices.astype(np.float32, copy=False)
    f = mesh.faces
    sdf = benchmark(mesh2sdf.core.compute, v, f, 128)
    assert sdf.shape == (128, 128, 128)
    assert sdf.dtype == np.float32
