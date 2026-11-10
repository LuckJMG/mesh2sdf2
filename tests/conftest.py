from pathlib import Path

import numpy as np
import pytest
import trimesh

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


@pytest.fixture(scope="session")
def cube_norm() -> tuple[np.ndarray, np.ndarray]:
    """Watertight cube with vertices already in [-1, 1]."""
    v, f = _watertight_cube(half=0.5)
    mesh = trimesh.Trimesh(v, f)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    return v, f


@pytest.fixture(scope="session")
def plane_norm() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """plane.obj normalized like example/demo.py (mesh_scale=0.8, 10% padding).

    Returns (vertices, faces, center, scale). Multiply vertices by ``scale``
    and add ``center`` to undo the normalization.
    """
    mesh = trimesh.load(PLANE_OBJ, force="mesh")
    verts = mesh.vertices
    bbmin = verts.min(0)
    bbmax = verts.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.8 / (bbmax - bbmin).max()
    v = ((verts - center) * scale).astype(np.float32)
    return v, mesh.faces, center, scale
