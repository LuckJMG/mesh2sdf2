from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLANE_OBJ = REPO_ROOT / "example" / "data" / "plane.obj"


def _watertight_cube(half: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned cube centered at origin, consistent winding."""
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
    v, f = _watertight_cube(half=0.5)
    return v, f


@pytest.fixture(scope="session")
def plane_norm() -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(PLANE_OBJ, force="mesh")
    verts = mesh.vertices
    bbmin = verts.min(0)
    bbmax = verts.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.8 / (bbmax - bbmin).max()
    norm = (verts - center) * scale
    return norm.astype(np.float32), mesh.faces
