from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANE_OBJ = REPO_ROOT / "example" / "data" / "plane.obj"


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
