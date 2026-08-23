"""Bitwise equivalence of the serial and wavefront-parallel fast sweep.

The sweep dispatches to the OpenMP wavefront path only when more than one
thread is available and the grid holds at least 2**18 cells, so size=96 with
OMP_NUM_THREADS>1 exercises `sweep_parallel` while OMP_NUM_THREADS=1 forces
the legacy sequential path. The wavefront level order is a topological order
of the same dependency DAG as the sequential walk, so every configuration
must produce exactly identical arrays.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANE_OBJ = REPO_ROOT / "example" / "data" / "plane.obj"
SIZE = 96  # above the 2**18-cell threshold in makelevelset3.cpp
THREAD_COUNTS = ("1", "2", "5")

_RUNNER = """
import sys
import numpy as np
import mesh2sdf

data = np.load(sys.argv[1])
sdf = mesh2sdf.compute(data["v"], data["f"], size=int(sys.argv[3]))
np.save(sys.argv[2], sdf)
"""


def _normalized(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    verts = mesh.vertices
    bbmin, bbmax = verts.min(0), verts.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.8 / (bbmax - bbmin).max()
    v = ((verts - center) * scale).astype(np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    return v, f


@pytest.fixture(scope="module")
def mesh_inputs(tmp_path_factory) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("sweep_eq_inputs")
    meshes = {
        "cube": trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
        "icosphere": trimesh.creation.icosphere(subdivisions=3),
        "plane": trimesh.load(PLANE_OBJ, force="mesh"),
    }
    inputs = {}
    for name, mesh in meshes.items():
        v, f = _normalized(mesh)
        path = out_dir / f"{name}.npz"
        np.savez(path, v=v, f=f)
        inputs[name] = path
    return inputs


@pytest.mark.parametrize("name", ["cube", "icosphere", "plane"])
def test_wavefront_bitwise_matches_serial(name, mesh_inputs, tmp_path):
    outputs = []
    for threads in THREAD_COUNTS:
        sdf_path = tmp_path / f"{name}.t{threads}.npy"
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        subprocess.run(
            [sys.executable, "-c", _RUNNER,
             str(mesh_inputs[name]), str(sdf_path), str(SIZE)],
            env=env, check=True, capture_output=True, text=True,
        )
        outputs.append(np.load(sdf_path))

    reference = outputs[0]
    assert reference.shape == (SIZE, SIZE, SIZE)
    for candidate in outputs[1:]:
        assert np.array_equal(reference, candidate)
