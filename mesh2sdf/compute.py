import numpy as np
import skimage.measure
import trimesh

import mesh2sdf.core


def compute(
    vertices: np.ndarray,
    faces: np.ndarray,
    size: int = 128,
    fix: bool = False,
    level: float = 0.015,
    return_mesh: bool = False,
    device: str = "cpu",
):
    r"""Converts a input mesh to signed distance field (SDF).

    Args:
      vertices (np.ndarray): The vertices of the input mesh, the vertices MUST be
          in range [-1, 1].
      faces (np.ndarray): The faces of the input mesh.
      size (int): The resolution of the resulting SDF.
      fix (bool): If the input mesh is not watertight, set :attr:`fix` as True.
      level (float): The value used to extract level sets when :attr:`fix` is True,
          with a default value of 0.015 (as a reference 2/128 = 0.015625). And the
          recommended default value is 2/size.
      return_mesh (bool): If True, also return the fixed mesh.
      device (str): 'cpu', or 'cuda' to compute the SDF on the GPU with CuPy.
          Both devices give bit-identical results.
    """
    if device == "cpu":
        sdf_core = mesh2sdf.core.compute
    elif device == "cuda":
        try:
            from mesh2sdf import gpu
        except ImportError as e:
            raise ImportError("device='cuda' needs CuPy: pip install mesh2sdf2[gpu]") from e
        sdf_core = gpu.compute
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

    sdf = sdf_core(vertices, faces, size)
    if not fix:
        return (sdf, trimesh.Trimesh(vertices, faces)) if return_mesh else sdf

    # NOTE: the negative value is not reliable if the mesh is not watertight
    sdf = np.abs(sdf)
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

    mesh = trimesh.Trimesh(vertices, faces)
    mesh = max(mesh.split(only_watertight=False),
               key=lambda c: c.extents.max())
    mesh.vertices = mesh.vertices * mesh2sdf.core.grid_dx(size) + mesh2sdf.core.box_min

    sdf = sdf_core(mesh.vertices, mesh.faces, size)
    return (sdf, mesh) if return_mesh else sdf
