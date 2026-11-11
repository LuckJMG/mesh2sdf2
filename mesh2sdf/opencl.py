from pathlib import Path

import numpy as np
import pyopencl as cl

import mesh2sdf.core
from mesh2sdf.sweeps import LIMITS, SWEEPS

# PYOPENCL_CTX selects the device, for example PYOPENCL_CTX=0:0.
_context = cl.create_some_context(interactive=False)
_queue = cl.CommandQueue(_context)
for _device in _context.devices:
    if not _device.single_fp_config & cl.device_fp_config.CORRECTLY_ROUNDED_DIVIDE_SQRT:
        raise RuntimeError(
            f"OpenCL device {_device.name!r} has no correctly rounded float division and sqrt, "
            "so its SDF would differ from the CPU. Select another device with PYOPENCL_CTX."
        )
_program = cl.Program(_context, Path(__file__).with_name("sweep.cl").read_text(encoding="utf-8")).build(
    options=["-cl-fp32-correctly-rounded-divide-sqrt"]
)
_sweep_plane = cl.Kernel(_program, "sweep_plane")
_relayout = cl.Kernel(_program, "relayout")
_relayout_flip_i = cl.Kernel(_program, "relayout_flip_i")


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def compute(vertices: np.ndarray, faces: np.ndarray, size: int = 128) -> np.ndarray:
    phi, closest_tri, counts, triangles = mesh2sdf.core.init(vertices, faces, size)
    n = np.int32(size)
    rw = cl.mem_flags.READ_WRITE
    copy = cl.mem_flags.COPY_HOST_PTR

    phi_buf = cl.Buffer(_context, rw, phi.nbytes)
    tri_buf = cl.Buffer(_context, rw, closest_tri.nbytes)
    cl.enqueue_copy(_queue, phi_buf, phi, is_blocking=False)
    cl.enqueue_copy(_queue, tri_buf, closest_tri, is_blocking=False)
    last_buf = cl.Buffer(_context, rw, size**3)
    cl.enqueue_fill_buffer(_queue, last_buf, np.int8(-1), 0, size**3)
    spare_phi, spare_tri, spare_last = (cl.Buffer(_context, rw, b.size) for b in (phi_buf, tri_buf, last_buf))
    # OpenCL does not allow empty buffers, and a mesh without faces reads no triangle
    triangle_buf = cl.Buffer(_context, cl.mem_flags.READ_ONLY | copy, hostbuf=triangles if len(triangles) else np.zeros(1))
    limits_buf = cl.Buffer(_context, cl.mem_flags.READ_ONLY | copy, hostbuf=LIMITS)

    def relayout(src, dst):
        nonlocal phi_buf, tri_buf, last_buf, spare_phi, spare_tri, spare_last
        layouts = (np.int32(x) for x in (*src, *dst))
        args = (phi_buf, tri_buf, last_buf, spare_phi, spare_tri, spare_last, n, *layouts)
        if src[0] != 0 and src[0] != dst[0] and src[1] == dst[1]:
            tiles = _round_up(size, 32) // 32
            _relayout_flip_i(_queue, (tiles * 32, tiles * 8, size), (32, 8, 1), *args)
        else:
            _relayout(_queue, (_round_up(size, 128), size, size), (128, 1, 1), *args)
        phi_buf, spare_phi, tri_buf, spare_tri = spare_phi, phi_buf, spare_tri, tri_buf
        last_buf, spare_last = spare_last, last_buf

    block = (32, 8)
    grid = (_round_up(size - 1, block[0]), _round_up(size - 1, block[1]))
    layout = (0, 0, 0)  # the standard layout of mesh2sdf.core.init
    for sweep, step in enumerate(SWEEPS * 2):
        relayout(layout, step)
        layout = step
        _sweep_plane.set_args(
            triangle_buf, phi_buf, tri_buf, last_buf, limits_buf, np.float32(mesh2sdf.core.box_min),
            np.float32(mesh2sdf.core.grid_dx(size)), n, *(np.int32(x) for x in step), np.int32(sweep),
        )
        for plane in range(3, 3 * (size - 1) + 1):
            cl.enqueue_nd_range_kernel(_queue, _sweep_plane, (*grid, 1), (*block, 1), global_work_offset=(0, 0, plane))
    relayout(layout, (0, 0, 0))

    cl.enqueue_copy(_queue, phi, phi_buf)
    mesh2sdf.core.apply_signs(phi, counts)
    return phi
