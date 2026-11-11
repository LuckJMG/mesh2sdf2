from pathlib import Path

import cupy as cp
import numpy as np

import mesh2sdf.core

_module = cp.RawModule(
    code=Path(__file__).with_name("gpu.cu").read_text(encoding="utf-8"),
    options=("--fmad=false", "--std=c++17"),
)
_init = _module.get_function("init_distances_and_counts")
_sweep_plane = _module.get_function("sweep_plane")
_relayout = _module.get_function("relayout")
_relayout_flip_i = _module.get_function("relayout_flip_i")

TRIANGLE_DATA_BYTES = 152
SWEEPS = (
    (+1, +1, +1), (-1, -1, -1), (+1, +1, -1), (-1, -1, +1),
    (+1, -1, +1), (-1, +1, -1), (+1, -1, -1), (-1, +1, +1),
)
# axes (i, j, k) that each of the 7 checks in sweep_plane steps back along
OFFSETS = ((1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


def _limit(sweep: int, offset: tuple) -> int:
    """Last earlier sweep that ran this check, or -2. See sweep_plane in gpu.cu."""
    d = SWEEPS[sweep % 8]
    for q in range(sweep - 1, -1, -1):
        if all(not step or SWEEPS[q % 8][axis] == d[axis] for axis, step in enumerate(offset)):
            return q
    return -2


LIMITS = np.array([[_limit(s, o) for o in OFFSETS] for s in range(16)], dtype=np.int32)


def compute(vertices: np.ndarray, faces: np.ndarray, size: int = 128) -> np.ndarray:
    """GPU twin of `mesh2sdf.core.compute`. The result is bit-identical."""
    dx = np.float32(mesh2sdf.core.grid_dx(size))
    origin = np.float32(mesh2sdf.core.box_min)
    n = np.int32(size)
    v = cp.asarray(np.ascontiguousarray(vertices, dtype=np.float32))
    f = cp.asarray(np.ascontiguousarray(faces, dtype=np.uint32))
    ntri = len(f)

    phi_init = np.float32(3 * size) * dx
    sentinel = (np.uint64(phi_init.view(np.uint32)) << np.uint64(32)) | np.uint64(0xFFFFFFFF)
    packed = cp.full((size, size, size), sentinel, dtype=cp.uint64)
    counts = cp.zeros((size, size, size), dtype=cp.int32)
    triangle_data = cp.empty(ntri * TRIANGLE_DATA_BYTES, dtype=cp.uint8)
    if ntri:
        _init((ntri,), (128,), (v, f, origin, dx, n, packed, counts, triangle_data))

    words = packed.view(cp.uint32)
    closest_tri = cp.ascontiguousarray(words[..., 0::2]).view(cp.int32)
    phi = cp.ascontiguousarray(words[..., 1::2]).view(cp.float32)
    del packed, words

    # each sweep runs on its own skewed layout, see gpu.cu; (0, 0, 0) is the standard layout
    last_change = cp.full((size, size, size), -1, dtype=cp.int8)
    spare_phi, spare_tri, spare_last = cp.empty_like(phi), cp.empty_like(closest_tri), cp.empty_like(last_change)
    limits = cp.asarray(LIMITS)
    tiles = -(-size // 32)

    def relayout(src, dst):
        nonlocal phi, closest_tri, last_change, spare_phi, spare_tri, spare_last
        layouts = (np.int32(x) for x in (*src, *dst))
        args = (phi, closest_tri, last_change, spare_phi, spare_tri, spare_last, n, *layouts)
        if src[0] != 0 and src[0] != dst[0] and src[1] == dst[1]:
            _relayout_flip_i((tiles, tiles, size), (32, 8), args)
        else:
            _relayout((-(-size // 128), size, size), (128,), args)
        phi, spare_phi, closest_tri, spare_tri = spare_phi, phi, spare_tri, closest_tri
        last_change, spare_last = spare_last, last_change

    block = (32, 8)
    grid = (-(-(size - 1) // block[0]), -(-(size - 1) // block[1]))
    layout = SWEEPS[0]  # init writes this layout
    for sweep, step in enumerate(SWEEPS * 2):
        if step != layout:
            relayout(layout, step)
            layout = step
        args = (triangle_data, phi, closest_tri, last_change, limits[sweep], origin, dx, n,
                *(np.int32(x) for x in step), np.int32(sweep))
        for plane in range(3, 3 * (size - 1) + 1):
            slices = tuple(np.int32((plane - t) % size) for t in range(4))
            _sweep_plane(grid, block, (*args, np.int32(plane), *slices))
    relayout(layout, (0, 0, 0))

    inside = cp.cumsum(counts, axis=2) % 2 == 1
    return cp.asnumpy(cp.where(inside, -phi, phi))
