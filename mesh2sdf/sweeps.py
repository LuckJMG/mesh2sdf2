import numpy as np

SWEEPS = (
    (+1, +1, +1), (-1, -1, -1), (+1, +1, -1), (-1, -1, +1),
    (+1, -1, +1), (-1, +1, -1), (+1, -1, -1), (-1, +1, +1),
)
# axes (i, j, k) that each of the 7 checks in sweep_plane steps back along
OFFSETS = ((1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


# Last earlier sweep that ran this check, or -2 if none did. See sweep_plane in gpu.cu.
def _limit(sweep: int, offset: tuple) -> int:
    d = SWEEPS[sweep % 8]
    for q in range(sweep - 1, -1, -1):
        if all(not step or SWEEPS[q % 8][axis] == d[axis] for axis, step in enumerate(offset)):
            return q
    return -2


LIMITS = np.array([[_limit(s, o) for o in OFFSETS] for s in range(16)], dtype=np.int32)
