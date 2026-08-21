import numpy as np
import pytest


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
