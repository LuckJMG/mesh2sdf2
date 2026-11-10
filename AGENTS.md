# mesh2sdf2

Fork of [mesh2sdf](https://github.com/wang-ps/mesh2sdf). The PyPI name is
`mesh2sdf2`. The import name stays `mesh2sdf` for drop-in compatibility.
Same algorithm and public API as upstream. Python 3.10–3.14, C++20.

## Algorithm

1. Compute an unsigned SDF on the input mesh with fast sweeping.
2. If `fix=True`, run marching cubes on `abs(sdf)` at `level` and keep the largest component.
3. Compute the signed SDF on the fixed mesh.

## Layout

- `mesh2sdf/compute.py` — Python wrapper and the `fix=True` path.
- `csrc/pybind.cpp` — pybind11 module `mesh2sdf.core`. This file is the only Modern C++ file.
- `csrc/makelevelset3.cpp` — serial sweep, init, signs, and the dispatcher.
- `csrc/parallel.h/.cpp` — OpenMP sweep and init, with size thresholds.
- `csrc/geometry.h`, `csrc/array3.h`, `csrc/vec.h` — header-only geometry and containers.
- `tests/` — unit tests, thread equivalence tests, and benchmarks in `tests/bench/`.
- `example/demo.py`, `example/visualize_sdf.py` — example workflow.

## Public API

`mesh2sdf.compute(vertices, faces, size=128, fix=False, level=0.015, return_mesh=False)`

- `vertices`: `float32`, shape `(N, 3)`, inside `[-1, 1]`. Out-of-range input gives a wrong SDF without an error.
- `faces`: `uint`, shape `(M, 3)`.
- Returns `sdf` as `float32` with shape `(size, size, size)`. If `return_mesh=True`, also returns a `trimesh.Trimesh`.

The grid spans `[-1, 1]` inclusive with `dx = 2/(size - 1)`. If `size < 2`, the function raises `ValueError`.

## Commands

```sh
just build       # uv sync + reinstall the C++ extension
just test        # unit tests
.venv/bin/python -m pytest tests/ -v   # all tests, includes thread equivalence
just check       # ruff + clang-format + clang-tidy
just bench       # benchmarks (disabled in normal test runs)
just run         # example/demo.py on example/data/plane.obj
```

The dev environment is `.venv` (Python 3.14, uv-managed).

## Rules

- Do not rename the `mesh2sdf/` directory or the `mesh2sdf.core` module. The rename breaks drop-in compatibility.
- Do not edit `LICENSE`. Put fork attribution in `pyproject.toml`.
- Keep the `version` in `pyproject.toml` as the only version literal. `setup.py` and the justfile read it.
- Write the C++ core in Orthodox C++: C headers, `malloc`/`free`, no allocating STL. Only `csrc/pybind.cpp` uses Modern C++.
- Do not add `-fno-exceptions`. The pybind boundary needs exceptions.
- Keep `check_neighbour`/`update_cell` duplicated next to each sweep. The duplication is intentional for locality.
- Parallel paths must stay bit-identical to the serial paths. `tests/test_sweep_equivalence.py` makes sure of this.
- If you change the `[-1, 1]` box convention, update `csrc/pybind.cpp`, `mesh2sdf/compute.py`, and `example/demo.py` together.
- Keep `MANIFEST.in` grafting `csrc/`. Without it, the sdist has no C++ sources.
- Do not commit generated `*.fixed.obj` or `*.npy` files.
