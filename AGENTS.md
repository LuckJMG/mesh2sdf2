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
- `mesh2sdf/gpu.py`, `mesh2sdf/gpu.cu` — optional CUDA path for `device="cuda"`. CuPy compiles `gpu.cu` with NVRTC at import.
- `mesh2sdf/opencl.py`, `mesh2sdf/sweep.cl` — optional OpenCL path for `device="opencl"`. Init and signs run on the CPU (`mesh2sdf.core.init`, `mesh2sdf.core.apply_signs`). Only the sweeps run on the device.
- `mesh2sdf/sweeps.py` — sweep order and skip limits for both GPU paths.
- `tests/` — unit tests, thread equivalence tests, and GPU equivalence tests (`tests/test_gpu.py`, skipped without CuPy or a GPU, and `tests/test_opencl.py`, which runs on pocl without a GPU). Benchmarks are in `tests/bench/`.
- `docs/plots.py`, `docs/img/` — the performance charts of the README, in a light and a dark SVG each. After a new benchmark run, update the numbers in `docs/plots.py` and run it.
- `example/demo.py`, `example/visualize_sdf.py` — example workflow.

## Public API

`mesh2sdf.compute(vertices, faces, size=128, fix=False, level=0.015, return_mesh=False, device="cpu")`

- `vertices`: `float32`, shape `(N, 3)`, inside `[-1, 1]`. Out-of-range input gives a wrong SDF without an error.
- `faces`: `uint`, shape `(M, 3)`.
- `device`: `"cpu"`, `"cuda"`, or `"opencl"`. `"cuda"` needs the `gpu` extra (`cupy-cuda12x`). `"opencl"` needs the `opencl` extra (`pyopencl`), and `PYOPENCL_CTX` selects the device. Other values raise `ValueError`.
- Returns `sdf` as `float32` with shape `(size, size, size)`. If `return_mesh=True`, also returns a `trimesh.Trimesh`.

The grid spans `[-1, 1]` inclusive with `dx = 2/(size - 1)`. If `size < 2`, the function raises `ValueError`.

## Commands

```sh
just build       # uv sync --all-extras + reinstall the C++ extension
just test        # unit tests
.venv/bin/python -m pytest tests/ -v   # all tests, includes thread equivalence
just check       # ruff + clang-format + clang-tidy
just bench       # benchmarks (disabled in normal test runs)
just bench-devices  # CPU, OpenCL, and CUDA on the same workloads (about 15 min)
just run         # example/demo.py on example/data/plane.obj
```

The dev environment is `.venv` (Python 3.14, uv-managed).

## Release

1. Change `version` in `pyproject.toml`, then commit and push to `main`.
2. Create a GitHub release with a tag that is equal to the version, for example `2.5.0`.

`.github/workflows/release.yml` then makes sure that the tag is equal to the version, builds the sdist and the wheels (Linux x86_64 and aarch64, Windows AMD64, Python 3.10–3.14), tests each wheel, and uploads all files to PyPI with trusted publishing. A manual run of the workflow builds and tests the same files without an upload.

## Rules

- Do not rename the `mesh2sdf/` directory or the `mesh2sdf.core` module. The rename breaks drop-in compatibility.
- Do not edit `LICENSE`. Put fork attribution in `pyproject.toml`.
- Keep the `version` in `pyproject.toml` as the only version literal. `setup.py` and the justfile read it.
- Write the C++ core in Orthodox C++: C headers, `malloc`/`free`, no allocating STL. Only `csrc/pybind.cpp` uses Modern C++.
- Do not add `-fno-exceptions`. The pybind boundary needs exceptions.
- Keep `check_neighbour`/`update_cell` duplicated next to each sweep. The duplication is intentional for locality.
- Parallel paths must stay bit-identical to the serial paths. `tests/test_sweep_equivalence.py` makes sure of this.
- The GPU paths must stay bit-identical to the CPU path. `tests/test_gpu.py` and `tests/test_opencl.py` make sure of this. To test the OpenCL path on each device, run `tests/test_opencl.py` once for each `PYOPENCL_CTX` value.
- `mesh2sdf/gpu.cu` copies the geometry of `csrc/geometry.h` and `csrc/vec.h` one operation at a time. If you change one copy, change the other. One difference is intentional: `point_segment_distance` divides in float, not in double. The result is the same bit for bit (see the comment in `gpu.cu`).
- `mesh2sdf/sweep.cl` copies the float part of the geometry of `csrc/geometry.h`. If you change one copy, change the others.
- `opencl.py` refuses a device without correctly rounded float division and `sqrt`. It builds with `-cl-fp32-correctly-rounded-divide-sqrt`, and `sweep.cl` sets `FP_CONTRACT OFF`.
- `opencl.py` sends the plane number as the global work offset of dimension 2. In PyOpenCL, `set_arg` costs about 10 µs for each call, and a sweep at size 512 launches about 1500 planes.
- Do not let the compiler contract floating-point operations into FMA. `gpu.py` passes `--fmad=false` to NVRTC. Do not add `-march=native` or `-ffp-contract=fast` to `setup.py`.
- The GPU sweeps run one diagonal plane per kernel launch. A cell reads only cells on earlier planes, so the result is the same as the serial sweep.
- Each GPU sweep uses its own skewed layout, so that the reads of one plane coalesce. The `relayout` kernel changes the layout between sweeps. If two sweeps differ in `di` but not in `dj`, `relayout_flip_i` does the change through shared-memory tiles, because a plain gather does not coalesce there. Init writes the layout of the first sweep directly.
- The GPU sweep skips checks for each cell, not for each row as the CPU does. When a GPU cell runs, its source rows are not complete, so the row skip of the CPU cannot work there. `last_change` holds one byte per cell, in the same layout as `phi`. `LIMITS` in `mesh2sdf/sweeps.py` holds, for each sweep and each of the 7 checks, the last earlier sweep that ran the same check.
- `csrc/pybind.cpp` owns the `[-1, 1]` box and `dx`. `compute.py` reads them from `mesh2sdf.core` (`box_min`, `grid_dx`). If you change the box, update `csrc/pybind.cpp` and `example/demo.py` together.
- Allocate each grid with `alloc_grid()` from `csrc/array3.h` and release it with `free()`. On Linux it asks for huge pages, which removes about 130k page faults per array at `size=512`.
- Use a signed index type in each `omp for` loop. MSVC supports only OpenMP 2.0, which rejects an unsigned index such as `size_t`.
- Keep `MANIFEST.in` grafting `csrc/`. Without it, the sdist has no C++ sources.
- Do not commit generated `*.fixed.obj` or `*.npy` files.
