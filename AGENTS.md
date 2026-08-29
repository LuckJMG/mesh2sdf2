# mesh2sdf2

Community-maintained fork of
[mesh2sdf](https://github.com/wang-ps/mesh2sdf) by Peng-Shuai Wang. Same
algorithm and public API, modernized build: Python 3.10–3.14, setuptools
≥68, pybind11 ≥2.13. License: MIT (verbatim from upstream).

Distribution name on PyPI is `mesh2sdf2`. Import name stays `mesh2sdf`
for drop-in compatibility. Maintainer: Lucas Mosquera. Upstream credit
in `pyproject.toml` `authors` + `[project.urls].Upstream`.

## What it does

Algorithm:

1. Fast-sweep unsigned SDF on input mesh. Works on non-watertight, no sign
   accuracy loss.
2. If `fix=True`: take `abs(sdf)`, run marching cubes at `level`, get manifold
   surface, keep largest bbox component.
3. Re-run signed SDF on fixed mesh. Return signed result.

C++ fast-sweep from SDFGen by Christopher Batty, now Orthodox C++ on C++20.
C headers (`<assert.h>`, `<math.h>`, `<string.h>`, `<stdint.h>`, `<stdlib.h>`),
manual `malloc`/`free`, no allocating STL. `pybind11` island stays Modern C++.
`Vec3` keeps `operator+/-/*` by math exception. Box hard-coded `[-1, 1]` in
`csrc/pybind.cpp:30`. `dx = 2/(size - 1)`, so `size` samples span `[-1, 1]`
inclusive; `size < 2` raises `ValueError`. The algorithm matches upstream SDFGen,
but the code diverged: `csrc/makelevelset3.cpp` splits into per-phase static
helpers, `csrc/array3.h` is `malloc`/`free`-backed `Array3<T>` (`T* a`, 129
lines, `NOLINT` on `operator()`), `csrc/vec.h` keeps `Vec3f`/`Vec3ui` with C
headers but retained overloads (63 lines), and `csrc/util.h` was removed.

## Layout

- `mesh2sdf/__init__.py` — re-export `compute`. Import name `mesh2sdf`
  preserved for drop-in use.
- `mesh2sdf/compute.py` — wrap `mesh2sdf.core.compute`. Run `fix=True` path.
- `csrc/pybind.cpp` — pybind11 module `core`. Function `compute(v, f, size)`.
  Releases GIL around `make_level_set3`, copies result with `memcpy`.
- `csrc/makelevelset3.cpp` — fast-sweep algorithm impl. The sweep has two
  parallel paths:
  - Init: `init_distances_and_counts_parallel` with packed 64-bit CAS
    `((bits<<32)|t)` for dense meshes (`≥4096` tris, `1<<12`). Serial init
    otherwise. `omp atomic` on `intersection_count`.
  - Sweep: wavefront-parallel variant (cells grouped by `di*i+dj*j+dk*k`,
    one `omp parallel for` per level) for grids `≥2^18` cells. Dispatch picks
    the parallel path only with OpenMP, >1 threads, and the size threshold.
  Level order is a topological order of the same dependency DAG as the
  sequential walk: results are bit-identical at any thread count.
  `OMP_NUM_THREADS=1` forces the legacy paths. Guarded by
  `tests/test_sweep_equivalence.py`. Sign pass is `omp parallel for collapse(2)`.
- `csrc/makelevelset3.h` — `make_level_set3` decl. Two overloads:
  `std::vector` bridge for `pybind.cpp` + orthodox raw-pointer core
  (`Vec3ui* tri, int ntri, Vec3f* x`).
- `csrc/array3.h` — `Array3<T>` over `malloc`/`free` (129 lines, `T* a`,
  `index(i,j,k)=i+ni*(j+nj*k)`, `NOLINT` on `operator()` for analyzer).
- `csrc/vec.h` — `Vec3f{float x,y,z}`, `Vec3ui{uint x,y,z}` with `dot`,
  `mag2`, `dist` (63 lines, `operator[]` via `(&x)[i]`, C headers
  `<assert.h>/<math.h>` with operator overloads kept).
- `setup.py` — minimal; declares `Pybind11Extension('mesh2sdf.core', [...])`
  + `cmdclass={'build_ext': build_ext}`. Adds OpenMP flags (`-fopenmp`,
  `/openmp` on MSVC) and `-std=c++20` (`/std:c++20` on MSVC). Derives
  `__version__` from `pyproject.toml` and passes it to `VERSION_INFO`.
- `pyproject.toml` — PEP 621 metadata + build-system. `[project]`
  declares name `mesh2sdf2`, `version` (single source of truth for the
  whole project), deps, classifiers, urls. `[build-system]`
  requires `setuptools>=68, pybind11>=2.13`. `[dependency-groups].dev`
  has `matplotlib`, `pybind11>=2.13`, `pytest`, `pytest-benchmark`,
  `ruff>=0.16.4`.
- `MANIFEST.in` — graft `csrc/`. Needed for sdist.
- `LICENSE` — MIT verbatim.
- `example/demo.py` — load OBJ, normalize, SDF, save `.fixed.obj` + `.npy`.
- `example/visualize_sdf.py` — slice `.npy`, render PNG + level-set OBJ.
- `example/data/plane.obj` — default test mesh (`result.png`/`result.svg`
  are rendered outputs, not inputs).
- `tests/test_mesh2sdf.py` — unit tests (output contract, resolution,
  correctness, grid alignment `[-1,1]` inclusive, fix path, plane e2e).
- `tests/test_sweep_equivalence.py` — bit-identical sweep/init at 1/2/5
  threads (`SIZE=96`, covers both parallel paths).
- `tests/bench/` — pytest-benchmark suites (`test_core.py`, `test_e2e.py`,
  `conftest.py` session fixtures).
- `justfile` — dev tasks (`run`, `build`, `test`, `bench`, `bench-compare`,
  `check`, `clean`, `init`).
- `.clang-format`, `.clang-tidy` — C++ lint config.

## Build

Source install needs a C++ compiler. pybind11 supported compilers:
https://github.com/pybind/pybind11#supported-compilers

```sh
pip install mesh2sdf2            # PyPI (fork name)
pip install -e .                 # source, editable (from repo root)
```

Build deps from `pyproject.toml [build-system].requires`:
`setuptools>=68`, `pybind11>=2.13`. `wheel` removed (setuptools
auto-provides).

Runtime deps from `pyproject.toml [project.dependencies]`:
`numpy`, `trimesh`, `scikit-image`.

`uv sync` resolves build + runtime + dev via `uv.lock` (populated since v2).

The project has no Python pin in the working tree. The single `.venv` is
the dev environment and is currently Python 3.14 (uv-managed).

Local dev setup, if `.venv` is missing or built against the wrong Python:

```sh
uv python install 3.14
uv venv --python "$(uv python find 3.14 | grep -v '/usr/bin')" .venv
uv sync --no-dev
# or: just init  (uv sync)  /  just build  (uv sync + force-reinstall)
```

## Public API

`mesh2sdf.compute(vertices, faces, size=128, fix=False, level=0.015, return_mesh=False)`

Inputs:

- `vertices`: `np.ndarray` shape `(N, 3)`, dtype `float32`. MUST lie in `[-1, 1]`. Out-of-range input produce wrong SDF silently.
- `faces`: `np.ndarray` shape `(M, 3)`, dtype `uint`. Triangle indices.
- `size`: int. Output grid resolution.
- `fix`: bool. Run marching-cubes + largest-component fix pass.
- `level`: float. Level-set value for marching cubes. Recommended `2/size`. Default `0.015` ≈ `2/128`.
- `return_mesh`: bool. Return fixed mesh too.

Returns:

- `sdf`: `np.ndarray` shape `(size, size, size)`, dtype `float32`.
- mesh: `trimesh.Trimesh`. Only if `return_mesh=True`.

## Example workflow

```sh
python example/demo.py                    # use example/data/plane.obj
python example/demo.py path/to/foo.obj    # custom; write foo.fixed.obj + foo.npy
python example/visualize_sdf.py           # use example/data/plane.npy
python example/visualize_sdf.py foo.npy   # custom .npy
# or: just run / just run path/to/foo.obj
```

`example/demo.py` flow:

1. Load OBJ via `trimesh.load(force='mesh')`.
2. Normalize vertices to `[-1, 1]` with `mesh_scale=0.8` (10% padding each side).
3. Call `mesh2sdf.compute(..., fix=True, level=2/size, return_mesh=True)`.
4. Un-scale fixed mesh, export `<name>.fixed.obj`. Save `sdf` as `<name>.npy`.

`example/visualize_sdf.py` flow:

1. Load `.npy`.
2. For each `level` in `[-0.02, 0.0, 0.02]`: marching cubes, export OBJ.
3. For each slice `i` in `range(size)`: matplotlib contourf PNG.

## Verification

Tests: `pytest`. Unit `tests/test_mesh2sdf.py` (output contract, resolution,
correctness, grid alignment `[-1,1]` inclusive, fix path, plane e2e).
Equivalence `tests/test_sweep_equivalence.py` (bit-identical at 1/2/5 threads,
`SIZE=96`, covers both init and sweep parallel paths). Fixtures in
`tests/conftest.py` and `tests/bench/conftest.py` (session-scoped where
possible). Bench `tests/bench/` disabled by default via `addopts`.

```sh
just test        # pytest tests/test_mesh2sdf.py -v
.venv/bin/python -m pytest tests/test_sweep_equivalence.py -v
.venv/bin/python -m pytest tests/ -v
```

## C++ lint

`csrc/**` is formatted with `clang-format` (LLVM base, 4-tab indent, K&R
braces, `InsertBraces: true`, `BeforeElse`/`BeforeCatch` on new line) and
analyzed with `clang-tidy` (conservative checks: `clang-analyzer-*`,
`bugprone-*` except `bugprone-easily-swappable-parameters`,
`performance-*`). Config files are `.clang-format` and
`.clang-tidy` at the repo root. Run everything with `just check`:

```sh
just check       # ruff check + clang-format --dry-run --Werror + clang-tidy
```

`just check` reads the pybind11 and Python include paths from the venv and
passes them directly to clang-tidy (`-Icsrc -I$pybind11_inc -I$python_inc
-DVERSION_INFO=<version from pyproject.toml> -std=c++20`); no
`compile_commands.json` is generated. Orthodox core is `NOLINT`-suppressed
for `Array3::operator()` false positive (see `csrc/array3.h:69`); otherwise
`just check` is green.

Smoke test (Python 3.14, single `.venv`):

```sh
.venv/bin/python example/demo.py
just run
```

Pass if `<output>.fixed.obj` and `<output>.npy` produced without traceback.

Version sanity:

```sh
.venv/bin/python -c "import mesh2sdf.core; print(mesh2sdf.core.__version__)"   # 2.1.1
```

Visualization needs `matplotlib` (in `[dependency-groups].dev`):

```sh
uv sync                                       # installs dev group
uv run python example/visualize_sdf.py
```

## Benchmarking

Pytest-driven benchmarks live in `tests/bench/` and use `pytest-benchmark`.
Default `addopts` in `pyproject.toml` sets `--benchmark-disable`, so the
unit suite (`just test`) does not run benchmarks.

Run the full benchmark suite:

```sh
just bench
just bench my_run          # --benchmark-save=my_run
just bench-compare         # diff against last saved runs, fails on >10% median regression
```

This invokes `pytest tests/bench --benchmark-enable ...` and prints a
comparison table with `min`, `median`, `max`, `stddev`, `iterations`.

Scenarios:

| File | Scope | What it covers |
|---|---|---|
| `tests/bench/test_core.py` | `mesh2sdf.core.compute` (C++ fast-sweep) | Cube @ {32,64,128}, plane @ {64,128}, icosphere subdivisions {2,3,4} @ size=128 |
| `tests/bench/test_e2e.py` | `mesh2sdf.compute` (full wrapper) | Plane `fix=True` and `fix=False` @ {64,128}, icosphere `fix=True` @ {64,128} |

Fixtures in `tests/bench/conftest.py` are session-scoped where possible.
Numbers are hardware-specific; treat them as a local baseline, not a
target.

## Gotchas

- **This is a fork.** Distribution is `mesh2sdf2`, but `import mesh2sdf`
  works. Do not rename the `mesh2sdf/` directory or the `mesh2sdf.core`
  C++ module — that would break drop-in compatibility.
- **Upstream** is at github.com/wang-ps/mesh2sdf. Upstream copyright
  (Peng-Shuai Wang, 2022) is preserved verbatim in `LICENSE`. Fork
  attribution goes via `pyproject.toml` `authors` and `[project.urls]`.
  Do not edit `LICENSE` text.
- `pyproject.toml` has full PEP 621 metadata (ported from `setup.py` in v2).
  All package metadata lives in `pyproject.toml`; `setup.py` only declares
  the extension and `cmdclass`.
- `pyproject.toml` `version` is the single source of truth. `setup.py`
  derives `__version__` from it at build time, and the justfile parses it
  for clang-tidy's `-DVERSION_INFO`. Never duplicate the literal elsewhere.
- `MANIFEST.in` grafts `csrc/`. Without it, sdist missing C++ sources,
  install fails.
- `compute.py:43` re-normalizes fixed mesh to `[-1, 1]` after marching
  cubes. `example/demo.py:32` un-scales for output. Keep both in sync if
  you touch box convention.
- `csrc/pybind.cpp:14` declares `vertices` as `float`, `faces` as
  `unsigned int`. Wrong dtype = pybind error or silent wrong SDF.
- Orthodox C++: `csrc/pybind.cpp` is the unorthodox island (Modern C++,
  `pybind11`, exceptions). Do not add `-fno-exceptions` globally — it would
  break the pybind boundary. Core `csrc/makelevelset3.cpp` / `array3.h` /
  `vec.h` are orthodox (`-std=c++20`, C headers, `malloc`/`free`).
- `.gitignore` is minimal (12 lines): `*.so`, `build/`, `dist/`,
  `*.egg-info/`, `__pycache__/`, `*.py[cod]`, `.venv/`, `.benchmarks/`,
  `.pytest_cache/`, `.ruff_cache/`, `compile_commands.json`, `.DS_Store`.
  Generated `*.fixed.obj`/`*.npy` are untracked by default — do not commit.
