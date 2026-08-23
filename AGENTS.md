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

C++ fast-sweep from SDFGen by Christopher Batty. Box hard-coded `[-1, 1]` in
`csrc/pybind.cpp:29`. `dx = 2/size`. The algorithm matches upstream SDFGen,
but the code diverged from upstream and no longer diffs cleanly against it:
`csrc/makelevelset3.cpp` splits into per-phase static helpers, and vendored
headers (`array1.h`, `vec.h`) keep only the surface the build uses.

## Layout

- `mesh2sdf/__init__.py` — re-export `compute`. Import name `mesh2sdf`
  preserved for drop-in use.
- `mesh2sdf/compute.py` — wrap `mesh2sdf.core.compute`. Run `fix=True` path.
- `csrc/pybind.cpp` — pybind11 module `core`. Function `compute(v, f, size)`.
- `csrc/makelevelset3.cpp` — fast-sweep algorithm impl. The sweep has two
  paths: legacy sequential, and a wavefront-parallel variant (cells grouped
  by `di*i+dj*j+dk*k`, one `omp parallel for` per level). Dispatch picks the
  parallel path only with OpenMP, >1 threads, and grids ≥ 2^18 cells.
  Level order is a topological order of the same dependency DAG as the
  sequential walk: results are bit-identical at any thread count.
  `OMP_NUM_THREADS=1` forces the legacy path. Guarded by
  `tests/test_sweep_equivalence.py`.
- `csrc/makelevelset3.h` — `make_level_set3` decl.
- `csrc/array1.h`, `array3.h` — 1/3D array types.
- `csrc/vec.h` — `Vec3f`, `Vec3ui` types.
- `csrc/util.h` — misc helpers.
- `setup.py` — minimal; declares `Pybind11Extension('mesh2sdf.core', [...])`
  + `cmdclass={'build_ext': build_ext}`. Adds OpenMP flags (`-fopenmp`,
  `/openmp` on MSVC). Derives `__version__` from
  `pyproject.toml` and passes it to the C++ `VERSION_INFO` macro.
- `pyproject.toml` — PEP 621 metadata + build-system. `[project]`
  declares name `mesh2sdf2`, `version` (single source of truth for the
  whole project), deps, classifiers, urls. `[build-system]`
  requires `setuptools>=68, pybind11>=2.13`. `[dependency-groups].dev`
  has `matplotlib` (visualization only).
- `MANIFEST.in` — graft `csrc/`. Needed for sdist.
- `LICENSE` — MIT verbatim. Renamed from `LISCENCE` typo in v2.
- `example/demo.py` — load OBJ, normalize, SDF, save `.fixed.obj` + `.npy`.
- `example/visualize_sdf.py` — slice `.npy`, render PNG + level-set OBJ.
- `example/data/plane.obj` — default test mesh.

## Build

Source install needs a C++ compiler. pybind11 supported compilers:
https://github.com/pybind/pybind11#supported-compilers

```sh
pip install mesh2sdf2            # PyPI (fork name)
pip install -e ./mesh2sdf2       # source, editable
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

No test suite. No CI. No linter, formatter, typecheck configured.

## C++ lint

`csrc/**` is formatted with `clang-format` (LLVM base, 4-tab indent, K&R
braces, `InsertBraces: true`, `BeforeElse`/`BeforeCatch` on new line) and
analyzed with `clang-tidy` (conservative checks: `clang-analyzer-*`,
`bugprone-*`, `performance-*`). Config files are `.clang-format` and
`.clang-tidy` at the repo root. Run everything with `just check`. The
justfile reads the pybind11 and Python include paths from the venv and
passes them directly to clang-tidy; no `compile_commands.json` is
generated.

Smoke test (Python 3.14, single `.venv`):

```sh
.venv/bin/python example/demo.py
```

Pass if `<output>.fixed.obj` and `<output>.npy` produced without traceback.

Version sanity:

```sh
.venv/bin/python -c "import mesh2sdf.core; print(mesh2sdf.core.__version__)"   # 2.0.0
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
  install fails. setuptools auto-includes `LICENSE` at root after the v2 rename.
- `compute.py:43` re-normalizes fixed mesh to `[-1, 1]` after marching
  cubes. `example/demo.py:32` un-scales for output. Keep both in sync if
  you touch box convention.
- `csrc/pybind.cpp:14` declares `vertices` as `float`, `faces` as
  `unsigned int`. Wrong dtype = pybind error or silent wrong SDF.
- `.gitignore` includes `build/`, `*.so`, `__pycache__/`, `.venv/`,
  standard Python ignores. Smoke-test outputs
  (`example/data/*.fixed.obj`, `*.npy`) are untracked — do not commit.
