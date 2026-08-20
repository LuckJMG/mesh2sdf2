# mesh2sdf

Python wrapper around pybind11 C++ ext. Compute signed distance field from
triangle mesh. Handle non-watertight input via marching cubes + biggest bbox
component pick.

## What it does

Algorithm:

1. Fast-sweep unsigned SDF on input mesh. Works on non-watertight, no sign
   accuracy loss.
2. If `fix=True`: take `abs(sdf)`, run marching cubes at `level`, get manifold
   surface, keep largest bbox component.
3. Re-run signed SDF on fixed mesh. Return signed result.

C++ fast-sweep from SDFGen by Christopher Batty. Box hard-coded `[-1, 1]` in
`csrc/pybind.cpp:27`. `dx = 2/size`.

## Layout

- `mesh2sdf/__init__.py` — re-export `compute`.
- `mesh2sdf/compute.py` — wrap `mesh2sdf.core.compute`. Run `fix=True` path.
- `csrc/pybind.cpp` — pybind11 module `core`. Function `compute(v, f, size)`.
- `csrc/makelevelset3.cpp` — fast-sweep algorithm impl.
- `csrc/makelevelset3.h` — `make_level_set3` decl.
- `csrc/array1.h`, `array2.h`, `array3.h` — 1/2/3D array types.
- `csrc/vec.h` — `Vec3f`, `Vec3ui` types.
- `csrc/util.h` — misc helpers.
- `csrc/main.backup.cpp` — legacy standalone. NOT in `ext_modules`. Ignore.
- `setup.py` — `Pybind11Extension('mesh2sdf.core', ['csrc/pybind.cpp', 'csrc/makelevelset3.cpp'])`.
- `pyproject.toml` — build-system only. No metadata.
- `MANIFEST.in` — graft `csrc/`. Needed for sdist.
- `example/test.py` — load OBJ, normalize, SDF, save `.fixed.obj` + `.npy`.
- `example/visualize_sdf.py` — slice `.npy`, render PNG + level-set OBJ.
- `example/data/plane.obj` — default test mesh.

## Build

Source install needs C++ compiler. pybind11 supported compilers:
https://github.com/pybind/pybind11#supported-compilers

```sh
pip install ./mesh2sdf
```

Build deps from `pyproject.toml`: `pybind11>=2.8.0`, `setuptools>=42`, `wheel`.

Runtime deps from `setup.py`: `numpy`, `trimesh`, `scikit-image`.

`uv sync` alone won't install runtime deps — `uv.lock` empty. Either
`uv pip install numpy trimesh scikit-image matplotlib` or
`pip install -e ./mesh2sdf`.

Python 3.12 pinned via `.python-version`. `.venv` uv-managed.

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
python example/test.py                    # use example/data/plane.obj
python example/test.py path/to/foo.obj    # custom; write foo.fixed.obj + foo.npy
python example/visualize_sdf.py           # use example/data/plane.npy
python example/visualize_sdf.py foo.npy   # custom .npy
```

`example/test.py` flow:

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

Smoke test = `python example/test.py` on `example/data/plane.obj`. Pass if
`<output>.fixed.obj` and `<output>.npy` produced without traceback.

Example also needs `matplotlib` (not in `install_requires`).

## Gotchas

- `pyproject.toml` no project metadata. Don't assume PEP 621 — read `setup.py`.
- `MANIFEST.in` grafts `csrc/`. Without it, sdist missing C++ sources, install fails.
- `compute.py:43` re-normalize fixed mesh to `[-1, 1]` after marching cubes. `example/test.py:32` un-scale for output. Keep both in sync if you touch box convention.
- `csrc/main.backup.cpp` has its own OBJ loader. Don't confuse with active code.
- `csrc/pybind.cpp:14` declare `vertices` as `float`, `faces` as `unsigned int`. Wrong dtype = pybind error or silent wrong SDF.
- `.gitignore` include `build/`, `*.so`, `__pycache__/`, `.venv/`, standard Python ignores.
- `LISCENCE` (typo, missing E). MIT. Don't rename — git history track it.
- Branch `master`.