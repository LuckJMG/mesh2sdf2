"""Compare the C++ core of mesh2sdf across commits: upstream, each perf step, HEAD.

Each commit is extracted with `git archive` into .benchmarks/versions/<sha>,
built once with `build_ext --inplace`, then timed in a fresh subprocess.

    python tests/bench/compare_versions.py [--repeats 3] [--rebuild]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".benchmarks" / "versions"

STAGES = [
    ("33f7177", "upstream mesh2sdf 1.1.0"),
    ("9b8e2c1", "+ cache triangle distance data"),
    ("b427937", "+ release GIL"),
    ("ab09b7c", "+ parallel sign application"),
    ("f242204", "+ wavefront parallel sweep"),
    ("3e9f840", "+ memcpy output grid"),
    ("9eed7e8", "+ parallel triangle init (2.1.0)"),
    ("badede4", "+ Orthodox C++ port"),
    ("HEAD", "HEAD (geometry/parallel split)"),
]

# (name, mesh, size). Icosphere radius is 0.9 so vertices stay inside [-1, 1].
WORKLOADS = [
    ("cube 64", "cube", 64),
    ("cube 128", "cube", 128),
    ("plane 128", "plane", 128),
    ("icosphere 5k tris 128", 4, 128),
    ("icosphere 20k tris 128", 5, 128),
]
GIL_WORKLOAD = "4 py threads x cube 128"
PY_THREADS = 4


def build(sha: str, rebuild: bool) -> Path:
    full = subprocess.check_output(
        ["git", "rev-parse", sha], cwd=ROOT, text=True
    ).strip()
    tree = CACHE / full[:7]
    if rebuild or not list(tree.glob("mesh2sdf/core*.so")):
        tree.mkdir(parents=True, exist_ok=True)
        archive = subprocess.Popen(
            ["git", "archive", full], cwd=ROOT, stdout=subprocess.PIPE
        )
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(tree, filter="data")
        subprocess.check_call(
            ["uv", "run", "--no-project", "--python", sys.executable,
             "--with", "setuptools", "--with", "pybind11",
             "python", "setup.py", "build_ext", "--inplace"],
            cwd=tree, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
    return tree


def worker(tree: str, repeats: int, gil: bool) -> None:
    import numpy as np
    import trimesh

    sys.path.insert(0, tree)
    from mesh2sdf import core

    assert core.__file__.startswith(tree), core.__file__

    def mesh(kind):
        if kind == "cube":
            m = trimesh.creation.box(extents=(1, 1, 1))
        elif kind == "plane":
            m = trimesh.load(ROOT / "example/data/plane.obj", force="mesh")
            lo, hi = m.vertices.min(0), m.vertices.max(0)
            m.vertices = (m.vertices - (lo + hi) / 2) * (1.6 / (hi - lo).max())
        else:
            m = trimesh.creation.icosphere(subdivisions=kind, radius=0.9)
        return m.vertices.astype(np.float32), m.faces.astype(np.uint32)

    def median_ms(fn, *args):
        fn(*args)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)

    out = {}
    for name, kind, size in WORKLOADS:
        v, f = mesh(kind)
        sdf = core.compute(v, f, size)
        assert sdf.shape == (size,) * 3 and sdf.dtype == np.float32
        out[name] = median_ms(core.compute, v, f, size)
    if gil:
        v, f = mesh("cube")

        def concurrent():
            with ThreadPoolExecutor(PY_THREADS) as pool:
                list(pool.map(lambda _: core.compute(v, f, 128), range(PY_THREADS)))

        out[GIL_WORKLOAD] = median_ms(concurrent)
    print(json.dumps(out))


def run_mode(
    title: str, omp_threads: str | None, gil: bool, trees, repeats: int
) -> None:
    env = dict(os.environ)
    if omp_threads:
        env["OMP_NUM_THREADS"] = omp_threads
    rows = []
    for (_, label), tree in zip(STAGES, trees):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            str(tree),
            "--repeats",
            str(repeats),
        ]
        if gil:
            cmd.append("--gil")
        rows.append(
            (label, json.loads(subprocess.check_output(cmd, env=env, text=True)))
        )

    cols = list(rows[0][1])
    width = max(len(label) for label, _ in rows)
    print(f"\n=== {title}: median ms (speedup vs upstream) ===")
    print(f"{'stage':<{width}}  " + "  ".join(f"{c:>26}" for c in cols))
    for label, res in rows:
        cells = [f"{res[c]:>12.1f} ({rows[0][1][c] / res[c]:>5.2f}x)" for c in cols]
        print(f"{label:<{width}}  " + "  ".join(f"{c:>26}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--gil", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker, args.repeats, args.gil)

    trees = [build(sha, args.rebuild) for sha, _ in STAGES]
    print(f"cpus: {os.cpu_count()}, python: {sys.version.split()[0]}")
    run_mode(f"all {os.cpu_count()} threads", None, False, trees, args.repeats)
    run_mode("1 OpenMP thread (serial gains only; last column shows GIL release)",
             "1", True, trees, args.repeats)  # fmt: skip


if __name__ == "__main__":
    main()
