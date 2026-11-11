"""Compare the devices of mesh2sdf.compute: CPU with 1 and all threads, OpenCL, CUDA.

Each device runs in a fresh subprocess, so the memory pool of one GPU path
does not hold memory that the next one needs. A device without its extra or
its driver shows n/a. PYOPENCL_CTX selects the OpenCL device.

    python tests/bench/compare_devices.py [--repeats 3] [--rounds 5]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (name, mesh, size). The first four match compare_versions.py.
WORKLOADS = [
    ("cube 128", "cube", 128),
    ("plane 128", "plane", 128),
    ("icosphere 5k tris 128", 4, 128),
    ("icosphere 20k tris 128", 5, 128),
    ("plane 256", "plane", 256),
    ("plane 512", "plane", 512),
]

# (label, device, OMP_NUM_THREADS, multi_round). The 1-thread times vary little
# and take minutes at size 512, so they run one round.
DEVICES = [
    ("1 thread", "cpu", "1", False),
    (f"{os.cpu_count()} threads", "cpu", None, True),
    ("OpenCL", "opencl", None, True),
    ("CUDA", "cuda", None, True),
]


def worker(device: str, repeats: int) -> None:
    import numpy as np
    import trimesh

    import mesh2sdf

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

    out = {}
    for name, kind, size in WORKLOADS:
        v, f = mesh(kind)
        sdf = mesh2sdf.compute(v, f, size, device=device)
        if device != "cpu":
            assert np.array_equal(sdf, mesh2sdf.compute(v, f, size)), name
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mesh2sdf.compute(v, f, size, device=device)
            times.append((time.perf_counter() - t0) * 1e3)
        out[name] = statistics.median(times)
    print(json.dumps(out))


def measure(device: str, omp_threads: str | None, rounds: int, repeats: int):
    env = dict(os.environ)
    if omp_threads:
        env["OMP_NUM_THREADS"] = omp_threads
    cmd = [sys.executable, __file__, "--worker", device, "--repeats", str(repeats)]
    results = []
    for _ in range(rounds):
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            return None
        results.append(json.loads(run.stdout.splitlines()[-1]))
    return {name: statistics.median(r[name] for r in results) for name in results[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--worker")
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker, args.repeats)

    print(f"cpus: {os.cpu_count()}, python: {sys.version.split()[0]}")
    columns = {
        label: measure(device, threads, args.rounds if multi else 1, args.repeats)
        for label, device, threads, multi in DEVICES
    }
    width = max(len(name) for name, _, _ in WORKLOADS)
    print(f"\n=== median ms ===\n{'workload':<{width}}" + "".join(f"{c:>14}" for c in columns))
    for name, _, _ in WORKLOADS:
        cells = "".join(f"{res[name]:>14.1f}" if res else f"{'n/a':>14}" for res in columns.values())
        print(f"{name:<{width}}{cells}")


if __name__ == "__main__":
    main()
