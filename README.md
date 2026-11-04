# Mesh2SDF2

[![Downloads](https://static.pepy.tech/badge/mesh2sdf2)](https://pepy.tech/project/mesh2sdf2)
[![Downloads](https://static.pepy.tech/badge/mesh2sdf2/month)](https://pepy.tech/project/mesh2sdf2/month)
[![PyPI](https://img.shields.io/pypi/v/mesh2sdf2)](https://pypi.org/project/mesh2sdf2/)


> **This is a community-maintained fork** of
> [mesh2sdf](https://github.com/wang-ps/mesh2sdf) by Peng-Shuai Wang. It adds
> Python 3.10–3.14 support, modernizes the build (setuptools ≥68,
> pybind11 ≥2.13), and improve the compute speed. Original project, paper, and algorithm: see the
> [upstream repo](https://github.com/wang-ps/mesh2sdf). License: MIT.

Converts an input mesh to a signed distance field. It can work with arbitrary
meshes, even **non-watertight** meshes from ShapeNet.

`mesh2sdf` is used in their paper
[Dual Octree Graph Networks (SIGGRAPH 2022)](https://wang-ps.github.io/dualocnn)
to generate the training data.
Please cite their paper if you find the code useful for your research.


## Installation

`mesh2sdf2` depends on [pybind11](https://github.com/pybind/pybind11), and C++
compilers are needed to build the code. Supported compilers are listed
[here](https://github.com/pybind/pybind11#supported-compilers).

- Install via the following command:
    ``` shell
    pip install mesh2sdf2
    ```

- Alternatively, install from the source code via the following commands.
    ``` shell
    git clone https://github.com/LuckJMG/mesh2sdf2.git
    cd mesh2sdf2
    pip install .
    ```

- Alternatively, install from the source code via the following commands.
    ``` shell
    git clone https://github.com/wang-ps/mesh2sdf.git
    pip install ./mesh2sdf
    ```

## Example

After installing `mesh2sdf`, run the following command to process an input mesh
from ShapeNet:

```shell
python example/demo.py
```

![Example of a mesh from ShapeNet](https://raw.githubusercontent.com/wang-ps/mesh2sdf/master/example/data/result.png)


## Performance

`mesh2sdf2` 2.2.0 computes the SDF about **6x faster** than upstream
`mesh2sdf` 1.1.0 on a 12-thread CPU. On one thread, it is about 1.8x faster.
Both numbers are the geometric mean over five workloads (the four below plus a `size=64` cube; median of 3 runs, Python 3.14).

| Workload | upstream | 2.2.0, 12 threads | 2.2.0, 1 thread |
|---|---|---|---|
| cube | 6066 ms | 1077 ms (5.6x) | 3163 ms (1.9x) |
| plane | 7245 ms | 1155 ms (6.3x) | 4007 ms (1.8x) |
| icosphere, 5k triangles | 7442 ms | 1200 ms (6.2x) | 4312 ms (1.7x) |
| icosphere, 20k triangles | 9040 ms | 1409 ms (6.4x) | 5774 ms (1.6x) |

Gain from each improvement (cube, `size=128`, 12 threads, cumulative):

| Stage | Time | Speedup |
|---|---|---|
| upstream 1.1.0 | 6066 ms | 1.00x |
| cache triangle distance data | 4011 ms | 1.51x |
| wavefront parallel sweep | 1321 ms | 4.59x |
| Orthodox C++ port | 1167 ms | 5.20x |
| geometry and parallel split (2.2.0) | 1077 ms | 5.63x |

The C++ core releases the GIL. With `OMP_NUM_THREADS=1`, four Python threads
that each call `compute` on the cube finish 7.6x faster than upstream
(24.3 s to 3.2 s). The GIL release alone gives 4x of that. The parallel sign, memcpy, and parallel init steps
give no measurable gain on these meshes. The sweep takes almost all the time.

To reproduce the numbers, run `just bench-versions`. The run takes about 30 minutes.
The speedup depends on the number of CPU cores.


## How does it work?

- Given an input mesh, we first compute the **unsigned** distance field with the
  fast sweeping algorithm implemented by
  [Christopher Batty (SDFGen)](https://github.com/christopherbatty/SDFGen).
  Note that the unsigned distance field can always be reliably and accurately
  computed even though the input mesh is non-watertight.

- Then we extract the level sets with a small value **d** with the marching cube
  algorithm. The extracted level sets are represented with triangle meshes and
  are guaranteed to be manifold.

- There exist multiple connected components in the extracted meshes, and we only
  keep the mesh with the largest bounding box.

- Compute the signed distance field again with the kept triangle mesh as the
  final output. In this way, the signed distance field (SDF) is computed for a
  non-watertight input mesh.


## Citation

```
@article {Wang-Sig2022,
  title      = {Dual Octree Graph Networks for Learning Adaptive Volumetric
                Shape Representations},
  author     = {Wang, Peng-Shuai and Liu, Yang and Tong, Xin},
  journal    = {ACM Transactions on Graphics (SIGGRAPH)},
  volume     = {41},
  number     = {4},
  year       = {2022},
}

```
