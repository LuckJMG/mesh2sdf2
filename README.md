# Mesh2SDF

[![Downloads](https://static.pepy.tech/badge/mesh2sdf2)](https://pepy.tech/project/mesh2sdf2)
[![Downloads](https://static.pepy.tech/badge/mesh2sdf2/month)](https://pepy.tech/project/mesh2sdf2/month)
[![PyPI](https://img.shields.io/pypi/v/mesh2sdf2)](https://pypi.org/project/mesh2sdf2/)


> **This is a community-maintained fork** of
> [mesh2sdf](https://github.com/wang-ps/mesh2sdf) by Peng-Shuai Wang. It adds
> Python 3.10–3.14 support and modernizes the build (setuptools ≥68,
> pybind11 ≥2.13). Original project, paper, and algorithm: see the
> [upstream repo](https://github.com/wang-ps/mesh2sdf). License: MIT.

Converts an input mesh to a signed distance field. It can work with arbitrary
meshes, even **non-watertight** meshes from ShapeNet.

`mesh2sdf` is used in our paper
[Dual Octree Graph Networks (SIGGRAPH 2022)](https://wang-ps.github.io/dualocnn)
to generate the training data.
Please cite our paper if you find the code useful for your research.


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
    pip install ./mesh2sdf2
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