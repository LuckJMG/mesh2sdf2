from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "2.0.0"

ext_modules = [
    Pybind11Extension(
        "mesh2sdf.core",
        ["csrc/pybind.cpp", "csrc/makelevelset3.cpp"],
        include_dirs=["csrc"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
