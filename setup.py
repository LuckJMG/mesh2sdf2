import re
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# single source of truth: [project] version in pyproject.toml
__version__ = re.search(
    r'^version = "(.*?)"', Path("pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE
).group(1)

extra_compile_args = ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]
extra_link_args = [] if sys.platform == "win32" else ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "mesh2sdf.core",
        ["csrc/pybind.cpp", "csrc/makelevelset3.cpp"],
        include_dirs=["csrc"],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
