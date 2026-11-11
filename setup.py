import re
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# single source of truth: [project] version in pyproject.toml
__version__ = re.search(
    r'^version = "(.*?)"', Path("pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE
).group(1)

# -ffp-contract=off: GCC fuses a*b - c*d into one FMA wherever the target has FMA
# (aarch64, -march=native). The exact sign test in orientation() then breaks, and
# the SDF gets the wrong sign. MSVC does not contract without /fp:contract.
extra_compile_args = (
	["/openmp", "/std:c++20"] if sys.platform == "win32" else ["-fopenmp", "-std=c++20", "-ffp-contract=off"]
)
extra_link_args = [] if sys.platform == "win32" else ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "mesh2sdf.core",
        ["csrc/pybind.cpp", "csrc/makelevelset3.cpp", "csrc/parallel.cpp"],
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
