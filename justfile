# mesh2sdf dev tasks

set shell := ["bash", "-uc"]

# Python interpreter to use (override via PYTHON env var)
py := env_var_or_default("PYTHON", ".venv/bin/python")

# Project version (single source: [project] version in pyproject.toml)
version := `awk -F'"' '/^version/ {print $2; exit}' pyproject.toml`

# run the example workflow (default: example/data/plane.obj)
run mesh="example/data/plane.obj":
    {{py}} example/demo.py {{mesh}}

# build the project (sync deps + recompile the C++ extension)
build:
    uv sync
    uv pip install -e . --force-reinstall

# run pytest suite
test:
    {{py}} -m pytest tests/test_mesh2sdf.py -v

# run benchmark suite (skip by default in `test`; opt-in here)
bench name="":
    {{py}} -m pytest tests/bench --benchmark-enable --benchmark-min-rounds=3 --benchmark-columns=min,median,max,stddev,iterations --benchmark-sort=mean --benchmark-save={{name}} -v

# run benchmark suite and diff against the most recent saved run (fails on >10% median regression)
bench-compare old="" new="":
    pytest-benchmark compare {{old}} {{new}} --group-by=name --columns=min,mean,median,stddev

# run linting: ruff + clang-format + clang-tidy
check:
    #!/usr/bin/env bash
    set -euo pipefail

    "{{py}}" -m ruff check .
    find csrc -type f \( -name '*.cpp' -o -name '*.h' \) -exec clang-format --dry-run --Werror {} +

    pybind11_inc="$('{{py}}' -c 'import pybind11; print(pybind11.get_include())')"
    python_inc="$('{{py}}' -c 'import sysconfig; print(sysconfig.get_path("include"))')"

    clang-tidy csrc/pybind.cpp csrc/makelevelset3.cpp -- \
        -Icsrc \
        -I"$pybind11_inc" \
        -I"$python_inc" \
        -DVERSION_INFO="{{version}}" \
        -std=c++20

# remove build artifacts
clean:
    rm -rf build/ *.egg-info/ mesh2sdf.egg-info/ mesh2sdf2.egg-info/
    rm -f mesh2sdf/*.so

# initialize the project on a new computer
init:
    uv sync
