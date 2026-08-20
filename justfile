# mesh2sdf dev tasks

set shell := ["bash", "-uc"]

# Python interpreter to use (override via PYTHON env var)
py := env_var_or_default("PYTHON", ".venv/bin/python")

# run the example workflow (default: example/data/plane.obj)
run mesh="example/data/plane.obj":
    {{py}} example/test.py {{mesh}}

# build the project (sync deps + recompile the C++ extension)
build:
    uv sync
    uv pip install -e . --force-reinstall

# run pytest suite
test:
    {{py}} -m pytest tests/test_mesh2sdf.py -v

# run linting: ruff + C++ lint
check:
    {{py}} -m ruff check .
    bash tools/lint_cpp.sh

# remove build artifacts and Python caches
clean:
    rm -rf build/ *.egg-info/ mesh2sdf.egg-info/ mesh2sdf2.egg-info/
    rm -f mesh2sdf/*.so
    find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
    find . -path ./.venv -prune -o -type f -name '*.pyc' -delete
    rm -f compile_commands.json

# initialize the project on a new computer
init:
    uv sync
