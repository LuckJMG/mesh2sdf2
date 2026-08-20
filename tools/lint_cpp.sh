#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${VENV:-.venv}"
VENV_PYBIND11_INCLUDE="$(
  find "$VENV/lib" -path '*/site-packages/pybind11/include' -print -quit 2>/dev/null || true
)"
PYTHON_INCLUDE="$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")"

echo "==> clang-format"
find csrc -type f \( -name '*.cpp' -o -name '*.h' \) -print0 |
  xargs -0 clang-format --dry-run --Werror

echo "==> compiledb"
uv pip install -e . --force-reinstall --verbose 2>&1 |
  "${VENV}/bin/compiledb" -f > /dev/null

# uv's isolated PEP 517 builds reference pybind11 from a transient cache
# (e.g. ~/.cache/uv/builds-v0/.tmpXXXX/...). Rewrite those paths to the
# stable venv location so clang-tidy can resolve them after the build dir
# is gone.
if [[ -n "${VENV_PYBIND11_INCLUDE}" ]]; then
  python3 - "$VENV_PYBIND11_INCLUDE" "$PYTHON_INCLUDE" <<'PY'
import json, sys, re
pybind11_inc, python_inc = sys.argv[1], sys.argv[2]
with open("compile_commands.json") as f:
    data = json.load(f)
for entry in data:
    args = entry["arguments"]
    new_args = []
    for arg in args:
        if re.match(r"^-I.*/site-packages/pybind11/include$", arg):
            new_args.append("-I" + pybind11_inc)
        else:
            new_args.append(arg)
    entry["arguments"] = new_args
with open("compile_commands.json", "w") as f:
    json.dump(data, f, indent=1)
PY
fi

echo "==> clang-tidy"
clang-tidy csrc/pybind.cpp csrc/makelevelset3.cpp
