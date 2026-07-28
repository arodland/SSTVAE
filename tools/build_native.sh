#!/usr/bin/env bash
# Configure and build the native tree, including the pybind11 module,
# against this project's virtualenv.
#
#   tools/build_native.sh              # build everything
#   tools/build_native.sh --sanitize   # ASan + UBSan
#   tools/build_native.sh --test       # build, then run ctest and pytest --native
#
# The interpreter matters: the extension module must be built for the
# same Python that runs pytest, or the import silently fails and the
# parity run skips instead of failing.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/native/build"

python_exe="${PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python_exe" ]]; then
    python_exe="$(command -v python3)"
    echo "note: no .venv found, using $python_exe" >&2
fi

cmake_args=(-S "$root/native" -B "$build" -G Ninja
            "-DPython3_EXECUTABLE=$python_exe")

# pybind11 is usually a system package while the interpreter is the
# venv's, so its CMake package has to be located explicitly. Ask
# whichever interpreter actually has it.
if ! pybind11_dir="$("$python_exe" -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)"; then
    pybind11_dir="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || true)"
fi
if [[ -n "${pybind11_dir:-}" ]]; then
    cmake_args+=("-Dpybind11_DIR=$pybind11_dir")
else
    echo "warning: pybind11 not found; the parity module will not build" >&2
fi

run_tests=0
for arg in "$@"; do
    case "$arg" in
        --sanitize) cmake_args+=(-DSSTVAE_SANITIZE=ON) ;;
        --test)     run_tests=1 ;;
        *)          cmake_args+=("$arg") ;;
    esac
done

cmake "${cmake_args[@]}"
cmake --build "$build"

if (( run_tests )); then
    ctest --test-dir "$build" --output-on-failure
    "$python_exe" -m pytest "$root/tests" --native -q
fi
