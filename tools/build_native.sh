#!/usr/bin/env bash
# Configure and build the native tree, including the pybind11 module,
# against this project's virtualenv.
#
#   tools/build_native.sh              # build everything
#   tools/build_native.sh --sanitize   # ASan + UBSan (implies --no-codec)
#   tools/build_native.sh --no-codec   # skip the codec and its onnxruntime download
#   tools/build_native.sh --test       # build, then run ctest and pytest --native
#
# The codec is the only part that downloads anything (a pinned
# onnxruntime binary, 9-80 MB depending on platform). --no-codec gives
# an offline build of everything else, which is the entire modem.
#
# **--no-codec is sticky, and dropping the flag does not undo it.** It
# writes SSTVAE_BUILD_CODEC=OFF into the CMake cache, and a later plain
# `tools/build_native.sh` in the same build directory says nothing and
# keeps it off -- which also silently turns off the GUI, since
# SSTVAE_BUILD_GUI is AUTO and the GUI needs the codec. The symptom is
# ctest quietly reporting 19 passing tests instead of 29, with the ten
# widget tests never registered rather than failing: the same shape of
# green-but-testing-nothing that SSTVAE_REQUIRE_CODEC exists to catch
# elsewhere. To go back, say so: `cmake -DSSTVAE_BUILD_CODEC=ON .` in
# the build directory, or delete it.
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
        # onnxruntime is a prebuilt binary nobody here instrumented, so
        # under ASan it would report its allocations rather than ours.
        --sanitize) cmake_args+=(-DSSTVAE_SANITIZE=ON -DSSTVAE_BUILD_CODEC=OFF) ;;
        --no-codec) cmake_args+=(-DSSTVAE_BUILD_CODEC=OFF) ;;
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
