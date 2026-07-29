# The pybind11 module that lets pytest drive the C++ core.
#
# This must be built against the *same* interpreter that runs pytest --
# usually the project's .venv, not the system python. Point CMake at it
# explicitly:
#
#   cmake -S native -B native/build -G Ninja \
#         -DPython3_EXECUTABLE=$PWD/.venv/bin/python
#
# tools/build_native.sh does this for you.

# Development.Module (not full Development) is the right component for
# an extension module: it wants the headers and, on Windows, the import
# library, but never the embeddable libpython. Asking for Development
# would fail on any distro that ships headers without a static library,
# and would quietly link something no extension should link.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sstvae_native MODULE module.cpp)
target_link_libraries(sstvae_native PRIVATE sstvae_core)

if(SSTVAE_BUILD_CODEC)
  target_link_libraries(sstvae_native PRIVATE sstvae_codec)
  target_compile_definitions(sstvae_native PRIVATE SSTVAE_HAVE_CODEC)
  # The extension has to find libonnxruntime at import time. An RPATH
  # beats setting LD_LIBRARY_PATH in every test invocation, and beats
  # copying the library somewhere it could go stale.
  if(APPLE)
    set_property(TARGET sstvae_native APPEND PROPERTY
      BUILD_RPATH "${SSTVAE_ONNXRUNTIME_LIBDIR}")
  elseif(NOT WIN32)
    set_property(TARGET sstvae_native APPEND PROPERTY
      BUILD_RPATH "${SSTVAE_ONNXRUNTIME_LIBDIR}")
  else()
    # Windows has no RPATH: put the DLL beside the .pyd instead.
    add_custom_command(TARGET sstvae_native POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${SSTVAE_ONNXRUNTIME_LIBDIR}/onnxruntime.dll"
              "$<TARGET_FILE_DIR:sstvae_native>"
      VERBATIM)
  endif()
endif()

# Dropped in one predictable place that tests/conftest.py adds to
# sys.path, rather than installed into site-packages: the module is a
# test fixture, and installing it would let a stale copy shadow a fresh
# build without anyone noticing.
set_target_properties(sstvae_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

message(STATUS "pybind11 module will be built for ${Python3_EXECUTABLE}")
