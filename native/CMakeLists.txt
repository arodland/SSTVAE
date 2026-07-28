# Native SSTVAE -- C++20 core, Qt front end to follow.
#
# See docs/native-app.md. Phase 0 is scaffolding plus the parity
# harness: the Python package remains the normative definition of the
# on-air format, and everything here is checked against it.
#
#   cmake -S native -B native/build -G Ninja
#   cmake --build native/build
#   ctest --test-dir native/build --output-on-failure
#
# Options:
#   SSTVAE_BUILD_PYMODULE  pybind11 module, so pytest can drive the C++ core
#   SSTVAE_SANITIZE        ASan + UBSan (there is no memory-safety net here)

cmake_minimum_required(VERSION 3.24)
project(sstvae_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(SSTVAE_BUILD_PYMODULE "Build the pybind11 module for pytest parity runs" ON)
option(SSTVAE_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

# The numeric core is compared against Python to ~1e-12. Fast-math would
# licence the compiler to reassociate exactly the sums those tolerances
# are measuring, so parity failures would become a function of the
# optimizer rather than of the code. Not negotiable in this project.
if(MSVC)
  add_compile_options(/fp:precise /W4)
else()
  add_compile_options(-fno-fast-math -Wall -Wextra)
endif()

if(SSTVAE_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "SSTVAE_SANITIZE is for the GCC/Clang builds")
  endif()
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- core -------------------------------------------------------------------
# No QtWidgets anywhere below here; enforced by tools/check_layering.py.
add_library(sstvae_core STATIC
  core/golay/golay.cpp
  core/dsp/dsp.cpp
  core/framing/framing.cpp
  core/framing/interleaver_table.cpp
  core/ofdm/ofdm.cpp
)
target_include_directories(sstvae_core PUBLIC core third_party/pocketfft)
# Position-independent so the static core can be linked into the
# pybind11 shared module.
set_target_properties(sstvae_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# config.hpp is generated from sstvae/config.py and must never be
# hand-edited. It is committed, so a plain cmake build needs no Python;
# this target regenerates it on demand and CI checks the diff is empty.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_custom_target(sstvae_config_header
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_config_header.py
    COMMENT "Regenerating native/core/config.hpp from sstvae/config.py"
    VERBATIM)
endif()

# --- tests ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- bindings ---------------------------------------------------------------
if(SSTVAE_BUILD_PYMODULE)
  add_subdirectory(bindings/module)
endif()
