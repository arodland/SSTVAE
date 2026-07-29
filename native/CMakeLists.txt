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
# The codec is a separate library and a separate switch because it is
# the only part of the core with a heavyweight external dependency.
# Everything through Phase 1 -- the entire modem -- builds and tests
# without downloading anything, and that stays true.
option(SSTVAE_BUILD_CODEC "Build the ONNX codec (downloads onnxruntime)" ON)
# Soundcard I/O needs Qt Multimedia. AUTO builds it when Qt6 is present
# and quietly skips it otherwise, which is what keeps a machine with no
# Qt able to build and test the entire modem, codec and both engines --
# the engines take their player as a seam for exactly this reason.
# ON makes a missing Qt an error, for a packaging build that must not
# silently ship without audio.
set(SSTVAE_BUILD_QTAUDIO "AUTO" CACHE STRING "Build Qt Multimedia audio (ON/OFF/AUTO)")
set_property(CACHE SSTVAE_BUILD_QTAUDIO PROPERTY STRINGS ON OFF AUTO)
# Rig control needs libhamlib. Same AUTO/ON/OFF contract as the audio
# layer, and for the same reason: the threading design that makes rig
# control safe lives in core/rig/controller.cpp, which has no external
# dependency at all and is tested against a fake backend -- so a machine
# with no Hamlib still builds and checks the part that can be wrong.
set(SSTVAE_BUILD_RIG "AUTO" CACHE STRING "Build libhamlib rig control (ON/OFF/AUTO)")
set_property(CACHE SSTVAE_BUILD_RIG PROPERTY STRINGS ON OFF AUTO)

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
  core/beacon/beacon.cpp
  core/dsp/dsp.cpp
  core/framing/framing.cpp
  core/audio/audio.cpp
  core/audio/wavio.cpp
  core/framing/interleaver_table.cpp
  core/images/images.cpp
  core/modem/modem.cpp
  core/latents/latents.cpp
  core/util/event.cpp
  core/rx/ringbuffer.cpp
  core/rig/controller.cpp
  core/rx/engine.cpp
  core/tx/engine.cpp
  core/settings/settings.cpp
  core/ofdm/ofdm.cpp
  core/overlay/model.cpp
  core/sync/sync.cpp
)
target_include_directories(sstvae_core PUBLIC core third_party/pocketfft)
# stb is SYSTEM: 20k lines of vendored single-header C that is not
# warning-clean under -Wall -Wextra, and its warnings are not signal.
target_include_directories(sstvae_core SYSTEM PUBLIC third_party/stb
                                                   third_party/dr_libs
                                                   third_party)
# Position-independent so the static core can be linked into the
# pybind11 shared module.
set_target_properties(sstvae_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- codec ------------------------------------------------------------------
if(SSTVAE_BUILD_CODEC)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  include(onnxruntime)
  add_library(sstvae_codec STATIC core/codec/codec.cpp)
  set_target_properties(sstvae_codec PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_codec PUBLIC core)
  target_link_libraries(sstvae_codec PUBLIC sstvae_core PRIVATE onnxruntime)
  # onnxruntime's headers are not ours and are not warning-clean under
  # -Wall -Wextra; treat them as system so our own warnings stay signal.
  target_include_directories(sstvae_codec SYSTEM PRIVATE
    ${SSTVAE_ONNXRUNTIME_ROOT}/include)
endif()

# --- Qt audio ---------------------------------------------------------------
# Deliberately a separate library from sstvae_core, not a source file in
# it: linking Qt into the core would make the modem, the golden-vector
# tests and the pybind11 parity module all depend on a GUI toolkit for
# no reason. Nothing above this line knows Qt exists.
if(SSTVAE_BUILD_QTAUDIO STREQUAL "OFF")
  set(_want_qtaudio OFF)
else()
  # 6.2 (the first LTS) rather than something newer: QAudioSource,
  # QAudioSink, QAudioDevice and QMediaDevices have all been there
  # since 6.0, so a higher floor buys nothing and costs
  # distributions -- ubuntu-latest ships 6.4.2, which a 6.5 floor
  # rejected outright.
  find_package(Qt6 6.2 COMPONENTS Core Multimedia QUIET)
  if(Qt6_FOUND)
    set(_want_qtaudio ON)
  elseif(SSTVAE_BUILD_QTAUDIO STREQUAL "ON")
    message(FATAL_ERROR
      "SSTVAE_BUILD_QTAUDIO=ON but Qt6 Multimedia was not found. "
      "Install it, or pass -DSSTVAE_BUILD_QTAUDIO=OFF to build without "
      "soundcard support.")
  else()
    set(_want_qtaudio OFF)
    message(STATUS "Qt6 Multimedia not found; building without soundcard audio")
  endif()
endif()

if(_want_qtaudio)
  add_library(sstvae_audio_qt STATIC core/audio/qt/qtaudio.cpp)
  set_target_properties(sstvae_audio_qt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_audio_qt PUBLIC core)
  target_link_libraries(sstvae_audio_qt PUBLIC sstvae_core
                        PRIVATE Qt6::Core Qt6::Multimedia)
  message(STATUS "Qt Multimedia audio: ${Qt6_VERSION}")
endif()

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

# --- rig --------------------------------------------------------------------
# Only the libhamlib backend lives here; the controller is in the core.
if(SSTVAE_BUILD_RIG STREQUAL "OFF")
  set(_want_rig OFF)
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(HAMLIB QUIET IMPORTED_TARGET hamlib)
  endif()
  if(HAMLIB_FOUND)
    set(_want_rig ON)
  elseif(SSTVAE_BUILD_RIG STREQUAL "ON")
    message(FATAL_ERROR
      "SSTVAE_BUILD_RIG=ON but libhamlib was not found. Install it (the "
      "-dev/-devel package), or pass -DSSTVAE_BUILD_RIG=OFF to build "
      "without rig control.")
  else()
    set(_want_rig OFF)
    message(STATUS "libhamlib not found; building without rig control")
  endif()
endif()

if(_want_rig)
  add_library(sstvae_rig STATIC core/rig/hamlib.cpp)
  set_target_properties(sstvae_rig PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_rig PUBLIC core)
  target_link_libraries(sstvae_rig PUBLIC sstvae_core PRIVATE PkgConfig::HAMLIB)
  message(STATUS "libhamlib rig control: ${HAMLIB_VERSION}")
endif()

# --- tests ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- bindings ---------------------------------------------------------------
if(SSTVAE_BUILD_PYMODULE)
  add_subdirectory(bindings/module)
endif()

# --- apps -------------------------------------------------------------------
# Headless command-line tools. Phase 2's exit criterion is that
# sstvae-decode turns a WAV into a picture that matches the Python CLI's
# on the same input; these link no GUI toolkit, and must not start to.
if(SSTVAE_BUILD_CODEC)
  add_executable(sstvae-decode apps/sstvae_decode.cpp)
  target_link_libraries(sstvae-decode PRIVATE sstvae_core sstvae_codec)
  if(NOT WIN32)
    set_property(TARGET sstvae-decode APPEND PROPERTY
      BUILD_RPATH "${SSTVAE_ONNXRUNTIME_LIBDIR}")
  else()
    add_custom_command(TARGET sstvae-decode POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${SSTVAE_ONNXRUNTIME_LIBDIR}/onnxruntime.dll"
              "$<TARGET_FILE_DIR:sstvae-decode>"
      VERBATIM)
  endif()
endif()

# The soundcard path's only real test. `core/audio/qt/` is device
# enumeration and moving bytes; the only way to check it is to move
# bytes through a device, so this is a tool rather than a ctest.
if(_want_qtaudio)
  add_executable(sstvae-audio-check apps/sstvae_audio_check.cpp)
  target_link_libraries(sstvae-audio-check PRIVATE sstvae_core sstvae_audio_qt
                        Qt6::Core)
endif()
