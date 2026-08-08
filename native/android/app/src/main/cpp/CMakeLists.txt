# The smoke test's native library.
#
# This is a *consumer* of native/, not a fork of it: it adds the real
# tree as a subdirectory and links `sstvae_core`, so the modem, the
# framing, the sync and the whole receive state machine here are byte
# for byte the ones the desktop app and the golden vectors test. An
# Android build is a fourth build of the same code (docs/android.md),
# and that is the property worth protecting -- the moment this directory
# starts holding its own copy of anything, the port has acquired a
# parity surface it does not need.

cmake_minimum_required(VERSION 3.22)
project(sstvae_smoke LANGUAGES CXX)

# The core's CMakeLists sets this, but that is *directory* scope and
# this target is created here, in the parent -- so without it
# `sstvae_smoke` compiles at the NDK's default and every `std::span` in
# a core header is an error, while the core's own files build fine.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SSTVAE_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")

# Everything the desktop build switches on that has no meaning here.
# The GUI is QtWidgets, the audio layer is QtMultimedia, rig control
# wants /dev/ttyUSB, and the parity module needs a host Python -- all
# four are off, and the core still builds, which is the whole point of
# them being separate libraries behind seams.
set(SSTVAE_BUILD_GUI OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_QTAUDIO OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_OVERLAY OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_RIG OFF CACHE BOOL "" FORCE)
set(SSTVAE_BUILD_PYMODULE OFF CACHE BOOL "" FORCE)

# The codec is the only piece that downloads anything. Left ON by
# default, but -DSSTVAE_BUILD_CODEC=OFF still produces a working app
# that reports frames, SNR and callsign without making a picture --
# which is most of what this smoke test is for, and is available only
# because `rx::Decoder` is a seam.
if(NOT DEFINED SSTVAE_BUILD_CODEC)
  set(SSTVAE_BUILD_CODEC ON CACHE BOOL "" FORCE)
endif()

# EXCLUDE_FROM_ALL so the desktop tools, tests and apps in that tree are
# configured but not built: this library needs the core libraries only,
# and building sstvae-decode for Android would fail on things it has no
# reason to have.
add_subdirectory("${SSTVAE_NATIVE_DIR}" sstvae_native EXCLUDE_FROM_ALL)

add_library(sstvae_smoke SHARED jni_bridge.cpp)
target_link_libraries(sstvae_smoke PRIVATE sstvae_core android log)

if(TARGET sstvae_codec)
  target_link_libraries(sstvae_smoke PRIVATE sstvae_codec)
  target_compile_definitions(sstvae_smoke PRIVATE SSTVAE_SMOKE_HAVE_CODEC)
endif()
