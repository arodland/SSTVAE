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
set(SSTVAE_BUILD_OVERLAY "AUTO" CACHE STRING "Build Qt overlay rendering (ON/OFF/AUTO)")
set_property(CACHE SSTVAE_BUILD_OVERLAY PROPERTY STRINGS ON OFF AUTO)
set(SSTVAE_BUILD_GUI "AUTO" CACHE STRING "Build the desktop application (ON/OFF/AUTO)")
set_property(CACHE SSTVAE_BUILD_GUI PROPERTY STRINGS ON OFF AUTO)
# Rig control. Plain ON/OFF rather than the audio layer's AUTO, because
# Hamlib is pinned and bundled (native/cmake/hamlib.cmake) -- there is
# no system library to detect, so "auto" would have nothing to decide.
# OFF still builds and tests the threading design in
# core/rig/controller.cpp, which has no external dependency at all.
option(SSTVAE_BUILD_RIG "Build libhamlib rig control" ON)

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
  core/checkpoint/checkpoint.cpp
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

# --- Overlay rendering ------------------------------------------------------
# Separate from sstvae_core for the same reason Qt audio is: the modem,
# the golden vectors and the parity module have no business needing a
# GUI toolkit. QtGui only -- check_layering.py allows it here and
# nowhere else under core/ -- so an overlay renders from the command
# line under QT_QPA_PLATFORM=offscreen, which is what keeps composition
# available to the headless tools and not just the editor.
if(SSTVAE_BUILD_OVERLAY STREQUAL "OFF")
  set(_want_overlay OFF)
else()
  find_package(Qt6 6.2 COMPONENTS Core Gui QUIET)
  if(Qt6_FOUND)
    set(_want_overlay ON)
  elseif(SSTVAE_BUILD_OVERLAY STREQUAL "ON")
    message(FATAL_ERROR
      "SSTVAE_BUILD_OVERLAY=ON but Qt6 Gui was not found. Install it, or "
      "pass -DSSTVAE_BUILD_OVERLAY=OFF to build without overlay rendering.")
  else()
    set(_want_overlay OFF)
    message(STATUS "Qt6 Gui not found; building without overlay rendering")
  endif()
endif()

if(_want_overlay)
  add_library(sstvae_overlay_render STATIC core/overlay/render.cpp)
  set_target_properties(sstvae_overlay_render PROPERTIES
                        POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_overlay_render PUBLIC core)
  target_link_libraries(sstvae_overlay_render PUBLIC sstvae_core
                        PRIVATE Qt6::Core Qt6::Gui)
  message(STATUS "Overlay rendering: Qt ${Qt6_VERSION}")
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

# --- checkpoint fetch -------------------------------------------------------
# Only the downloader; resolution and the cache live in the core with no
# dependency, so a build without this still honours --model and still
# finds an already-cached artifact.
if(_want_qtaudio)
  find_package(Qt6 6.2 COMPONENTS Network QUIET)
endif()
if(Qt6Network_FOUND)
  add_library(sstvae_fetch STATIC core/checkpoint/qt_fetcher.cpp)
  set_target_properties(sstvae_fetch PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_fetch PUBLIC core)
  target_link_libraries(sstvae_fetch PUBLIC sstvae_core
                        PRIVATE Qt6::Core Qt6::Network)
endif()

# --- rig --------------------------------------------------------------------
# Only the libhamlib backend lives here; the controller is in the core.
if(SSTVAE_BUILD_RIG)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  include(hamlib)
  add_library(sstvae_rig STATIC core/rig/hamlib.cpp)
  set_target_properties(sstvae_rig PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_rig PUBLIC core)
  target_link_libraries(sstvae_rig PUBLIC sstvae_core PRIVATE sstvae::hamlib)
  if(MSVC)
    # hamlib/rig.h includes <pthread.h> unconditionally and MSVC has
    # none. See the header for why a shim is safe here and what keeps it
    # so. SYSTEM, so it never shadows a real pthread.h on any other
    # toolchain and produces no warnings of its own.
    target_include_directories(sstvae_rig SYSTEM PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/msvc-pthread)
  endif()
endif()

# --- the desktop application ------------------------------------------------
# The only place QtWidgets is allowed, enforced by check_layering.py.
#
# Unlike every other optional piece here, this one is not optional
# *because of an external dependency we would rather avoid* -- Qt is the
# app's toolkit by design. It is a switch so that the modem, the CLI and
# the parity module still build on a machine with no GUI stack at all,
# which is every CI job but one and every headless station.
#
# It needs more than Qt: the codec to make a picture, the audio layer to
# hear one, the rig to key a radio, and the overlay renderer to compose
# one. Each of those is separately optional, so AUTO means "when
# everything is present" and ON has to say which piece is missing --
# "Qt6 not found" would be a lie when the real problem is -DSSTVAE_BUILD_CODEC=OFF.
# Placed before add_subdirectory(tests) with the rest: an if() on a
# variable defined later is silently false, which once left a test
# built-and-not-run and ctest reporting green over it.
if(SSTVAE_BUILD_GUI STREQUAL "OFF")
  set(_want_gui OFF)
else()
  set(_gui_missing "")
  find_package(Qt6 6.2 COMPONENTS Core Gui Widgets QUIET)
  if(NOT Qt6_FOUND)
    list(APPEND _gui_missing "Qt6 Widgets")
  endif()
  foreach(_dep sstvae_codec sstvae_audio_qt sstvae_rig sstvae_overlay_render)
    if(NOT TARGET ${_dep})
      list(APPEND _gui_missing ${_dep})
    endif()
  endforeach()

  if(_gui_missing)
    if(SSTVAE_BUILD_GUI STREQUAL "ON")
      string(REPLACE ";" ", " _gui_missing_text "${_gui_missing}")
      message(FATAL_ERROR
        "SSTVAE_BUILD_GUI=ON but the application needs: ${_gui_missing_text}. "
        "Install what is missing, or pass -DSSTVAE_BUILD_GUI=OFF.")
    endif()
    set(_want_gui OFF)
    string(REPLACE ";" ", " _gui_missing_text "${_gui_missing}")
    message(STATUS "Not building the app; missing: ${_gui_missing_text}")
  else()
    set(_want_gui ON)
  endif()
endif()

if(_want_gui)
  # AUTOMOC only here: everything under core/ is deliberately free of
  # QObject, so running moc over it would be scanning for something the
  # layering rules forbid.
  add_executable(sstvae-gui WIN32 MACOSX_BUNDLE
    gui/main.cpp
    gui/main_window.cpp
    gui/app_state.cpp)
  set_target_properties(sstvae-gui PROPERTIES AUTOMOC ON)
  target_include_directories(sstvae-gui PRIVATE core gui)
  target_link_libraries(sstvae-gui PRIVATE
    sstvae_core sstvae_codec sstvae_audio_qt sstvae_rig sstvae_overlay_render
    sstvae_fetch
    Qt6::Core Qt6::Gui Qt6::Widgets)
  sstvae_hamlib_copy_runtime(sstvae-gui)
  message(STATUS "Desktop application: Qt ${Qt6_VERSION}")
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
  # Optional: without it the CLI still honours --model and a warm cache,
  # it just cannot perform the first-run download.
  if(TARGET sstvae_fetch)
    target_link_libraries(sstvae-decode PRIVATE sstvae_fetch Qt6::Core)
    target_compile_definitions(sstvae-decode PRIVATE SSTVAE_HAVE_FETCH)
  endif()
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
