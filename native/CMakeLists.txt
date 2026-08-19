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
project(sstvae_native VERSION 0.4.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# clangd reads this to get the real include paths and defines for every
# target (Qt, onnxruntime, pybind11/Python.h, the bundled Hamlib
# headers); without it clangd falls back to guessed flags and reports
# most of the tree as undefined. .clangd points at the directory this
# lands in. Every generator CMake supports here (Ninja, Makefiles) can
# emit it; the root .gitignore already excludes the resulting file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The oldest macOS this is built to run on. Decision 2 in
# docs/native-app.md sets a conservative hardware floor, and Qt 6.8
# supports back to 12; stated explicitly so it is a choice rather than
# whatever SDK the runner happens to have.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "" FORCE)
endif()

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
  # A named, project-wide define rather than a compiler-detection macro
  # like __SANITIZE_ADDRESS__: a test that wants to trade trial count
  # for runtime under instrumentation is answering "is this the
  # sanitizer build", not "did the compiler happen to define ASan's own
  # macro" -- see test_beacon.cpp's use of it.
  add_compile_definitions(SSTVAE_SANITIZE_BUILD)
endif()

# --- core -------------------------------------------------------------------
# No QtWidgets anywhere below here; enforced by tools/check_layering.py.
add_library(sstvae_core STATIC
  core/golay/golay.cpp
  core/beacon/beacon.cpp
  core/dsp/dsp.cpp
  core/dsp/spectrum.cpp
  core/dsp/morse.cpp
  core/dsp/leader.cpp
  core/framing/framing.cpp
  core/audio/audio.cpp
  core/checkpoint/checkpoint.cpp
  core/audio/wavio.cpp
  core/framing/interleaver_table.cpp
  core/images/images.cpp
  core/modem/modem.cpp
  core/latents/latents.cpp
  core/log/log.cpp
  core/util/event.cpp
  core/rx/ringbuffer.cpp
  core/rig/controller.cpp
  core/rx/engine.cpp
  core/tx/engine.cpp
  core/settings/settings.cpp
  core/ofdm/ofdm.cpp
  core/optimize/optimize.cpp
  core/optimize/speculative.cpp
  core/overlay/model.cpp
  core/sync/sync.cpp
  # Vendored, and the only non-header-only one -- see
  # third_party/easyexif/README.md. Read for JPEG orientation alone.
  third_party/easyexif/exif.cpp
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
  add_library(sstvae_codec STATIC core/codec/codec.cpp core/codec/grad_session.cpp)
  set_target_properties(sstvae_codec PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_codec PUBLIC core)
  target_link_libraries(sstvae_codec PUBLIC sstvae_core PRIVATE onnxruntime)
  # onnxruntime's headers are not ours and are not warning-clean under
  # -Wall -Wextra; treat them as system so our own warnings stay signal.
  # The resolved include directory, not ROOT/include: the Android AAR
  # spells it `headers/`, and hardcoding the desktop layout finds the
  # right download and then looks for a directory that was never in it.
  target_include_directories(sstvae_codec SYSTEM PRIVATE
    ${SSTVAE_ONNXRUNTIME_INCLUDE})
endif()

# Put the onnxruntime runtime where `target` can find it.
#
# A function rather than the one POST_BUILD it started as: sstvae-decode
# had the copy and sstvae-gui did not, and the GUI still ran in CI only
# because the two share a build directory. That is not a dependency, it
# is a coincidence -- and it evaporates the moment anything is copied
# somewhere else, which is exactly what packaging does.
function(sstvae_onnxruntime_copy_runtime target)
  if(NOT TARGET sstvae_codec)
    return()
  endif()
  if(NOT WIN32)
    # Elsewhere CMake already adds a linked shared library's directory
    # to the build rpath; this is explicit so it survives a target that
    # links the codec only transitively.
    set_property(TARGET ${target} APPEND PROPERTY
      BUILD_RPATH "${SSTVAE_ONNXRUNTIME_LIBDIR}")
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SSTVAE_ONNXRUNTIME_LIBDIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${target}>"
    VERBATIM)
endfunction()

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

# --- Android audio ----------------------------------------------------------
# The Android counterpart of core/audio/qt/, and a separate library for the
# same reason: the modem, the codec and both engines must still build with no
# platform audio at all. Only built when targeting Android, where <jni.h> and
# the NDK's log/android libraries exist -- ON elsewhere would be a request
# that cannot be satisfied, so this is not a tri-state switch like the others.
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  add_library(sstvae_audio_android STATIC core/audio/android/androidaudio.cpp)
  set_target_properties(sstvae_audio_android PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_audio_android PUBLIC core)
  target_link_libraries(sstvae_audio_android PUBLIC sstvae_core PRIVATE android log)
  # The Java half lives with the C++ half rather than in an app: an app
  # supplying its own would be free to get the blocking-read loop wrong.
  # Consumers add this to their Gradle sourceSets.
  set(SSTVAE_ANDROID_AUDIO_JAVA_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/core/audio/android/java" CACHE INTERNAL "")
  message(STATUS "Android audio: ${ANDROID_ABI}")
endif()

# --- Qt image loading -------------------------------------------------------
# Loading a picture through Qt's decoders, so the app accepts what the
# platform can read (TIFF, WEBP, ICO, whatever the imageformats plugins
# offer) rather than the fixed list stb happens to implement.
#
# Gated on Qt6 Gui being available and nothing else, the same shape as
# sstvae_fetch below: there is no user-facing switch because there is
# nothing to decide -- the stb loader in sstvae_core is still there and
# is still what the headless tools and the parity suite use, so a build
# without Qt loses no capability it had.
#
# Before the overlay block, which links this: an if() on a variable or a
# target defined later is silently false, and that has already left a
# test built-and-not-run here once.
find_package(Qt6 6.2 COMPONENTS Core Gui QUIET)
if(Qt6Gui_FOUND)
  add_library(sstvae_images_qt STATIC core/images/qt/qtimages.cpp)
  set_target_properties(sstvae_images_qt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sstvae_images_qt PUBLIC core)
  target_link_libraries(sstvae_images_qt PUBLIC sstvae_core Qt6::Core Qt6::Gui)
  message(STATUS "Qt image loading: Qt ${Qt6_VERSION}")
endif()

# --- extra image format plugins ---------------------------------------------
# Formats Qt's own imageformats plugins do not carry, added the way Qt
# intends: a plugin in the loader's search path. **Nothing in our code
# refers to these** -- `core/images/qt/` goes through QImageReader, so a
# plugin here widens `readable_extensions()`, the transmit dialog's
# filter and what `images::qt::load` opens, with no change to any of
# them. That is the whole reason this is a plugin rather than a fourth
# branch in the loader.
#
# HEIC is behind its own switch, and not because of the build: libheif
# and libde265 are LGPL-3.0-or-later and fine to bundle, but HEVC is
# patent-encumbered, and whether to ship a decoder for it is a
# publisher's decision rather than a technical one. OFF removes the
# format and nothing else. See native/cmake/libheif.cmake.
set(SSTVAE_BUILD_HEIF "AUTO" CACHE STRING
    "Build the HEIF/HEIC image plugin (ON/OFF/AUTO)")
set_property(CACHE SSTVAE_BUILD_HEIF PROPERTY STRINGS ON OFF AUTO)

# One directory for every bundled plugin, laid out the way Qt expects
# (`<root>/imageformats/`), so a single QT_PLUGIN_PATH entry serves the
# tests, a developer running the app out of the build tree, and the
# packaging script.
set(SSTVAE_PLUGIN_DIR "${CMAKE_BINARY_DIR}/plugins" CACHE INTERNAL "")

if(SSTVAE_BUILD_HEIF STREQUAL "OFF")
  set(_want_heif OFF)
elseif(WIN32 AND NOT SSTVAE_BUILD_HEIF STREQUAL "ON")
  # **AUTO declines on Windows, and this is a known gap rather than a
  # judgement about the platform.** The plugin builds there and libheif
  # and libde265 build there, but Qt will not load the plugin: the format
  # never appears in `supportedImageFormats()`, so HEIC is simply absent.
  # Before the DLLs were placed beside the executable it *did* load and
  # then crashed inside the decode, which is consistent with it having
  # bound some other libheif on that machine -- but the current state is
  # a plugin that does not load, and the cause is not established.
  #
  # Four platforms carry HEIC; Windows still gains everything Qt's own
  # imageformats plugins bring (TIFF, WEBP, ICO, ICNS, TGA), which is the
  # larger part of what this change set is for. `-DSSTVAE_BUILD_HEIF=ON`
  # forces it on for anyone with a Windows machine to debug on -- which is
  # what this needs and what CI cannot provide. The evidence so far is
  # written up in docs/todo.md; do not start from scratch.
  set(_want_heif OFF)
  message(STATUS "HEIF/HEIC image plugin: off on Windows (known gap, see "
                 "docs/todo.md); pass -DSSTVAE_BUILD_HEIF=ON to build it anyway")
else()
  find_package(Qt6 6.2 COMPONENTS Core Gui QUIET)
  if(Qt6Gui_FOUND)
    set(_want_heif ON)
  elseif(SSTVAE_BUILD_HEIF STREQUAL "ON")
    message(FATAL_ERROR
      "SSTVAE_BUILD_HEIF=ON but Qt6 Gui was not found; a Qt image format "
      "plugin needs it. Pass -DSSTVAE_BUILD_HEIF=OFF to build without HEIC.")
  else()
    set(_want_heif OFF)
  endif()
endif()

if(_want_heif)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  include(libheif)
  # MODULE, not SHARED: a Qt plugin is dlopen'd and never linked against.
  add_library(sstvae_plugin_heif MODULE
    third_party/qt-heif-plugin/heif.cpp)
  set_target_properties(sstvae_plugin_heif PROPERTIES
    AUTOMOC ON
    OUTPUT_NAME "sstvae_qheif"
    # Qt looks for `<library path>/imageformats/`, so the subdirectory is
    # part of the contract and not tidiness.
    LIBRARY_OUTPUT_DIRECTORY "${SSTVAE_PLUGIN_DIR}/imageformats"
    RUNTIME_OUTPUT_DIRECTORY "${SSTVAE_PLUGIN_DIR}/imageformats")
  target_link_libraries(sstvae_plugin_heif PRIVATE Qt6::Gui sstvae::libheif)
  # Vendored and not warning-clean under -Wall -Wextra; its warnings are
  # not signal here, the same treatment stb gets.
  target_include_directories(sstvae_plugin_heif SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qt-heif-plugin)
  if(NOT MSVC)
    target_compile_options(sstvae_plugin_heif PRIVATE -w)
  endif()
  # **A relative rpath, because the binary that ships is the one built
  # here.** Nothing `install()`s this plugin -- `package_app.sh` copies it
  # out of the build tree -- so its *build* rpath is what a user gets, and
  # that contains an absolute path into `_deps` which will not exist on
  # their machine. It happens to work anyway, because the AppImage's
  # launcher sets LD_LIBRARY_PATH and that is searched first; relying on
  # that would mean the plugin only works when launched through the
  # wrapper. These entries make the staged layout self-sufficient:
  # plugins/imageformats -> lib on Linux, PlugIns/imageformats ->
  # Frameworks in a bundle. Windows searches the .exe's own directory and
  # needs no rpath at all.
  if(APPLE)
    set_property(TARGET sstvae_plugin_heif APPEND PROPERTY
      BUILD_RPATH "@loader_path/../../Frameworks")
  elseif(NOT WIN32)
    set_property(TARGET sstvae_plugin_heif APPEND PROPERTY
      BUILD_RPATH "$ORIGIN/../../lib")
  endif()
  message(STATUS "HEIF/HEIC image plugin: on, decode only")
endif()

# The libheif DLLs beside the *executable*, on Windows.
#
# Defined here rather than in cmake/libheif.cmake, and **unconditionally**,
# for the reason `sstvae_onnxruntime_copy_runtime` above is: that module is
# only included when the plugin is being built, so a call site would have
# to know whether the function exists. It did not, and
# `-DSSTVAE_BUILD_HEIF=OFF` therefore failed at configure time with
# "Unknown CMake command" -- a switch added and never once configured with
# it off. An early return keeps every call site unconditional.
#
# The failure this retires when it *is* built: Windows resolves a dlopen'd
# module's imports against the search order of the *process*, which starts
# at the .exe's directory and not the plugin's. So heif.dll next to
# plugins/imageformats/ is no use; it has to be next to the executable.
# Missing, the process still starts and everything works except that HEIC
# is quietly absent -- no error, no exit code. Hamlib's equivalent stops
# the process before `main`, which is louder and easier.
function(sstvae_libheif_copy_runtime target)
  if(NOT TARGET sstvae_plugin_heif OR NOT WIN32 OR NOT SSTVAE_LIBHEIF_RUNTIME_DIR)
    return()
  endif()
  file(GLOB _heif_dlls "${SSTVAE_LIBHEIF_RUNTIME_DIR}/*.dll")
  if(NOT _heif_dlls)
    message(FATAL_ERROR
      "No DLLs found in ${SSTVAE_LIBHEIF_RUNTIME_DIR}; the HEIF plugin would "
      "load nowhere and the format would be silently missing from ${target}.")
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_heif_dlls}
            "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying the libheif runtime beside ${target}")
endfunction()

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
  # TARGET rather than the find_package result alone: an inset image is
  # loaded through sstvae_images_qt, so that is a real requirement and
  # not a coincidence of both being gated on Qt6 Gui.
  if(Qt6_FOUND AND TARGET sstvae_images_qt)
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
  target_link_libraries(sstvae_overlay_render PUBLIC sstvae_core sstvae_images_qt
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
# Gated on Qt Network being available and nothing else. It used to be
# gated on `_want_qtaudio`, which happened to hold for every desktop
# build and is the wrong condition: the Android app wants the downloader
# and explicitly does *not* want Qt audio, and the coupling made the
# fetcher silently vanish there -- a missing artifact reported as
# "offline" on a machine with a working network.
find_package(Qt6 6.2 COMPONENTS Network QUIET)
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
  foreach(_dep sstvae_codec sstvae_audio_qt sstvae_rig sstvae_overlay_render
               sstvae_images_qt)
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
  # Everything but main() goes in a library, so the widgets can be
  # driven by a test as well as by the application. A QWidget's
  # behaviour is mostly its painting, which needs eyes -- but the parts
  # that are not (a scroll that must go the right way, a buffer that
  # must survive a resize) have been wrong before and are checkable.
  #
  # AUTOMOC only here: everything under core/ is deliberately free of
  # QObject, so running moc over it would be scanning for something the
  # layering rules forbid.
  add_library(sstvae_gui STATIC
    gui/main_window.cpp
    gui/style.cpp
    gui/flow_layout.cpp
    gui/pane_container.cpp
    gui/picture_box.cpp
    gui/waterfall.cpp
    gui/rx_panel.cpp
    gui/rig_config.cpp
    gui/settings_dialog.cpp
    gui/overlay_editor.cpp
    gui/tx_panel.cpp
    gui/banner.cpp
    gui/crop_dialog.cpp
    gui/log_pane.cpp
    gui/app_state.cpp)
  set_target_properties(sstvae_gui PROPERTIES AUTOMOC ON)
  # The application's own version, for the About box. It reported
  # Hamlib's and Qt's and omitted this one -- which is the number every
  # bug report needs first, and the only one nobody could read off the
  # running program.
  target_compile_definitions(sstvae_gui PUBLIC SSTVAE_VERSION="${PROJECT_VERSION}")
  target_include_directories(sstvae_gui PUBLIC core gui)
  target_link_libraries(sstvae_gui PUBLIC
    sstvae_core sstvae_codec sstvae_audio_qt sstvae_rig sstvae_overlay_render
    sstvae_images_qt sstvae_fetch
    Qt6::Core Qt6::Gui Qt6::Widgets)

  add_executable(sstvae-gui WIN32 MACOSX_BUNDLE gui/main.cpp)
  set_target_properties(sstvae-gui PROPERTIES AUTOMOC ON)

  # --- the icon, which has to be attached differently on each platform ---
  #
  # Note what this is *not*: a QIcon set at runtime. Two of the three
  # places an icon appears are drawn by the operating system without
  # asking the process -- Explorer and the taskbar read the executable's
  # resource table, Finder and the Dock read the bundle -- so an icon that
  # exists only in the running app's window is an icon most users never
  # see. Linux is the exception, and it reads the .desktop file instead,
  # which is why that platform's icon is handled by the packaging script.
  #
  # All three formats are generated from one SVG by tools/gen_icons.py and
  # committed, so this needs no image tooling at build time.
  set(SSTVAE_PACKAGING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/packaging")
  # Every platform: the window icon, compiled in. See packaging/icons.qrc
  # for why this is separate from the two below rather than instead of them.
  target_sources(sstvae-gui PRIVATE "${SSTVAE_PACKAGING_DIR}/icons.qrc")
  set_target_properties(sstvae-gui PROPERTIES AUTORCC ON)
  if(WIN32)
    # Explicit: CMake enables RC for MSVC on its own, but not for a MinGW
    # or clang-cl toolchain, where the .rc would then be treated as an
    # unknown source type and silently ignored -- leaving a working
    # executable with the generic icon, which looks like the resource
    # file being wrong rather than unbuilt.
    enable_language(RC)
    configure_file("${SSTVAE_PACKAGING_DIR}/sstvae.rc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/sstvae.rc" @ONLY)
    target_sources(sstvae-gui PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/sstvae.rc")
  endif()
  if(APPLE)
    # CMake's default Info.plist produces a bundle macOS will launch but
    # not one it will trust with a microphone, and a bundle with no
    # stable identifier is one TCC cannot remember a decision about.
    # See cmake/MacOSXBundleInfo.plist.in.
    #
    # The .icns is a *source* with a package location rather than an
    # install() rule, so it is copied into Contents/Resources as part of
    # building the target: the bundle in the build tree then has its icon
    # too, and there is no way to run a build that packages without it.
    set(_sstvae_icns "${SSTVAE_PACKAGING_DIR}/sstvae.icns")
    target_sources(sstvae-gui PRIVATE "${_sstvae_icns}")
    set_source_files_properties("${_sstvae_icns}" PROPERTIES
      MACOSX_PACKAGE_LOCATION Resources)
    set_target_properties(sstvae-gui PROPERTIES
      MACOSX_BUNDLE_INFO_PLIST
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MacOSXBundleInfo.plist.in"
      MACOSX_BUNDLE_BUNDLE_NAME "SSTVAE"
      MACOSX_BUNDLE_GUI_IDENTIFIER "org.cleverdomain.sstvae"
      MACOSX_BUNDLE_ICON_FILE "sstvae.icns"
      MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
      MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}")
  endif()
  target_link_libraries(sstvae-gui PRIVATE sstvae_gui)
  sstvae_hamlib_copy_runtime(sstvae-gui)
  sstvae_libheif_copy_runtime(sstvae-gui)
  sstvae_onnxruntime_copy_runtime(sstvae-gui)
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
  sstvae_onnxruntime_copy_runtime(sstvae-decode)
endif()

# The soundcard path's only real test. `core/audio/qt/` is device
# enumeration and moving bytes; the only way to check it is to move
# bytes through a device, so this is a tool rather than a ctest.
if(_want_qtaudio)
  add_executable(sstvae-audio-check apps/sstvae_audio_check.cpp)
  target_link_libraries(sstvae-audio-check PRIVATE sstvae_core sstvae_audio_qt
                        Qt6::Core)
endif()

# Renders the app's windows to PNG, headless -- so a layout can be
# looked at without a display. Not a ctest: "is this laid out well" has
# no oracle, and a stored-PNG comparison fails on every font and theme
# it was not recorded with. See the file's comment for what it caught.
if(TARGET sstvae_gui)
  add_executable(sstvae-gui-shot apps/sstvae_gui_shot.cpp)
  target_link_libraries(sstvae-gui-shot PRIVATE sstvae_gui)
  # Anything linking sstvae_gui pulls in sstvae_rig and therefore the
  # Hamlib DLL. On Windows an unresolved import stops the process before
  # main with no output at all -- see the note in hamlib.cmake.
  sstvae_hamlib_copy_runtime(sstvae-gui-shot)
  sstvae_libheif_copy_runtime(sstvae-gui-shot)
  sstvae_onnxruntime_copy_runtime(sstvae-gui-shot)
endif()
