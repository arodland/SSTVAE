# Tier 0: the Android listener (docs/android.md).
#
# A Qt Quick front end over the *same* native/core/ the desktop app uses
# -- an Android build is a fourth build of that code, not a
# reimplementation, which is what keeps the golden vectors and
# `pytest --native` covering it with no new parity surface.
#
# Not to be confused with `native/android/`, which is the pre-Tier-0
# smoke test: a plain Views probe that answered "can a phone select an
# audio device and decode a picture" (it can -- a TH-D75 over USB at
# >24 dB). This is the app that question was asked on behalf of.
#
# Qt for Android is a CMake build rather than a Gradle one: androiddeployqt
# generates the Gradle project. Configure with Qt's own toolchain file,
# which chains to the NDK's:
#
#   cmake -S native/android-app -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$QT_ANDROID/lib/cmake/Qt6/qt.toolchain.cmake \
#     -DQT_HOST_PATH=$QT_HOST -DANDROID_ABI=arm64-v8a
#   cmake --build build-android --target apk

cmake_minimum_required(VERSION 3.24)

# **Never ship or field-test an -O0 build of this app.** The NDK's
# Debug configuration passes no -O flag at all, which for clang means
# -O0, and the receive loop is scalar floating-point DSP over a 130 s
# ring buffer -- exactly the shape the optimizer matters most for.
# Measured on the same code: sync::acquire 171 ms -> 1513 ms (8.9x),
# the blind accumulator's full-buffer push 547 ms -> 8412 ms (15x).
#
# That is not an abstract slowdown. On a Galaxy S25+ an -O0 build spent
# 5-8 s per poll with excursions to 20-40 s, on a device whose real
# figure is a fraction of a second -- and it read as "onnxruntime is
# slow on Android", which it is not: the codec is not even inside the
# number that was being looked at. Same lesson as the sanitizer jobs in
# CLAUDE.md, on a platform where Debug is what the tooling hands you.
#
# So RelWithDebInfo is the default, and asking for Debug says so out
# loud rather than quietly costing an afternoon.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING
        "CMAKE_BUILD_TYPE=Debug builds the DSP at -O0, which costs ~6-15x "
        "in the receive loop and makes every timing figure meaningless. "
        "Use RelWithDebInfo (tools/build_android.sh) unless you are "
        "single-stepping the C++.")
endif()

project(sstvae_android VERSION 0.3.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Same reason as the desktop build: the numeric core is compared against
# Python to 1e-12, and fast-math would licence reassociating exactly the
# sums those tolerances measure.
add_compile_options(-fno-fast-math)

find_package(Qt6 6.8 REQUIRED COMPONENTS Core Gui Quick)
qt_standard_project_setup(REQUIRES 6.8)

# The shared core. Everything the desktop switches on that has no
# meaning here is off; the core still builds, which is the whole point
# of those being separate libraries behind seams.
set(SSTVAE_BUILD_GUI OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_QTAUDIO OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_OVERLAY OFF CACHE STRING "" FORCE)
set(SSTVAE_BUILD_RIG OFF CACHE BOOL "" FORCE)
set(SSTVAE_BUILD_PYMODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(.. sstvae_native EXCLUDE_FROM_ALL)

qt_add_executable(sstvae_android main.cpp listener.cpp session.cpp service_jni.cpp java_fetcher.cpp
    waterfall.cpp pictures.cpp composition.cpp transmitter.cpp)
qt_add_qml_module(sstvae_android URI SSTVAE VERSION 1.0
    QML_FILES Main.qml LevelMeter.qml CropView.qml TransmitPane.qml)
target_link_libraries(sstvae_android PRIVATE Qt6::Quick Qt6::Gui sstvae_core
                                             sstvae_audio_android
                                             sstvae_codec)

# **androiddeployqt only packages libraries it knows about.** Qt's own
# and the app's are found by walking dependencies; a third-party
# `SHARED IMPORTED` target like onnxruntime is not, so without this the
# APK builds and installs cleanly and then dies at launch with
# `dlopen failed: library "libonnxruntime.so" not found`. Same class of
# failure as the Windows import-library trap: everything links, and the
# process never reaches main.
get_target_property(_ort_so onnxruntime IMPORTED_LOCATION)
set_property(TARGET sstvae_android APPEND PROPERTY
    QT_ANDROID_EXTRA_LIBS "${_ort_so}")

# **The package source directory is assembled, not pointed at.**
#
# androiddeployqt takes exactly one `QT_ANDROID_PACKAGE_SOURCE_DIR`, and
# this app needs two sources in it: its own `android/` tree, and the
# audio layer's Java half, which lives with the layer
# (`SSTVAE_ANDROID_AUDIO_JAVA_DIR`) precisely so an app cannot supply its
# own subtly-wrong blocking-read loop. The obvious workaround -- keep a
# copy of `AudioBridge.java` under `android/src/` and remember to sync it
# -- is what was here, and it is a trap with no failure mode: the two
# copies were byte-identical, so nothing was wrong yet, and the first
# edit to either one would have shipped the other. That is exactly the
# hazard `native/core/config.hpp` has a generator and a CI gate for.
#
# So both are copied into a build-tree directory. `configure_file`
# rather than `file(COPY)` so CMake re-runs when a Java file changes, and
# `CONFIGURE_DEPENDS` so adding one is noticed too.
set(_pkg_dir "${CMAKE_CURRENT_BINARY_DIR}/android-package")
function(sstvae_stage_package_dir from into)
    file(GLOB_RECURSE _files CONFIGURE_DEPENDS RELATIVE "${from}" "${from}/*")
    foreach(_f IN LISTS _files)
        configure_file("${from}/${_f}" "${into}/${_f}" COPYONLY)
    endforeach()
endfunction()
sstvae_stage_package_dir("${CMAKE_CURRENT_SOURCE_DIR}/android" "${_pkg_dir}")
sstvae_stage_package_dir("${SSTVAE_ANDROID_AUDIO_JAVA_DIR}" "${_pkg_dir}/src")
set_property(TARGET sstvae_android APPEND PROPERTY
    QT_ANDROID_PACKAGE_SOURCE_DIR "${_pkg_dir}")

set_target_properties(sstvae_android PROPERTIES
    QT_ANDROID_PACKAGE_NAME "org.cleverdomain.sstvae"
    QT_ANDROID_APP_NAME "SSTVAE"
    QT_ANDROID_TARGET_SDK_VERSION 36
    QT_ANDROID_MIN_SDK_VERSION 28
    QT_ANDROID_VERSION_NAME "${PROJECT_VERSION}"
    QT_ANDROID_VERSION_CODE 1)

# **Never set `QT_ANDROID_PERMISSIONS` by hand.** Each element is itself
# a `name;<value>` list, joined into JSON as `"name": "<value>"`, so a
# bare permission string emits `{ "android.permission.RECORD_AUDIO" }` --
# valid CMake, invalid JSON, and androiddeployqt fails with a byte
# offset into a generated file rather than anything naming the property.
# `qt_add_android_permission` builds the pair correctly.
qt_add_android_permission(sstvae_android
    NAME android.permission.RECORD_AUDIO)
qt_add_android_permission(sstvae_android
    NAME android.permission.FOREGROUND_SERVICE)
qt_add_android_permission(sstvae_android
    NAME android.permission.FOREGROUND_SERVICE_MICROPHONE)
# The transmit half. Declared alongside the microphone one because half
# duplex swaps between them within a single service lifetime.
qt_add_android_permission(sstvae_android
    NAME android.permission.FOREGROUND_SERVICE_MEDIA_PLAYBACK)
# API 33+. Denying it costs the ongoing notification but not the
# session, which is why the service never treats it as fatal.
qt_add_android_permission(sstvae_android
    NAME android.permission.POST_NOTIFICATIONS)
