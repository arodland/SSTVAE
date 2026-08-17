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

project(sstvae_android VERSION 0.4.2 LANGUAGES CXX)

set(SSTVAE_ANDROID_VERSION_CODE 1 CACHE STRING
    "Play upload version code; must increase on every upload")

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
    assets.cpp waterfall.cpp pictures.cpp composition.cpp transmitter.cpp)
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

# **The model artifacts ship inside the APK** (assets.hpp says why: the
# model is part of the on-air contract, and first run has to work with
# no network). Downloaded at configure time and pinned by sha256, the
# same shape as the onnxruntime and Hamlib pins -- these are published
# immutable filenames, so a hash mismatch means the wrong file, never a
# stale one.
#
# `assets/` inside the package source dir, **not** a Qt resource:
# a .qrc is compiled into the app library, which the bundle carries once
# per ABI, so 20 MB of weights would land in the download twice. Assets
# live in the bundle's base module and are shared by every ABI.
#
# OFF is a supported configuration and needs no other change: with no
# assets staged, `assets::model_blob` returns nullopt, the codec falls
# through to `checkpoint::resolve_onnx`, and the app fetches exactly as
# it did before. That is what makes this switch cheap to flip while
# developing, where 20 MB per clean build tree is not free.
option(SSTVAE_ANDROID_BUNDLE_MODELS
       "Ship the ONNX codec inside the APK instead of fetching it" ON)
if(SSTVAE_ANDROID_BUNDLE_MODELS)
    # Pinned to `checkpoint::DEFAULT_REVISION` and
    # `DEFAULT_PRECISION`; the app looks the filenames up through
    # `checkpoint::onnx_filename`, so a revision bump is one edit there
    # plus the two hashes here, and a mismatch fails the build rather
    # than shipping an artifact the code will not ask for.
    # **A multi-ABI build configures this file once per ABI**, in nested
    # build directories, so a download location under `CMAKE_BINARY_DIR`
    # is fetched once per ABI -- which onnxruntime.cmake accepts for a
    # small AAR and is 40 MB here. Hence a shared location that
    # `tools/build_android.sh` pins to the *top* build directory;
    # pointing it at an already-populated directory (a Hub cache, a USB
    # stick) skips the network entirely.
    set(SSTVAE_ANDROID_MODEL_DIR "${CMAKE_BINARY_DIR}/sstvae-models" CACHE PATH
        "Where the bundled ONNX artifacts are downloaded to and read from")
    # **A cache variable does not reach the per-ABI builds by itself.**
    # Qt configures each extra ABI as a separate CMake project and
    # forwards only what this list names, so without it the secondary
    # ABI re-downloads all 20 MB into its own nested build directory --
    # which is exactly what it did, visibly, on the first two-ABI run.
    list(APPEND QT_ANDROID_MULTI_ABI_FORWARD_VARS SSTVAE_ANDROID_MODEL_DIR)
    set(_model_base "https://huggingface.co/arodland/sstvae/resolve/main")
    # `|` rather than `;` between name and hash: `foreach(IN LISTS)`
    # splits on semicolons, so a list of "a;b" pairs arrives flattened
    # into single items and the hash silently goes missing.
    foreach(_m IN ITEMS
        "v4-decoder-fp16.onnx|e82404514c822d394c3315c0a4d4ce3df05654e7c90d7e18296f34c52f44a006"
        "v4-encoder-fp16.onnx|15b0e072f9fe52282f2721209c2846f4d0f0b14474b7a399934bb3448b2a933d")
        string(REPLACE "|" ";" _pair "${_m}")
        list(GET _pair 0 _name)
        list(GET _pair 1 _sha)
        set(_dl "${SSTVAE_ANDROID_MODEL_DIR}/${_name}")
        if(NOT EXISTS "${_dl}")
            message(STATUS "Fetching model artifact ${_name}")
            file(DOWNLOAD "${_model_base}/${_name}" "${_dl}"
                 EXPECTED_HASH SHA256=${_sha}
                 SHOW_PROGRESS STATUS _st)
            list(GET _st 0 _code)
            if(NOT _code EQUAL 0)
                file(REMOVE "${_dl}")
                list(GET _st 1 _msg)
                message(FATAL_ERROR
                    "could not fetch ${_name}: ${_msg}\n"
                    "Build with -DSSTVAE_ANDROID_BUNDLE_MODELS=OFF to fall back "
                    "to fetching the model at run time.")
            endif()
        endif()
        configure_file("${_dl}" "${_pkg_dir}/assets/models/${_name}" COPYONLY)
    endforeach()
endif()
set_property(TARGET sstvae_android APPEND PROPERTY
    QT_ANDROID_PACKAGE_SOURCE_DIR "${_pkg_dir}")

set_target_properties(sstvae_android PROPERTIES
    QT_ANDROID_PACKAGE_NAME "org.cleverdomain.sstvae"
    QT_ANDROID_APP_NAME "SSTVAE"
    QT_ANDROID_TARGET_SDK_VERSION 36
    # **29, because of the gallery export.** Inserting into MediaStore
    # with `RELATIVE_PATH` and `IS_PENDING` is API 29; the API 28 way is
    # `WRITE_EXTERNAL_STORAGE`, a public-directory write and a
    # `MediaScannerConnection` scan -- a second implementation of the
    # same feature, and a runtime storage prompt on every install, to
    # support one API level from 2018. Qt 6.8 itself requires 28, so
    # this gives up a single version.
    QT_ANDROID_MIN_SDK_VERSION 29
    QT_ANDROID_VERSION_NAME "${PROJECT_VERSION}"
    # **Play requires this to increase on every upload, forever**, and it
    # is the one number a rebuild cannot infer: `PROJECT_VERSION` moves
    # for reasons of its own and a version that ships twice (a fixed
    # build of the same release) still needs a new code. So it stays an
    # explicit input -- `--version-code N` on the build script -- with 1
    # as the first upload. Getting it wrong is not dangerous, only
    # tedious: Play refuses the bundle at upload and says which number it
    # already has.
    QT_ANDROID_VERSION_CODE "${SSTVAE_ANDROID_VERSION_CODE}")

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
