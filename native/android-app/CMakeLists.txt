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

qt_add_executable(sstvae_android main.cpp listener.cpp session.cpp service_jni.cpp)
qt_add_qml_module(sstvae_android URI SSTVAE VERSION 1.0 QML_FILES Main.qml)
target_link_libraries(sstvae_android PRIVATE Qt6::Quick sstvae_core
                                             sstvae_audio_android)

# The audio layer's Java half ships with the layer, not with this app.
set_property(TARGET sstvae_android APPEND PROPERTY
    QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")

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
# API 33+. Denying it costs the ongoing notification but not the
# session, which is why the service never treats it as fatal.
qt_add_android_permission(sstvae_android
    NAME android.permission.POST_NOTIFICATIONS)
