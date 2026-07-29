# Hamlib, pinned and bundled rather than taken from the system.
#
# **Why pin.** Hamlib's public API moves between minor releases. Ubuntu
# 24.04 ships 4.5.5, where a configuration token is a `token_t`; 4.6
# renamed it `hamlib_token_t`, so `core/rig/hamlib.cpp` compiles against
# a developer's 4.7 and fails on the CI runner. Chasing that with
# version `#if`s means the app's rig support silently differs by
# platform, which for CAT control means "which radios work" differs by
# platform. One pinned version is the only way that claim stays
# checkable -- the same argument as the onnxruntime pin next door.
#
# **Why bundle.** `docs/native-app.md` (decision 3): users install
# nothing, and there is no per-platform `rigctld` to build and sign. The
# accepted cost is stated there too -- a Hamlib CVE or a new-radio
# backend becomes our release rather than the distro's, so this pin is a
# thing to bump deliberately and regularly.
#
# **Dynamically linked, deliberately.** Hamlib is LGPL-2.1+, so this
# follows the same reasoning as decision 6 for Qt: a shared library the
# user could replace, rather than a static link that would drag in
# LGPL relinking obligations. That is why the Linux/macOS build below
# passes --enable-shared --disable-static rather than the reverse, even
# though a static link would be more convenient to ship.
#
# Distro packagers, and anyone who would rather use the system Hamlib:
# -DSSTVAE_HAMLIB_SYSTEM=ON uses pkg-config and downloads nothing. That
# build is *not* what CI checks, so it is on the packager to confirm
# their Hamlib is new enough (>= 4.6 for hamlib_token_t).

set(SSTVAE_HAMLIB_VERSION "4.7.2" CACHE STRING "Bundled Hamlib version")
set(SSTVAE_HAMLIB_SYSTEM OFF CACHE BOOL
    "Use the system libhamlib via pkg-config instead of the pinned build")
set(SSTVAE_HAMLIB_DIR "" CACHE PATH
    "Prebuilt Hamlib tree (containing include/ and lib/); skips the download")

# --- the system option ------------------------------------------------------
if(SSTVAE_HAMLIB_SYSTEM)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(HAMLIB REQUIRED IMPORTED_TARGET hamlib)
  add_library(sstvae::hamlib ALIAS PkgConfig::HAMLIB)
  set(SSTVAE_HAMLIB_RUNTIME_DIR "" CACHE INTERNAL "")
  message(STATUS "Hamlib: system ${HAMLIB_VERSION} (not the pinned build)")
  return()
endif()

set(_hl_v "${SSTVAE_HAMLIB_VERSION}")

# Release digests for this exact version, as with onnxruntime: bumping
# the version without supplying hashes is deliberately impossible.
set(_hl_known
  "4.7.2|src|ae1fcf2dbc80ea0786ea8f047b09399c3f7737d1930442f61a031708ed33e88f"
  "4.7.2|w64|8553bc6c5c6032e8debf99c017e98f58fed7e07e7c25d04815dc3e8bbe3304c7"
)

function(_sstvae_hamlib_hash kind out)
  foreach(_row IN LISTS _hl_known)
    string(REPLACE "|" ";" _f "${_row}")
    list(GET _f 0 _rv)
    list(GET _f 1 _rk)
    list(GET _f 2 _rh)
    if(_rv STREQUAL SSTVAE_HAMLIB_VERSION AND _rk STREQUAL kind)
      set(${out} "SHA256=${_rh}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR
    "No pinned sha256 for Hamlib ${SSTVAE_HAMLIB_VERSION} (${kind}).\n"
    "Add it to native/cmake/hamlib.cmake, or point -DSSTVAE_HAMLIB_DIR at "
    "a prebuilt tree, or set -DSSTVAE_HAMLIB_SYSTEM=ON.")
endfunction()

include(FetchContent)

if(SSTVAE_HAMLIB_DIR)
  set(_hl_root "${SSTVAE_HAMLIB_DIR}")
  message(STATUS "Hamlib: using ${_hl_root}")

elseif(WIN32)
  # Windows gets the official prebuilt binaries. They are MinGW-built,
  # but the zip ships an MSVC import library alongside the DLL, so
  # linking from MSVC is supported by upstream rather than a trick.
  _sstvae_hamlib_hash("w64" _hl_hash)
  set(_hl_url
      "https://github.com/Hamlib/Hamlib/releases/download/${_hl_v}/hamlib-w64-${_hl_v}.zip")
  message(STATUS "Hamlib: fetching ${_hl_url}")
  FetchContent_Declare(hamlib_prebuilt URL "${_hl_url}" URL_HASH "${_hl_hash}")
  FetchContent_MakeAvailable(hamlib_prebuilt)
  set(_hl_root "${hamlib_prebuilt_SOURCE_DIR}")

else()
  # Linux and macOS build from the release tarball. Upstream publishes
  # no binaries for them, and a distro package is exactly what this file
  # exists to stop depending on.
  _sstvae_hamlib_hash("src" _hl_hash)
  set(_hl_url
      "https://github.com/Hamlib/Hamlib/releases/download/${_hl_v}/hamlib-${_hl_v}.tar.gz")
  message(STATUS "Hamlib: fetching and building ${_hl_url}")

  # MakeAvailable rather than the deprecated bare Populate. Hamlib's
  # tarball has no CMakeLists.txt, so this only unpacks it -- there is
  # no add_subdirectory to go wrong.
  FetchContent_Declare(hamlib_src URL "${_hl_url}" URL_HASH "${_hl_hash}")
  FetchContent_MakeAvailable(hamlib_src)
  # Installed beside the downloaded sources rather than inside the build
  # directory, so that whatever caches FETCHCONTENT_BASE_DIR caches the
  # *built* Hamlib too. CI throws its build tree away every run; without
  # this, every job on every platform would rebuild Hamlib from source.
  if(FETCHCONTENT_BASE_DIR)
    set(_hl_root "${FETCHCONTENT_BASE_DIR}/hamlib-install-${_hl_v}")
  else()
    set(_hl_root "${CMAKE_CURRENT_BINARY_DIR}/hamlib-install")
  endif()

  # Built once and stamped. ExternalProject would rebuild on every
  # configure of a fresh build directory even when the install tree is
  # already there, which is the common case with a warm CI cache.
  if(NOT EXISTS "${_hl_root}/include/hamlib/rig.h")
    # Unpacking a tarball gives every file the same mtime, so make
    # cannot tell that `configure` is already newer than `configure.ac`
    # and tries to re-run aclocal -- which fails on any machine without
    # the exact automake the release was rolled with (1.17 here). Hamlib
    # has no AM_MAINTAINER_MODE to switch that off, so the generated
    # files are re-stamped in dependency order instead. Separate
    # file(TOUCH) calls, not one: each is a later instant than the last,
    # which is what puts the prerequisites genuinely older.
    file(GLOB_RECURSE _hl_am "${hamlib_src_SOURCE_DIR}/*.am")
    file(GLOB _hl_m4 "${hamlib_src_SOURCE_DIR}/macros/*.m4"
                     "${hamlib_src_SOURCE_DIR}/*.m4")
    list(REMOVE_ITEM _hl_m4 "${hamlib_src_SOURCE_DIR}/aclocal.m4")
    file(TOUCH "${hamlib_src_SOURCE_DIR}/configure.ac" ${_hl_am} ${_hl_m4})
    file(TOUCH "${hamlib_src_SOURCE_DIR}/aclocal.m4")
    file(GLOB_RECURSE _hl_hin "${hamlib_src_SOURCE_DIR}/*/config.h.in")
    file(TOUCH "${hamlib_src_SOURCE_DIR}/configure" ${_hl_hin})
    file(GLOB_RECURSE _hl_min "${hamlib_src_SOURCE_DIR}/*/Makefile.in"
                              "${hamlib_src_SOURCE_DIR}/Makefile.in")
    file(TOUCH ${_hl_min})

    message(STATUS "Hamlib: configuring (this happens once)")
    include(ProcessorCount)
    ProcessorCount(_hl_jobs)
    if(_hl_jobs EQUAL 0)
      set(_hl_jobs 2)
    endif()

    # Only what this app uses. The C++ binding, the daemons and the
    # tools are all built by default and none of them are linked here;
    # skipping them is most of the build time. libusb is left off so
    # the build needs no system -dev packages -- it costs the handful
    # of USB-only backends, which is a trade to revisit if a user asks.
    set(_hl_configure_args
        "--prefix=${_hl_root}"
        "--enable-shared"
        "--disable-static"
        "--without-cxx-binding"
        "--without-libusb"
        "--disable-silent-rules")

    execute_process(
      COMMAND "${hamlib_src_SOURCE_DIR}/configure" ${_hl_configure_args}
      WORKING_DIRECTORY "${hamlib_src_BINARY_DIR}"
      RESULT_VARIABLE _hl_rc
      OUTPUT_FILE "${hamlib_src_BINARY_DIR}/configure.log"
      ERROR_FILE "${hamlib_src_BINARY_DIR}/configure.log")
    if(NOT _hl_rc EQUAL 0)
      message(FATAL_ERROR
        "Hamlib configure failed (${_hl_rc}); see "
        "${hamlib_src_BINARY_DIR}/configure.log")
    endif()

    message(STATUS "Hamlib: building with ${_hl_jobs} jobs")
    execute_process(
      COMMAND make -j${_hl_jobs} install
      WORKING_DIRECTORY "${hamlib_src_BINARY_DIR}"
      RESULT_VARIABLE _hl_rc
      OUTPUT_FILE "${hamlib_src_BINARY_DIR}/build.log"
      ERROR_FILE "${hamlib_src_BINARY_DIR}/build.log")
    if(NOT _hl_rc EQUAL 0)
      message(FATAL_ERROR
        "Hamlib build failed (${_hl_rc}); see ${hamlib_src_BINARY_DIR}/build.log")
    endif()
  else()
    message(STATUS "Hamlib: reusing the build in ${_hl_root}")
  endif()
endif()

if(NOT EXISTS "${_hl_root}/include/hamlib/rig.h")
  message(FATAL_ERROR
    "Hamlib was not unpacked/built into ${_hl_root}.\n"
    "If this is a cached build, the dependency tree is inconsistent -- "
    "delete FETCHCONTENT_BASE_DIR and configure again; it is only "
    "meaningful as a unit.")
endif()

add_library(sstvae_hamlib SHARED IMPORTED GLOBAL)
add_library(sstvae::hamlib ALIAS sstvae_hamlib)
set_target_properties(sstvae_hamlib PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_hl_root}/include")

if(WIN32)
  set_target_properties(sstvae_hamlib PROPERTIES
    IMPORTED_LOCATION "${_hl_root}/bin/libhamlib-4.dll"
    IMPORTED_IMPLIB "${_hl_root}/lib/gcc/libhamlib-4.lib")
  set(SSTVAE_HAMLIB_RUNTIME_DIR "${_hl_root}/bin" CACHE INTERNAL "")
elseif(APPLE)
  # libtool's current:revision:age for Hamlib gives libhamlib.4.dylib,
  # but that mapping is not something to bet a CI run on from a machine
  # that cannot check it -- so take whatever versioned dylib is actually
  # there and fall back to the unversioned symlink.
  file(GLOB _hl_dylib "${_hl_root}/lib/libhamlib.*.dylib")
  list(SORT _hl_dylib)
  if(_hl_dylib)
    list(GET _hl_dylib 0 _hl_dylib)
  else()
    set(_hl_dylib "${_hl_root}/lib/libhamlib.dylib")
  endif()
  set_target_properties(sstvae_hamlib PROPERTIES IMPORTED_LOCATION "${_hl_dylib}")
  set(SSTVAE_HAMLIB_RUNTIME_DIR "${_hl_root}/lib" CACHE INTERNAL "")
else()
  file(GLOB _hl_so "${_hl_root}/lib/libhamlib.so.[0-9]")
  list(SORT _hl_so)
  if(_hl_so)
    list(GET _hl_so 0 _hl_so)
  else()
    set(_hl_so "${_hl_root}/lib/libhamlib.so")
  endif()
  get_filename_component(_hl_soname "${_hl_so}" NAME)
  set_target_properties(sstvae_hamlib PROPERTIES
    IMPORTED_LOCATION "${_hl_so}"
    IMPORTED_SONAME "${_hl_soname}")
  set(SSTVAE_HAMLIB_RUNTIME_DIR "${_hl_root}/lib" CACHE INTERNAL "")
endif()

get_target_property(_hl_loc sstvae_hamlib IMPORTED_LOCATION)
if(NOT EXISTS "${_hl_loc}")
  message(FATAL_ERROR "Hamlib library not found at ${_hl_loc}")
endif()
message(STATUS "Hamlib: ${_hl_loc}")

set(SSTVAE_HAMLIB_ROOT "${_hl_root}" CACHE INTERNAL "")
