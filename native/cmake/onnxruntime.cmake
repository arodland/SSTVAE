# onnxruntime, fetched as an official prebuilt binary.
#
# Building onnxruntime from source takes hours and pulls in its own
# dependency tree; the project publishes per-platform CPU archives that
# are 9-80 MB and contain exactly what a consumer needs (headers, one
# shared library, a CMake config package). We use those.
#
# **The version is pinned to the same one the Python package resolves**,
# and that is not incidental. `docs/onnx.md` measures fp16 as identical
# to fp32 end to end, and the C++/Python spike measured the encoder
# bit-identical and the decoder byte-identical on the published fp16
# artifacts -- but "identical" is a statement about two builds of the
# *same* runtime version. Two versions could differ by a kernel rewrite
# and still both be correct, and the difference would show up as a
# picture that is subtly not the one the sender encoded. Bump this in
# step with pyproject.toml's onnxruntime pin, not ahead of it.
#
# Offline / distro builds: set SSTVAE_ONNXRUNTIME_DIR to an unpacked
# distribution (the directory containing include/ and lib/) and nothing
# is downloaded.

set(SSTVAE_ONNXRUNTIME_VERSION "1.28.0" CACHE STRING
    "onnxruntime version; keep in step with the Python pin")
set(SSTVAE_ONNXRUNTIME_DIR "" CACHE PATH
    "Unpacked onnxruntime distribution; skips the download when set")

function(_sstvae_onnxruntime_archive out_url out_hash out_ext)
  set(_v "${SSTVAE_ONNXRUNTIME_VERSION}")
  # Hashes are the GitHub release digests for this exact version. A new
  # version means new hashes -- there is deliberately no way to bump the
  # version without also supplying them, because an unpinned download of
  # a binary that runs on every received picture is not something to
  # leave to the network.
  set(_known
    "1.28.0|linux-x64|tgz|a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407"
    "1.28.0|linux-aarch64|tgz|e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb"
    "1.28.0|osx-arm64|tgz|1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8"
    "1.28.0|win-x64|zip|abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc"
    "1.28.0|win-arm64|zip|cbe4547463ece092b505c3581376ed5896d22b5429f39d5e645e425ecdd369ad"
  )

  # Platform slug. Note there is no osx-x64 archive any more: the macOS
  # build is Apple silicon only, which is also why CI dropped macos-13.
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_os "win")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_os "osx")
  else()
    set(_os "linux")
  endif()

  set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(_arch "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  string(TOLOWER "${_arch}" _arch)
  if(_arch MATCHES "^(x86_64|amd64)$")
    set(_arch "x64")
  elseif(_arch MATCHES "^(aarch64|arm64)$")
    if(_os STREQUAL "linux")
      set(_arch "aarch64")
    else()
      set(_arch "arm64")
    endif()
  endif()

  set(_slug "${_os}-${_arch}")
  foreach(_row IN LISTS _known)
    string(REPLACE "|" ";" _f "${_row}")
    list(GET _f 0 _rv)
    list(GET _f 1 _rs)
    list(GET _f 2 _re)
    list(GET _f 3 _rh)
    if(_rv STREQUAL _v AND _rs STREQUAL _slug)
      set(${out_url}
          "https://github.com/microsoft/onnxruntime/releases/download/v${_v}/onnxruntime-${_slug}-${_v}.${_re}"
          PARENT_SCOPE)
      set(${out_hash} "SHA256=${_rh}" PARENT_SCOPE)
      set(${out_ext} "${_re}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR
    "No pinned onnxruntime ${_v} archive for '${_slug}'.\n"
    "Either add it (with its sha256) to native/cmake/onnxruntime.cmake, "
    "or point -DSSTVAE_ONNXRUNTIME_DIR at an unpacked distribution.")
endfunction()

if(SSTVAE_ONNXRUNTIME_DIR)
  set(_ort_root "${SSTVAE_ONNXRUNTIME_DIR}")
  message(STATUS "onnxruntime: using ${_ort_root}")
else()
  _sstvae_onnxruntime_archive(_ort_url _ort_hash _ort_ext)
  message(STATUS "onnxruntime: fetching ${_ort_url}")
  include(FetchContent)
  FetchContent_Declare(onnxruntime_prebuilt URL "${_ort_url}" URL_HASH "${_ort_hash}")
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(_ort_root "${onnxruntime_prebuilt_SOURCE_DIR}")
endif()

if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
  if(SSTVAE_ONNXRUNTIME_DIR)
    message(FATAL_ERROR
      "onnxruntime at ${_ort_root} has no include/onnxruntime_cxx_api.h; "
      "SSTVAE_ONNXRUNTIME_DIR should be the directory containing include/ and lib/.")
  endif()
  # The download reported success but left nothing behind. In practice
  # this means a *partially restored* FetchContent tree: the `-subbuild`
  # directory carries stamp files saying the archive was already
  # extracted, so nothing re-extracts, while the extracted `-src` is
  # missing. A CI cache that saved one without the other did exactly
  # this and failed on three platforms at once.
  message(FATAL_ERROR
    "onnxruntime was not extracted to ${_ort_root}.\n"
    "If this is a cached build, the FetchContent tree is inconsistent: "
    "its stamp files say the archive is already extracted but the "
    "extracted content is gone. Delete the whole _deps (or "
    "FETCHCONTENT_BASE_DIR) tree and configure again -- it is only "
    "meaningful as a unit, so anything caching it must cache all of it.")
endif()

# The archives ship a CMake config package, but it names an imported
# target whose location has moved between releases. An imported target
# built here is one thing to keep working rather than two.
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include")

if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${_ort_root}/lib/onnxruntime.lib")
elseif(APPLE)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_root}/lib/libonnxruntime.${SSTVAE_ONNXRUNTIME_VERSION}.dylib")
else()
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_root}/lib/libonnxruntime.so.${SSTVAE_ONNXRUNTIME_VERSION}"
    IMPORTED_SONAME "libonnxruntime.so.1")
endif()

get_target_property(_ort_loc onnxruntime IMPORTED_LOCATION)
if(NOT EXISTS "${_ort_loc}")
  message(FATAL_ERROR "onnxruntime library not found at ${_ort_loc}")
endif()

# Tests and the app both need the shared library findable at run time.
set(SSTVAE_ONNXRUNTIME_ROOT "${_ort_root}" CACHE INTERNAL "")
set(SSTVAE_ONNXRUNTIME_LIBDIR "${_ort_root}/lib" CACHE INTERNAL "")
