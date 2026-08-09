# onnxruntime, fetched as an official prebuilt binary.
#
# Building onnxruntime from source takes hours and pulls in its own
# dependency tree; the project publishes per-platform CPU archives that
# are 9-80 MB and contain exactly what a consumer needs (headers, one
# shared library, a CMake config package). We use those.
#
# **One platform is deliberately an exception**: see
# SSTVAE_ONNXRUNTIME_VERSION_OSX_X86_64 below.
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

# Intel macOS, and only Intel macOS, stays on an older runtime.
#
# onnxruntime stopped publishing a macOS x86_64 build between 1.22 and
# 1.26 -- there is no tgz and no wheel for the current pin, so an Intel
# Mac cannot be served by it at any price. 1.22.0 is the last release
# that has one.
#
# This *does* break the "same version, two builds" basis of the codec
# parity claim, and only for this artifact. Accepted deliberately
# (Andrew, 2026-07-29): what has to match between two stations is the
# *model*, which is published and identical; a runtime version
# difference lands as noise underneath the channel's, in the same way
# `docs/onnx.md` measures quantisation doing. The parity tests are
# stronger than the on-air requirement, so a small drift confined to one
# legacy artifact is a fair trade for a station that would otherwise
# have no app at all -- and the build can be *labelled* as the lower
# compatibility tier, the way the int8 artifacts are.
#
# The arm64 macOS build is untouched and its parity claim stays exact.
# Raise this pin the moment onnxruntime publishes x86_64 again; delete
# it, and Intel support, if Apple's own support ends first.
set(SSTVAE_ONNXRUNTIME_VERSION_OSX_X86_64 "1.22.0" CACHE STRING
    "onnxruntime version for Intel macOS, which the main pin has no build for")
set(SSTVAE_ONNXRUNTIME_DIR "" CACHE PATH
    "Unpacked onnxruntime distribution; skips the download when set")

function(_sstvae_onnxruntime_archive out_url out_hash out_ext out_version)
  set(_v "${SSTVAE_ONNXRUNTIME_VERSION}")
  # Hashes are the GitHub release digests for this exact version. A new
  # version means new hashes -- there is deliberately no way to bump the
  # version without also supplying them, because an unpinned download of
  # a binary that runs on every received picture is not something to
  # leave to the network.
  set(_known
    "1.22.0|osx-x86_64|tgz|e4ec94a7696de74fb1b12846569aa94e499958af6ffa186022cfde16c9d617f0"
    "1.28.0|linux-x64|tgz|a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407"
    "1.28.0|linux-aarch64|tgz|e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb"
    "1.28.0|osx-arm64|tgz|1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8"
    "1.28.0|win-x64|zip|abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc"
    "1.28.0|win-arm64|zip|cbe4547463ece092b505c3581376ed5896d22b5429f39d5e645e425ecdd369ad"
  )

  # Platform slug.
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

  # Intel macOS is spelled `osx-x86_64` by the release that has it, not
  # `osx-x64` like every other x86 archive, and it carries its own
  # version. Both exceptions are here rather than in the table so the
  # table stays a plain list of pinned digests.
  if(_os STREQUAL "osx" AND _arch STREQUAL "x64")
    set(_slug "osx-x86_64")
    set(_v "${SSTVAE_ONNXRUNTIME_VERSION_OSX_X86_64}")
  endif()
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
      # Reported back because it is not always
      # SSTVAE_ONNXRUNTIME_VERSION: Intel macOS pins its own, and the
      # library's filename carries the version, so a caller that assumes
      # the default looks for a file that was never downloaded.
      set(${out_version} "${_v}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR
    "No pinned onnxruntime ${_v} archive for '${_slug}'.\n"
    "Either add it (with its sha256) to native/cmake/onnxruntime.cmake, "
    "or point -DSSTVAE_ONNXRUNTIME_DIR at an unpacked distribution.")
endfunction()

# The version actually in use. Only differs from the pin on the
# platforms that have their own (Intel macOS), and is left at the pin for
# an unpacked distribution, whose version we cannot know.
set(_ort_version "${SSTVAE_ONNXRUNTIME_VERSION}")

# Android is served by the Maven AAR rather than a GitHub release
# archive, and the AAR's layout is its own: `headers/` where every other
# platform has `include/`, and `jni/<abi>/libonnxruntime.so` where they
# have `lib/`. Hence a branch rather than another row in the table.
#
# The important part is that the **version is the same pin**. onnxruntime
# publishes the Android AAR for 1.28.0, so the codec's "same runtime
# version, two builds" basis -- the thing the bit-identical encoder and
# byte-identical decoder claims rest on -- holds here exactly. That is a
# better position than Intel macOS, which is the one platform that had to
# give it up.
#
# This is the full AAR, not `onnxruntime-mobile`: the reduced build ships
# a trimmed operator set and would need the published graphs re-exported,
# which is precisely the divergence the pin exists to prevent.
set(_ort_is_aar OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT SSTVAE_ONNXRUNTIME_DIR)
  set(_ort_is_aar ON)
  set(_ort_aar_sha256
      "f351a0638696f54b35184290dbc001d66daae17281ad0b548d2c70347d53b8a9")
  set(_ort_aar_url
      "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${_ort_version}/onnxruntime-android-${_ort_version}.aar")
  set(_ort_root "${CMAKE_BINARY_DIR}/onnxruntime-android-${_ort_version}")
  # Downloaded and extracted by hand rather than through FetchContent,
  # which picks its extractor from the file extension and does not know
  # `.aar` (it is a zip). Naming the format is one less thing to be
  # surprised by than relying on that guess.
  if(NOT EXISTS "${_ort_root}/headers/onnxruntime_cxx_api.h")
    set(_ort_aar "${CMAKE_BINARY_DIR}/onnxruntime-android-${_ort_version}.aar")
    if(NOT EXISTS "${_ort_aar}")
      message(STATUS "onnxruntime: fetching ${_ort_aar_url}")
      file(DOWNLOAD "${_ort_aar_url}" "${_ort_aar}"
           EXPECTED_HASH "SHA256=${_ort_aar_sha256}" SHOW_PROGRESS STATUS _dl)
      list(GET _dl 0 _dl_code)
      if(NOT _dl_code EQUAL 0)
        list(GET _dl 1 _dl_msg)
        file(REMOVE "${_ort_aar}")
        message(FATAL_ERROR "onnxruntime AAR download failed: ${_dl_msg}")
      endif()
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_ort_aar}" DESTINATION "${_ort_root}")
  endif()
  message(STATUS "onnxruntime: ${_ort_root} (AAR, ${ANDROID_ABI})")
endif()

if(_ort_is_aar)
  # Nothing below this point applies: the AAR's paths are set here.
elseif(SSTVAE_ONNXRUNTIME_DIR)
  set(_ort_root "${SSTVAE_ONNXRUNTIME_DIR}")
  message(STATUS "onnxruntime: using ${_ort_root}")
else()
  _sstvae_onnxruntime_archive(_ort_url _ort_hash _ort_ext _ort_version)
  message(STATUS "onnxruntime: fetching ${_ort_url}")
  include(FetchContent)
  FetchContent_Declare(onnxruntime_prebuilt URL "${_ort_url}" URL_HASH "${_ort_hash}")
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(_ort_root "${onnxruntime_prebuilt_SOURCE_DIR}")
endif()

if(_ort_is_aar)
  set(_ort_include "${_ort_root}/headers")
else()
  set(_ort_include "${_ort_root}/include")
endif()

if(NOT EXISTS "${_ort_include}/onnxruntime_cxx_api.h")
  if(_ort_is_aar)
    message(FATAL_ERROR
      "onnxruntime AAR at ${_ort_root} has no headers/onnxruntime_cxx_api.h. "
      "Delete it and configure again; a partial extraction is the likely cause.")
  endif()
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
  INTERFACE_INCLUDE_DIRECTORIES "${_ort_include}")

if(_ort_is_aar)
  # One unversioned .so per ABI, and ANDROID_ABI is what says which. Get
  # this wrong and the link succeeds against the wrong architecture only
  # if the file happens to exist, so name it explicitly and check.
  set(_ort_lib "${_ort_root}/jni/${ANDROID_ABI}/libonnxruntime.so")
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_lib}"
    IMPORTED_SONAME "libonnxruntime.so")
elseif(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${_ort_root}/lib/onnxruntime.lib")
else()
  # The versioned filename, then whatever versioned library is actually
  # there. The fallback is what makes an unpacked distribution
  # (SSTVAE_ONNXRUNTIME_DIR) work at any version rather than only at the
  # pinned one -- and it is the same mistake that made a platform with
  # its own pin look for a file it had never downloaded.
  if(APPLE)
    set(_ort_lib "${_ort_root}/lib/libonnxruntime.${_ort_version}.dylib")
    set(_ort_glob "${_ort_root}/lib/libonnxruntime.*.dylib")
  else()
    set(_ort_lib "${_ort_root}/lib/libonnxruntime.so.${_ort_version}")
    set(_ort_glob "${_ort_root}/lib/libonnxruntime.so.[0-9]*")
  endif()
  if(NOT EXISTS "${_ort_lib}")
    file(GLOB _ort_found "${_ort_glob}")
    list(SORT _ort_found)
    if(_ort_found)
      list(GET _ort_found 0 _ort_lib)
    endif()
  endif()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${_ort_lib}")
  if(NOT APPLE)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_SONAME "libonnxruntime.so.1")
  endif()
endif()

get_target_property(_ort_loc onnxruntime IMPORTED_LOCATION)
if(NOT EXISTS "${_ort_loc}")
  message(FATAL_ERROR "onnxruntime library not found at ${_ort_loc}")
endif()

# Tests and the app both need the shared library findable at run time.
# ROOT is consumed as an include path by sstvae_codec, so on Android it
# has to be the AAR's `headers` rather than its top level.
set(SSTVAE_ONNXRUNTIME_ROOT "${_ort_root}" CACHE INTERNAL "")
if(_ort_is_aar)
  get_filename_component(_ort_libdir "${_ort_lib}" DIRECTORY)
  set(SSTVAE_ONNXRUNTIME_LIBDIR "${_ort_libdir}" CACHE INTERNAL "")
  set(SSTVAE_ONNXRUNTIME_INCLUDE "${_ort_include}" CACHE INTERNAL "")
else()
  set(SSTVAE_ONNXRUNTIME_LIBDIR "${_ort_root}/lib" CACHE INTERNAL "")
  set(SSTVAE_ONNXRUNTIME_INCLUDE "${_ort_root}/include" CACHE INTERNAL "")
endif()
