# libheif and libde265, pinned and bundled, for the HEIF/HEIC image plugin.
#
# Same shape and the same reasons as `hamlib.cmake` next door: a pinned
# release tarball with a sha256 per artifact, built once into
# `FETCHCONTENT_BASE_DIR` so it survives a discarded build tree, and
# exposed as an IMPORTED shared library. What differs is that both of
# these are CMake projects, so there is no autotools re-stamping dance --
# the whole module is a configure, a build and an install.
#
# **Dynamically linked, and that is a licensing decision.** libheif and
# libde265 are both LGPL-3.0-or-later. A shared library the operator can
# replace is the posture `docs/native-app.md` decision 6 already takes
# for Qt and `hamlib.cmake` takes for Hamlib; a static link into the
# plugin would make that plugin a combined work and pull in LGPLv3's
# relinking obligations for no gain. So: `BUILD_SHARED_LIBS=ON`, and the
# packaging script ships both `.so`/`.dylib`/`.dll` files beside Qt's.
#
# **Decode only, and that is also a licensing decision -- the sharper
# one.** HEVC *decoding* is libde265, LGPL-3.0-or-later. HEVC *encoding*
# is x265, GPL-2.0-or-later, which would make the whole distribution
# GPLv2+ and is a licence this project cannot satisfy: `NOTICE` records
# that the application icon is licensed artwork not sublicensed to
# recipients, and it is compiled into the executable. So `WITH_X265=OFF`
# is load-bearing rather than a size saving, and the assertion below
# checks libheif's own configuration summary rather than trusting the
# flag -- a silently-enabled encoder is a licence violation that builds
# and ships perfectly.
#
# **What no licence here addresses is patents.** HEVC is covered by
# several patent pools, which is why browsers do not ship HEVC decoders
# and why distributions have historically kept libde265 out of their
# main archives. That risk is a decision for whoever publishes the
# packages; it is not a copyright question and nothing in this file
# changes it. `-DSSTVAE_BUILD_HEIF=OFF` is the switch that declines it,
# and the app then simply reports no HEIC support.
#
# Distro packagers: -DSSTVAE_LIBHEIF_SYSTEM=ON uses pkg-config and
# downloads nothing. Their libheif's codec configuration is then theirs
# to answer for, encoders included.

set(SSTVAE_LIBHEIF_VERSION "1.23.1" CACHE STRING "Bundled libheif version")
set(SSTVAE_LIBDE265_VERSION "1.0.19" CACHE STRING "Bundled libde265 version")
set(SSTVAE_LIBHEIF_SYSTEM OFF CACHE BOOL
    "Use the system libheif via pkg-config instead of the pinned build")
set(SSTVAE_LIBHEIF_DIR "" CACHE PATH
    "Prebuilt libheif tree (containing include/ and lib/); skips the download")

# --- the system option ------------------------------------------------------
if(SSTVAE_LIBHEIF_SYSTEM)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBHEIF REQUIRED IMPORTED_TARGET libheif)
  add_library(sstvae::libheif ALIAS PkgConfig::LIBHEIF)
  set(SSTVAE_LIBHEIF_RUNTIME_DIR "" CACHE INTERNAL "")
  message(STATUS "libheif: system ${LIBHEIF_VERSION} (not the pinned build); "
                 "its codec set, encoders included, is the packager's to answer for")
  return()
endif()

# Release digests for these exact versions. Bumping a version without
# supplying its hash is deliberately impossible, as with onnxruntime and
# Hamlib -- these are published immutable filenames, so a mismatch means
# the wrong file and never a stale one.
set(_heif_known
  "libheif|1.23.1|0de0327f60fcd47de90d5654c6fe152232738d60d84fe084ec3e0f35e03b166a"
  "libde265|1.0.19|bb19a0b485d2643e0eeb7e91f3ab32d1ad617e7c487dbedc91214ca3dbd8d7eb"
)

function(_sstvae_heif_hash project version out)
  foreach(_row IN LISTS _heif_known)
    string(REPLACE "|" ";" _f "${_row}")
    list(GET _f 0 _rp)
    list(GET _f 1 _rv)
    list(GET _f 2 _rh)
    if(_rp STREQUAL project AND _rv STREQUAL version)
      set(${out} "${_rh}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR
    "No sha256 recorded for ${project} ${version}. Add one to "
    "native/cmake/libheif.cmake's _heif_known, or point "
    "-DSSTVAE_LIBHEIF_DIR at a prebuilt tree, or pass "
    "-DSSTVAE_BUILD_HEIF=OFF.")
endfunction()

if(NOT FETCHCONTENT_BASE_DIR)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
endif()
set(_heif_root "${SSTVAE_LIBHEIF_DIR}")

if(NOT _heif_root)
  set(_heif_root
      "${FETCHCONTENT_BASE_DIR}/libheif-install-${SSTVAE_LIBHEIF_VERSION}")

  # What the two builds are asked for, and what must not creep back in.
  # A CMake `if()` on a misspelled option is silently false and a
  # misspelled `-DWITH_*` is silently ignored, so the summary check after
  # the libheif configure is what actually holds the encoder out.
  set(_de265_args
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${_heif_root}"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    # The command-line tools and the SDL viewer, neither of which we ship.
    -DENABLE_DECODER=OFF
    -DENABLE_ENCODER=OFF
    -DENABLE_SDL=OFF)

  set(_heif_args
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${_heif_root}"
    "-DCMAKE_PREFIX_PATH=${_heif_root}"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -DWITH_EXAMPLES=OFF
    # Codecs are compiled in rather than dlopen'd. libheif's plugin
    # loading would mean shipping a second tier of plugins inside our own
    # plugin directory, and a codec that fails to load there is invisible
    # -- the format simply does not work, which is the failure mode this
    # whole change exists to retire.
    -DENABLE_PLUGIN_LOADING=OFF
    -DWITH_LIBDE265=ON
    -DWITH_LIBDE265_PLUGIN=OFF
    # Every encoder off. x265 and kvazaar are the GPL ones; the rest are
    # off because we do not write HEIC and an unused codec is only
    # attack surface and download size.
    -DWITH_X265=OFF
    -DWITH_KVAZAAR=OFF
    -DWITH_UVG266=OFF
    -DWITH_VVENC=OFF
    -DWITH_VVDEC=OFF
    -DWITH_AOM_ENCODER=OFF
    -DWITH_AOM_DECODER=OFF
    -DWITH_RAV1E=OFF
    -DWITH_SvtEnc=OFF
    -DWITH_DAV1D=OFF
    -DWITH_FFMPEG_DECODER=OFF
    -DWITH_JPEG_ENCODER=OFF
    -DWITH_JPEG_DECODER=OFF
    -DWITH_OpenJPEG_ENCODER=OFF
    -DWITH_OpenJPEG_DECODER=OFF
    -DWITH_OPENJPH_ENCODER=OFF
    # libsharpyuv is an RGB->YUV helper used when *writing* YUV, so a
    # decode-only build has no use for it -- and leaving it to
    # autodetection is actively harmful rather than merely wasteful.
    # libheif enables it if it finds one anywhere, which on a macOS runner
    # means Homebrew's: that broke the x86_64 cross build outright
    # (`_SharpYuvConvert` undefined for architecture x86_64, from a host
    # arm64 dylib) and, worse, silently *succeeded* on the arm64 slice --
    # linking an unpinned Homebrew library into something we ship, which
    # is the thing pinning exists to prevent. Off everywhere, so no
    # platform can differ from another.
    -DWITH_LIBSHARPYUV=OFF)

  # **Nothing from the host's package manager, on either macOS slice.**
  # The point of a pinned build is that its inputs are the pinned ones, and
  # a Homebrew prefix on the search path quietly undoes that -- for the
  # cross slice it is a link error, and for the native one it is a
  # dependency that leaves the machine it was built on. The `\;` keeps
  # this one argument rather than two.
  if(APPLE)
    list(APPEND _de265_args "-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew\;/usr/local")
    list(APPEND _heif_args "-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew\;/usr/local")
  endif()

  # libheif needs libde265 at run time, and the two travel together into
  # `lib/`. A relative rpath means that works without LD_LIBRARY_PATH --
  # for the test suite running out of the build tree, and in the packaged
  # app, where the alternative is the packaging script patching an rpath
  # it did not create.
  if(APPLE)
    list(APPEND _heif_args "-DCMAKE_INSTALL_RPATH=@loader_path"
                           -DCMAKE_INSTALL_NAME_DIR=@rpath)
    list(APPEND _de265_args -DCMAKE_INSTALL_NAME_DIR=@rpath)
  elseif(NOT WIN32)
    list(APPEND _heif_args "-DCMAKE_INSTALL_RPATH=$ORIGIN")
  endif()

  # macOS: CMake honours CMAKE_OSX_ARCHITECTURES, so unlike Hamlib's
  # autotools build a cross slice needs nothing but the variable
  # forwarded. Forwarded explicitly all the same -- a nested CMake
  # project inherits nothing from this one.
  if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES)
      list(APPEND _de265_args "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
      list(APPEND _heif_args "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
      list(APPEND _de265_args
           "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
      list(APPEND _heif_args
           "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
  endif()

  # One function for both, because they differ only in name, hash and
  # arguments -- and because the "already built" short circuit has to be
  # identical for both or a warm tree rebuilds half of itself. Measured
  # cold on this machine: libde265 ~20 s, libheif ~55 s. Warm: nothing.
  function(_sstvae_heif_build project version url_template args)
    set(_marker "${_heif_root}/.sstvae-${project}-${version}")
    if(EXISTS "${_marker}")
      return()
    endif()
    _sstvae_heif_hash("${project}" "${version}" _hash)
    string(REPLACE "@VERSION@" "${version}" _url "${url_template}")
    set(_tar "${FETCHCONTENT_BASE_DIR}/${project}-${version}.tar.gz")
    set(_src "${FETCHCONTENT_BASE_DIR}/${project}-src-${version}")

    if(NOT EXISTS "${_tar}")
      message(STATUS "${project}: fetching ${_url}")
      file(DOWNLOAD "${_url}" "${_tar}" EXPECTED_HASH "SHA256=${_hash}"
           STATUS _st SHOW_PROGRESS)
      list(GET _st 0 _code)
      if(NOT _code EQUAL 0)
        file(REMOVE "${_tar}")
        list(GET _st 1 _msg)
        message(FATAL_ERROR "could not fetch ${project} ${version}: ${_msg}")
      endif()
    endif()

    file(REMOVE_RECURSE "${_src}")
    file(MAKE_DIRECTORY "${_src}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tar}"
      WORKING_DIRECTORY "${_src}" RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "could not unpack ${_tar}")
    endif()
    # The tarballs unpack into a single versioned directory; find it
    # rather than assuming its name, since libheif's and libde265's
    # conventions have differed between releases.
    file(GLOB _inner LIST_DIRECTORIES true "${_src}/*")
    foreach(_d IN LISTS _inner)
      if(IS_DIRECTORY "${_d}" AND EXISTS "${_d}/CMakeLists.txt")
        set(_top "${_d}")
      endif()
    endforeach()
    if(NOT _top)
      message(FATAL_ERROR "${_tar} has no CMake project inside it")
    endif()

    message(STATUS "${project}: configuring and building (this happens once)")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -S "${_top}" -B "${_top}/_build"
              -G "${CMAKE_GENERATOR}" ${args}
      OUTPUT_VARIABLE _cfg ERROR_VARIABLE _cfg RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "${project} configure failed:\n${_cfg}")
    endif()
    set(_configure_log "${_cfg}" PARENT_SCOPE)

    cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_LOGICAL_CORES)
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build "${_top}/_build" --parallel ${_jobs}
      OUTPUT_VARIABLE _out ERROR_VARIABLE _out RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "${project} build failed:\n${_out}")
    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} --install "${_top}/_build"
      OUTPUT_VARIABLE _out ERROR_VARIABLE _out RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "${project} install failed:\n${_out}")
    endif()
    file(WRITE "${_marker}" "${version}\n")
  endfunction()

  _sstvae_heif_build(libde265 "${SSTVAE_LIBDE265_VERSION}"
    "https://github.com/strukturag/libde265/releases/download/v@VERSION@/libde265-@VERSION@.tar.gz"
    "${_de265_args}")
  _sstvae_heif_build(libheif "${SSTVAE_LIBHEIF_VERSION}"
    "https://github.com/strukturag/libheif/releases/download/v@VERSION@/libheif-@VERSION@.tar.gz"
    "${_heif_args}")

  # **The encoder check, against libheif's own summary rather than our
  # flags.** `-DWITH_X265=OFF` is only as good as the spelling; libheif
  # prints one line per codec saying what it actually built, so that is
  # what is read. A GPL encoder compiled in would be a licence violation
  # that builds, tests and ships without a symptom.
  if(_configure_log AND _configure_log MATCHES "x265 HEVC encoder[ ]*:[ ]*\\+")
    message(FATAL_ERROR
      "libheif was configured WITH the x265 encoder, which is GPL-2.0-or-later "
      "and cannot be distributed with this application -- see the top of "
      "native/cmake/libheif.cmake. Its configuration summary said:\n"
      "${_configure_log}")
  endif()
endif()

# --- the imported targets ---------------------------------------------------
if(NOT EXISTS "${_heif_root}/include/libheif/heif.h")
  message(FATAL_ERROR
    "libheif headers not found under ${_heif_root}. If this is a prebuilt "
    "tree from -DSSTVAE_LIBHEIF_DIR, it needs include/ and lib/.")
endif()

set(SSTVAE_LIBHEIF_INCLUDE "${_heif_root}/include" CACHE INTERNAL "")

# Both libraries, because the plugin links libheif and libheif needs
# libde265 at run time -- and the packaging script has to ship both. A
# missing transitive dependency fails exactly as invisibly as a direct
# one, which is the lesson `sstvae_hamlib_copy_runtime` records.
#
# `stem` is the library name *without* any platform prefix -- "heif", not
# "libheif". **MSVC emits no `lib` prefix**, so a glob for `libheif*.dll`
# matches nothing while the file sits there called `heif.dll`; the
# patterns below therefore allow the prefix rather than requiring it, and
# they are reported verbatim on failure. The first draft did neither, and
# the result was the worst kind of CI failure: `list GET given empty
# list` from inside the error path, so the message explaining what was
# wrong never ran.
function(_sstvae_heif_import name stem)
  add_library(${name} SHARED IMPORTED GLOBAL)
  if(WIN32)
    # bin/ is where CMake installs a RUNTIME artifact, but glob lib/ too
    # rather than depend on upstream having said so.
    set(_patterns "${_heif_root}/bin/*${stem}*.dll" "${_heif_root}/lib/*${stem}*.dll")
    set(_implib_patterns "${_heif_root}/lib/*${stem}*.lib")
    file(GLOB _found ${_patterns})
    file(GLOB _implib ${_implib_patterns})
    list(APPEND _patterns ${_implib_patterns})
    if(_found AND _implib)
      list(GET _found 0 _loc)
      list(GET _implib 0 _implib0)
      set_target_properties(${name} PROPERTIES
        IMPORTED_LOCATION "${_loc}" IMPORTED_IMPLIB "${_implib0}")
    endif()
  elseif(APPLE)
    set(_patterns "${_heif_root}/lib/lib${stem}.*.dylib"
                  "${_heif_root}/lib/lib${stem}.dylib")
    file(GLOB _found ${_patterns})
    if(_found)
      list(GET _found 0 _loc)
      set_target_properties(${name} PROPERTIES IMPORTED_LOCATION "${_loc}")
    endif()
  else()
    set(_patterns "${_heif_root}/lib/lib${stem}.so.*")
    file(GLOB _found ${_patterns})
    if(_found)
      list(GET _found 0 _loc)
      set_target_properties(${name} PROPERTIES IMPORTED_LOCATION "${_loc}")
    endif()
  endif()
  get_target_property(_loc ${name} IMPORTED_LOCATION)
  if(NOT _loc OR NOT EXISTS "${_loc}")
    string(REPLACE ";" "\n    " _shown "${_patterns}")
    message(FATAL_ERROR
      "${stem} was built or supplied but no shared library was found for it. "
      "Looked for:\n    ${_shown}\n"
      "If the build produced a differently-named file, that is the bug.")
  endif()
  # **The directory the library was actually found in, not an assumption
  # about where it should be.** This was hardcoded to `bin` on Windows
  # while the glob above deliberately searches `bin` *and* `lib`, so a
  # build that installed its DLLs anywhere else would leave this pointing
  # at nothing -- and `package_app.sh` copies `$LIBHEIF_RUNTIME_DIR/*.dll`
  # from it, so the packaged app would silently ship without libheif and
  # lose HEIC with no error anywhere. Derived, so the two cannot disagree.
  get_filename_component(_dir "${_loc}" DIRECTORY)
  set(SSTVAE_LIBHEIF_RUNTIME_DIR "${_dir}" CACHE INTERNAL "" FORCE)
  target_include_directories(${name} INTERFACE "${_heif_root}/include")
endfunction()

_sstvae_heif_import(sstvae_libde265 de265)
_sstvae_heif_import(sstvae_libheif heif)
target_link_libraries(sstvae_libheif INTERFACE sstvae_libde265)
add_library(sstvae::libheif ALIAS sstvae_libheif)

get_target_property(_heif_loc sstvae_libheif IMPORTED_LOCATION)
message(STATUS "libheif: ${_heif_loc} (decode only)")

