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
set(SSTVAE_ANDROID_API "" CACHE STRING
    "Android API level for the Hamlib cross-build; empty = detect")

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
  #
  # On Android the install tree is per ABI: two ABIs are two builds of
  # two different libraries, and sharing a prefix would have the second
  # find the first's headers, skip its own build, and link an arm64
  # library into an armeabi-v7a APK.
  set(_hl_suffix "")
  if(ANDROID)
    set(_hl_suffix "-${CMAKE_ANDROID_ARCH_ABI}")
  endif()
  if(FETCHCONTENT_BASE_DIR)
    set(_hl_root "${FETCHCONTENT_BASE_DIR}/hamlib-install-${_hl_v}${_hl_suffix}")
  else()
    set(_hl_root "${CMAKE_CURRENT_BINARY_DIR}/hamlib-install${_hl_suffix}")
  endif()

  # Built once and stamped. ExternalProject would rebuild on every
  # configure of a fresh build directory even when the install tree is
  # already there, which is the common case with a warm CI cache.
  #
  # **The stamp is the library, not just the header.** `SUBDIRS` puts
  # `include` second (Makefile.am:25), so `make install` copies
  # `hamlib/rig.h` into the prefix long before it reaches `src` -- which
  # means a build that fails anywhere after that leaves a tree this
  # check would accept. The next configure then skips the rebuild
  # entirely and fails much later with "Hamlib library not found",
  # naming a path rather than the compile error that actually stopped
  # it. Requiring both is what makes a failed build retry instead of
  # cementing itself.
  set(_hl_built FALSE)
  if(EXISTS "${_hl_root}/include/hamlib/rig.h")
    file(GLOB _hl_installed "${_hl_root}/lib/libhamlib*")
    if(_hl_installed)
      set(_hl_built TRUE)
    endif()
  endif()

  if(NOT _hl_built)
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

    # Cross-compiling for Android.
    #
    # **Unverified as of 2026-08-22**: this session had no NDK and
    # dl.google.com is unreachable from it, so the recipe below is
    # written from the NDK's documented layout and has never been run.
    # Treat a first failure as expected work rather than as a bug, and
    # see docs/android.md ("Rig control over a socket") for what it is
    # for. Everything downstream of it -- the bridge, the transport, the
    # PTT routing -- is covered by tests that run without it.
    #
    # Three things here are Android-specific and none of them are
    # optional:
    #
    #   * **`--host` and the compiler triple are not the same string on
    #     32-bit ARM.** autotools wants `arm-linux-androideabi`, which is
    #     what its config.sub knows; the NDK's clang wrapper is
    #     `armv7a-linux-androideabi<api>-clang`. Using either name for
    #     both fails -- one at configure, one at the first compile.
    #   * **The library must come out unversioned, and the flag for that
    #     is libtool's, not the linker's.** Otherwise libtool produces
    #     `libhamlib.so.4.0.7` with `libhamlib.so` a symlink to it, and
    #     Android's packager takes only files named exactly `lib*.so`
    #     from `libs/<abi>/` -- a versioned SONAME is silently not
    #     installed and the app dies at `dlopen` before reaching main.
    #     `-avoid-version` is what suppresses it, and putting it in
    #     `LDFLAGS` here is what the first attempt did: configure's very
    #     first link test runs the compiler *directly*, clang rejects an
    #     argument it has never heard of, and the whole thing stops at
    #     "C compiler cannot create executables". It is applied at make
    #     time instead, to the one automake variable that carries
    #     `-version-info` -- see the build step below.
    #   * **The malloc probes have to be answered.** configure cannot run
    #     what it just built, so `AC_FUNC_MALLOC` guesses "no" and
    #     substitutes its own `rpl_malloc`, which does not exist here.
    #   * **`CC` alone is not enough**, and the Android sensor *rotator*
    #     is what proves it -- it is C++, it cannot be switched off, and
    #     with no `CXX` the host g++ gets it. See the arguments below.
    #
    # Deliberately still `--enable-shared`: Hamlib is LGPL-2.1+ and the
    # reasoning at the top of this file does not change because the
    # target is a phone. An APK with a static Hamlib inside would carry
    # the relinking obligation.
    if(ANDROID)
      if(NOT CMAKE_ANDROID_NDK)
        message(FATAL_ERROR
          "Hamlib for Android needs CMAKE_ANDROID_NDK. Configure through "
          "Qt's android toolchain file, or set -DSSTVAE_BUILD_RIG=OFF.")
      endif()
      if(CMAKE_HOST_APPLE)
        set(_ndk_host "darwin-x86_64")
      elseif(CMAKE_HOST_WIN32)
        set(_ndk_host "windows-x86_64")
      else()
        set(_ndk_host "linux-x86_64")
      endif()
      set(_ndk_bin "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/${_ndk_host}/bin")

      # **The API level is not reliably in `CMAKE_SYSTEM_VERSION`**, and
      # believing it was is what made this block's first real run fail.
      # CMake defaults that variable to `1` when cross-compiling and
      # nothing sets it, which the NDK toolchain path does not always
      # do -- so the compiler wrapper came out as
      # `x86_64-linux-android1-clang`. Worse, the fallback written to
      # catch exactly that was `if(NOT _hl_api)`, and `1` is *true* in
      # CMake, so it never ran. A guard whose condition cannot fire is
      # not a guard.
      #
      # Ask everything that might know, in order of how directly it
      # means "API level", and take the first plausible answer.
      # `SSTVAE_ANDROID_API` is first so a build with an NDK layout
      # nobody here anticipated can be unblocked with one -D.
      set(_hl_api "")
      set(_hl_api_from "")
      foreach(_src SSTVAE_ANDROID_API CMAKE_ANDROID_API ANDROID_NATIVE_API_LEVEL
                   ANDROID_PLATFORM_LEVEL CMAKE_SYSTEM_VERSION)
        if(DEFINED ${_src} AND "${${_src}}" MATCHES "^[0-9]+$"
           AND "${${_src}}" GREATER_EQUAL 16)
          set(_hl_api "${${_src}}")
          set(_hl_api_from "${_src}")
          break()
        endif()
      endforeach()
      # The NDK's own spelling, "android-29".
      if(_hl_api STREQUAL "" AND DEFINED ANDROID_PLATFORM
         AND "${ANDROID_PLATFORM}" MATCHES "([0-9]+)")
        set(_hl_api "${CMAKE_MATCH_1}")
        set(_hl_api_from "ANDROID_PLATFORM")
      endif()
      if(_hl_api STREQUAL "")
        # The floor of every NDK that still exists, and safe against the
        # app's minSdk of 29: a library built for a *lower* level loads
        # on a higher one, never the reverse. Guessing high is the
        # failure that would only show up on somebody else's phone.
        set(_hl_api 21)
        set(_hl_api_from "fallback")
      endif()

      if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        set(_hl_cc_triple "aarch64-linux-android")
        set(_hl_triple "aarch64-linux-android")
      elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
        set(_hl_cc_triple "armv7a-linux-androideabi")
        set(_hl_triple "arm-linux-androideabi")
      elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
        set(_hl_cc_triple "x86_64-linux-android")
        set(_hl_triple "x86_64-linux-android")
      elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
        set(_hl_cc_triple "i686-linux-android")
        set(_hl_triple "i686-linux-android")
      else()
        message(FATAL_ERROR
          "Hamlib for Android: unknown ABI '${CMAKE_ANDROID_ARCH_ABI}'")
      endif()

      # The NDK ships one wrapper per API level, and which levels exist
      # depends on the NDK release -- r28 dropped everything below 21.
      # So a level that is right for the *app* can still have no wrapper
      # here. Rather than refuse, take the lowest one at or above it:
      # that is the same library, built against an older libc, and it
      # runs everywhere the requested one would have.
      set(_hl_cc "${_ndk_bin}/${_hl_cc_triple}${_hl_api}-clang")
      if(NOT EXISTS "${_hl_cc}")
        file(GLOB _hl_wrappers "${_ndk_bin}/${_hl_cc_triple}[0-9]*-clang")
        set(_hl_best "")
        set(_hl_have "")
        foreach(_w IN LISTS _hl_wrappers)
          if("${_w}" MATCHES "${_hl_cc_triple}([0-9]+)-clang$")
            set(_hl_level "${CMAKE_MATCH_1}")
            list(APPEND _hl_have "${_hl_level}")
            if(_hl_level GREATER_EQUAL _hl_api
               AND (_hl_best STREQUAL "" OR _hl_level LESS _hl_best))
              set(_hl_best "${_hl_level}")
            endif()
          endif()
        endforeach()
        if(_hl_best STREQUAL "")
          list(SORT _hl_have COMPARE NATURAL)
          # Joined, because an unjoined CMake list renders as
          # `21;22;23` -- which reads like shell syntax in the one
          # message somebody is reading precisely because they are stuck.
          string(REPLACE ";" ", " _hl_have "${_hl_have}")
          message(FATAL_ERROR
            "No NDK compiler for ${_hl_cc_triple} at API ${_hl_api} or above "
            "in ${_ndk_bin}.\n"
            "The level was taken from ${_hl_api_from}. Levels this NDK has: "
            "${_hl_have}.\n"
            "Set -DSSTVAE_ANDROID_API=<level> to choose one, or "
            "-DSSTVAE_ANDROID_RIG=OFF to build without CAT.")
        endif()
        message(STATUS
          "Hamlib: no API ${_hl_api} wrapper for ${_hl_cc_triple}; "
          "using ${_hl_best}")
        set(_hl_api "${_hl_best}")
        set(_hl_cc "${_ndk_bin}/${_hl_cc_triple}${_hl_api}-clang")
      endif()

      # **`CXX` as well as `CC`, and it is load-bearing.** Setting only
      # `CC` produced a failure late in the build: configure found the
      # cross compiler for C and then silently fell back to the *host*
      # `g++` for C++. Most of Hamlib is C, so it got all the way to
      # `rotators/androidsensor` -- the one C++ directory, and one that
      # cannot be switched off, see below -- before dying on
      # `-stdlib=libc++`, a flag configure adds for every Android host
      # and one the host g++ has never heard of. A host compiler quietly
      # standing in for a cross one is the shape of this bug.
      set(_hl_cxx "${_ndk_bin}/${_hl_cc_triple}${_hl_api}-clang++")

      list(APPEND _hl_configure_args
           "--host=${_hl_triple}"
           "CC=${_hl_cc}"
           "CXX=${_hl_cxx}"
           "AR=${_ndk_bin}/llvm-ar"
           "RANLIB=${_ndk_bin}/llvm-ranlib"
           "STRIP=${_ndk_bin}/llvm-strip"
           "ac_cv_func_malloc_0_nonnull=yes"
           "ac_cv_func_realloc_0_nonnull=yes")

      # **The Android sensor rotator cannot be switched off, and it was
      # tried.** It is a rotator backend that points an antenna by the
      # phone's accelerometer -- no use whatever to this app, and the
      # only C++ in the tree we build -- so `-DSSTVAE_ANDROID_API`'s
      # neighbour here was briefly
      # `ac_cv_header_android_sensor_h=no`, pre-seeding autoconf's cache
      # to fail the check upstream gates it on (`configure.ac:171`).
      #
      # That does drop the directory from `ROT_BACKEND_LIST`. It also
      # breaks the build, because `src/rot_reg.c` guards the backend's
      # two halves on **different conditions**:
      #
      #   line  87: `#if HAVE_ANDROID_SENSOR`                 (declaration)
      #   line 141: `#if defined(ANDROID) || defined(__ANDROID__)` (table entry)
      #
      # `__ANDROID__` is defined by the compiler and cannot be unset, so
      # the table entry is unconditional on Android while the
      # declaration is not. The two agree only when `HAVE_ANDROID_SENSOR`
      # is true -- which upstream is entitled to assume, since a real NDK
      # always has `android/sensor.h`. Answering "no" makes the file
      # reference a function nobody declared.
      #
      # So it builds, and `CXX` above is what makes that work rather
      # than a precaution. Do not reach for the cache override again
      # without also solving line 141, which cannot be solved from
      # outside the source.
      message(STATUS
        "Hamlib: cross-building for Android ${CMAKE_ANDROID_ARCH_ABI} "
        "(API ${_hl_api}, from ${_hl_api_from})")
    endif()

    # Cross-compiling to the other macOS architecture.
    #
    # CMake's CMAKE_OSX_ARCHITECTURES means nothing to autotools, so
    # without this the library is built for whatever the *runner* is and
    # the link fails with an architecture mismatch -- which is a
    # confusing error a long way from its cause. There is no Intel macOS
    # runner any more, so the x86_64 slice is necessarily cross-built on
    # Apple silicon.
    #
    # `--host` is what puts configure into cross mode, so it stops trying
    # to run what it just built; the arch and deployment-target flags go
    # in the compiler variables, which configure accepts as arguments.
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
      list(LENGTH CMAKE_OSX_ARCHITECTURES _hl_narch)
      if(_hl_narch GREATER 1)
        # A fat build would need two configure/make runs and a lipo, and
        # nothing here asks for one: the packaging builds one slice per
        # job. Refusing beats silently producing a thin library.
        message(FATAL_ERROR
          "Hamlib is built one architecture at a time; "
          "CMAKE_OSX_ARCHITECTURES is '${CMAKE_OSX_ARCHITECTURES}'. "
          "Configure once per slice and combine them with lipo.")
      endif()
      set(_hl_arch "${CMAKE_OSX_ARCHITECTURES}")
      if(_hl_arch STREQUAL "arm64")
        set(_hl_triple "aarch64-apple-darwin")
      else()
        set(_hl_triple "${_hl_arch}-apple-darwin")
      endif()
      set(_hl_archflags "-arch ${_hl_arch}")
      if(CMAKE_OSX_DEPLOYMENT_TARGET)
        string(APPEND _hl_archflags
               " -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
      endif()
      # Only when it actually differs from the machine doing the build;
      # passing --host unconditionally would put a native build into
      # cross mode for no reason.
      if(NOT _hl_arch STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
        list(APPEND _hl_configure_args "--host=${_hl_triple}")
      endif()
      list(APPEND _hl_configure_args
           "CFLAGS=${_hl_archflags}"
           "CXXFLAGS=${_hl_archflags}"
           "LDFLAGS=${_hl_archflags}")
      message(STATUS "Hamlib: building for ${_hl_arch}")
    endif()

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

    # **`-avoid-version` belongs here, not in configure's LDFLAGS.** It
    # is a libtool flag; configure's link test invokes the compiler
    # directly, so putting it there stops the build at "C compiler
    # cannot create executables" with the real reason two layers down in
    # config.log.
    #
    # Overriding `libhamlib_la_LDFLAGS` rather than passing plain
    # `LDFLAGS=-avoid-version` at make time, for two reasons that are
    # not about libtool's precedence -- it copes with `-version-info`
    # and `-avoid-version` together, and strips the version. First, a
    # command-line `LDFLAGS` *replaces* the tree's, silently discarding
    # anything configure computed. Second, it would reach every link in
    # the tree, including the fifteen tool executables, where the flag
    # means nothing. This names the one variable upstream puts
    # `-version-info` in (`src/Makefile.am:24`) and substitutes for
    # exactly that; `-no-undefined` is carried over from the same line
    # because dropping it is not part of what is wanted here.
    set(_hl_make_args "-j${_hl_jobs}")
    if(ANDROID)
      list(APPEND _hl_make_args
           "libhamlib_la_LDFLAGS=-no-undefined -avoid-version")
    endif()

    execute_process(
      COMMAND make ${_hl_make_args} install
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

  # The `-avoid-version` check, asserted rather than assumed.
  #
  # A versioned SONAME here is not a cosmetic problem: Android's
  # packager takes only files named exactly `lib*.so`, so
  # `libhamlib.so.4` is dropped from the APK without comment and the app
  # dies at `dlopen` before `main` -- the same shape of pre-main,
  # no-output failure as the Windows import-library trap this file
  # already records, and just as hard to tell from a deadlock.
  if(ANDROID)
    file(GLOB _hl_versioned "${_hl_root}/lib/libhamlib.so.[0-9]*")
    if(_hl_versioned)
      message(FATAL_ERROR
        "Hamlib built with a versioned SONAME (${_hl_versioned}).\n"
        "libtool did not honour -avoid-version, so this library cannot go "
        "into an APK: Android installs only files named lib*.so and would "
        "drop it silently, leaving a dlopen failure before main. Delete "
        "${_hl_root} and fix the LDFLAGS before going further.")
    endif()
    if(IS_SYMLINK "${_hl_root}/lib/libhamlib.so")
      message(FATAL_ERROR
        "${_hl_root}/lib/libhamlib.so is a symlink; an APK needs the real "
        "file under that name.")
    endif()
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
  # MSVC needs an import library it generates itself, from the .def.
  #
  # `lib/gcc/libhamlib-4.lib` has a .lib extension but is a GNU ar
  # archive of dlltool stubs -- MinGW's format, in a directory named
  # gcc, which is the giveaway that was missed. MSVC's linker reads it
  # far enough to resolve every symbol, so the build *succeeds*; what it
  # cannot do is build a valid import directory from GNU-convention
  # import members, and it says nothing when it fails to. The result
  # links, then dies at load with STATUS_DLL_NOT_FOUND (0xC0000135) --
  # before main, so with no output on any stream, which is
  # indistinguishable from a deadlock and was diagnosed as one for
  # several rounds. `dumpbin /dependents` is where it is visible: the
  # exe lists a dependency named `(null)` and does not mention
  # libhamlib-4.dll at all.
  #
  # `lib/msvc/` ships a .def for exactly this, and lib.exe turns it into
  # a real short-import library (the difference is an .idata$2 import
  # descriptor member, which the gcc one has none of). /NAME matters:
  # the .def carries no LIBRARY statement, so without it the import
  # would name the wrong file again.
  set(_hl_implib "${_hl_root}/lib/gcc/libhamlib-4.lib")
  if(MSVC)
    set(_hl_def "${_hl_root}/lib/msvc/libhamlib-4.def")
    if(NOT EXISTS "${_hl_def}")
      message(FATAL_ERROR
        "Hamlib ${_hl_v} did not ship ${_hl_def}. The gcc import library "
        "beside it links but produces an executable that cannot start; do "
        "not fall back to it.")
    endif()
    set(_hl_implib "${CMAKE_CURRENT_BINARY_DIR}/hamlib/libhamlib-4.lib")
    if(NOT EXISTS "${_hl_implib}")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hamlib")
      # CMAKE_AR is lib.exe under MSVC. The prebuilt zip is x64-only, so
      # there is no architecture to choose.
      execute_process(
        COMMAND "${CMAKE_AR}" "/def:${_hl_def}" /machine:x64
                /name:libhamlib-4.dll "/out:${_hl_implib}"
        RESULT_VARIABLE _hl_lib_rc OUTPUT_VARIABLE _hl_lib_out
        ERROR_VARIABLE _hl_lib_out)
      if(NOT _hl_lib_rc EQUAL 0 OR NOT EXISTS "${_hl_implib}")
        message(FATAL_ERROR
          "Could not build a Hamlib import library from ${_hl_def}:\n"
          "${_hl_lib_out}")
      endif()
      message(STATUS "Hamlib: generated ${_hl_implib} from the .def")
    endif()
  endif()
  set_target_properties(sstvae_hamlib PROPERTIES
    IMPORTED_LOCATION "${_hl_root}/bin/libhamlib-4.dll"
    IMPORTED_IMPLIB "${_hl_implib}")
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

# Put the Windows DLLs beside the executable that needs them.
#
# This is not a test fixture, it is the shipping layout: Windows always
# searches the directory the .exe lives in, first, with no environment
# involved, and the installer will have to put them there anyway. The
# PATH route works but makes running a binary conditional on being
# launched the right way -- and when it does not work, the loader fails
# *before* main, so there is no output, no exit code anyone sees, and
# nothing to distinguish it from a deadlock. That is worth spending a
# file copy to never diagnose twice.
#
# All four DLLs, not just libhamlib: upstream's build links libusb and
# carries its own libgcc/libwinpthread, and a missing transitive
# dependency fails exactly as invisibly as a missing direct one.
function(sstvae_hamlib_copy_runtime target)
  if(NOT WIN32 OR NOT SSTVAE_HAMLIB_RUNTIME_DIR)
    return()
  endif()
  file(GLOB _hl_dlls "${SSTVAE_HAMLIB_RUNTIME_DIR}/*.dll")
  if(NOT _hl_dlls)
    message(FATAL_ERROR
      "No DLLs found in ${SSTVAE_HAMLIB_RUNTIME_DIR}; ${target} would fail to "
      "start with no diagnostic at all.")
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_hl_dlls}
            "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying the Hamlib runtime beside ${target}")
endfunction()
