/* A pthread.h that exists only to let MSVC compile <hamlib/rig.h>.
 *
 * Hamlib's rig.h includes <pthread.h> unconditionally, with the comment
 * "For MSVC install the NuGet pthread package". The Windows binaries we
 * bundle are MinGW-built, so they carry mingw-w64's winpthreads inside
 * the DLL; MSVC has no pthread.h at all, and the include fails before
 * anything else can go wrong.
 *
 * Only two types are needed: `pthread_t` and `pthread_mutex_t` both
 * appear in `struct rig_state`, which `struct RIG` embeds. Nothing here
 * declares a function, because nothing in this project calls one.
 *
 * THE SIZES BELOW ARE NOT LOAD-BEARING, AND THAT IS THE POINT.
 *
 * If they disagreed with the winpthreads types the DLL was built
 * against, `struct rig_state` would be laid out differently here than
 * inside Hamlib, and every field after the first mutex would be at the
 * wrong offset -- a silent memory-corruption bug of the worst kind. So
 * `core/rig/hamlib.cpp` is written never to dereference a `RIG*`: it
 * gets the manufacturer and model through `rig_get_caps_cptr()`, which
 * takes a model number rather than a pointer, and the only struct it
 * ever reads through is `struct rig_caps` -- which contains no pthread
 * members and so cannot be affected by anything in this file.
 *
 * The pointer-sized definitions match mingw-w64's winpthreads, which
 * makes the layouts agree in practice as well; that is belt and braces,
 * not the argument.
 */

#ifndef SSTVAE_MSVC_PTHREAD_SHIM_H
#define SSTVAE_MSVC_PTHREAD_SHIM_H

#if !defined(_MSC_VER)
#error "This shim is for MSVC only; every other toolchain has a real pthread.h."
#endif

#include <stdint.h>

typedef uintptr_t pthread_t;
typedef void* pthread_mutex_t;

#endif /* SSTVAE_MSVC_PTHREAD_SHIM_H */
