// Downloading a published artifact, over plain HTTPS.
//
// QtNetwork rather than libcurl: Qt is a hard dependency of the app on
// all three targets, so this adds nothing to the install, and it brings
// the platform TLS story with it (SChannel, Secure Transport, OpenSSL)
// instead of a second one to configure and ship.
//
// Three things this does that a bare GET would not:
//
//   * **Verifies the checksum.** The Hub's 302 carries
//     `x-linked-etag`, which for an LFS object is the sha256 of the
//     content -- confirmed against the published decoder. So the file
//     is checked against a hash the server stated *before* sending it,
//     which is a real integrity check rather than a length comparison.
//     Redirects are therefore followed by hand: Qt's automatic
//     following would hide the response that carries it.
//   * **Writes atomically.** Download to `<name>.part`, verify, then
//     rename. A half-written artifact in the cache would be found by
//     `find_cached` on the next run and fed to onnxruntime, which is a
//     confusing failure a long way from its cause.
//   * **Reports progress.** First run pulls 9-21 MB; the design doc
//     makes a progress indication part of this phase.
//
// Needs a `QCoreApplication` to exist (it runs a local event loop), but
// does not need one to be running.

#ifndef SSTVAE_CHECKPOINT_QT_FETCHER_HPP
#define SSTVAE_CHECKPOINT_QT_FETCHER_HPP

#include <cstdint>
#include <functional>
#include <string>

#include "checkpoint/checkpoint.hpp"

namespace sstvae::checkpoint {

// received/total bytes; total is 0 while unknown.
using OnProgress = std::function<void(std::int64_t received, std::int64_t total)>;

// A Fetcher that downloads into `cache_dir()`.
Fetcher qt_fetcher(OnProgress on_progress = {});

// Make it the one `resolve_onnx` uses when given none. Call once at
// startup from anything that should be able to fetch on first run.
void install_qt_fetcher(OnProgress on_progress = {});

}  // namespace sstvae::checkpoint

#endif
