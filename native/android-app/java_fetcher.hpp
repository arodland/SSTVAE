// The Android model downloader: platform HTTPS, our verification.
//
// A `checkpoint::Fetcher`, so it drops into the same seam
// `qt_fetcher` uses and nothing above it changes -- `resolve_onnx`, the
// cache lookup and the offline message are all shared with the desktop.
//
// **Java does transport and only transport.** Qt for Android ships no
// TLS backend, so the alternative was bundling OpenSSL per ABI and
// owning its patch cadence; `HttpsURLConnection` uses the platform
// stack and the system trust store instead. What is *not* delegated is
// the part that can be wrong: the sha256 the Hub declares is compared
// here, and the `.part` file is renamed into place here, so a truncated
// or corrupted download cannot become a cache hit on the next run. That
// keeps one implementation of the check across both platforms, which is
// the reason the split falls where it does rather than at the obvious
// seam of "let Java return a path".

#ifndef SSTVAE_ANDROID_JAVA_FETCHER_HPP
#define SSTVAE_ANDROID_JAVA_FETCHER_HPP

#include <cstdint>
#include <functional>

#include "checkpoint/checkpoint.hpp"

namespace sstvae::androidapp {

using OnProgress = std::function<void(std::int64_t received, std::int64_t total)>;

checkpoint::Fetcher java_fetcher(OnProgress on_progress = {});

// Install it as the process-wide default, the counterpart of
// `checkpoint::install_qt_fetcher`.
void install_java_fetcher(OnProgress on_progress = {});

}  // namespace sstvae::androidapp

#endif
