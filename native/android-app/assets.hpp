// The model artifacts that ship inside the APK.
//
// **Bundled rather than fetched, and the reason is not download size.**
// The model *is* part of the on-air contract: the latent space is
// learned, so an encoder from one checkpoint only means anything to a
// decoder from the same one, and two stations must be running the same
// artifacts to talk at all (see the parity note in CLAUDE.md about what
// must match between stations). "Update the model without an app
// update" therefore reads as flexibility and behaves as a way to
// silently desynchronise a station from the band -- a codec revision is
// a coordinated, everyone-at-once event, which is exactly what an app
// release already is.
//
// What bundling buys is the first run working with no network. This is
// a radio app, and the moment of need is disproportionately a field
// site with no coverage; an operator who installs at home and first
// opens the app on a hilltop otherwise has a listener that cannot
// decode anything, and the failure is total rather than degraded. It
// costs ~18 MB on a ~34 MB download, and model weights barely compress
// (91%), so that is close to what it is.
//
// **The fetcher stays.** Bundling changes the default source, not the
// mechanism: `checkpoint::Fetcher` is still wired up, still used if an
// asset is missing, and is what a build with `SSTVAE_ANDROID_BUNDLE_
// MODELS=OFF` runs on unchanged.

#ifndef SSTVAE_ANDROID_ASSETS_HPP
#define SSTVAE_ANDROID_ASSETS_HPP

#include <optional>
#include <string>

#include "codec/codec.hpp"

namespace sstvae::androidapp::assets {

// The bytes are read out and handed over, and the codec releases them
// as soon as the ORT session is built -- so bundling costs APK size and
// a transient allocation during load, not resident memory for the life
// of the process. See `codec::ModelBlob`.
//
// Resolve the app's AssetManager. **Must be called from the UI thread**
// during startup, for the same reason `audio::android::set_java_vm` is:
// it needs the Java context, and after this nothing else does -- the
// `AAsset_*` family is a plain NDK C API with no JNIEnv anywhere in it,
// so every later call is thread-free by construction.
//
// Returns false if there is no context; the app then falls back to
// fetching, which is the pre-bundling behaviour and still correct.
bool init();

// A `codec::BlobResolver`: "encoder"/"decoder" -> the bytes of the
// bundled artifact, or nullopt when this build does not carry one.
//
// Nullopt is the ordinary answer for a `-DSSTVAE_ANDROID_BUNDLE_MODELS=
// OFF` build, and the codec treats it as "go and find it" rather than
// as an error -- so the two builds differ only in what is in the APK.
std::optional<codec::ModelBlob> model_blob(const std::string& part);

}  // namespace sstvae::androidapp::assets

#endif
