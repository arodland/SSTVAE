// Locating the model artifacts the codec decodes with.
//
// The counterpart of `sstvae/checkpoint.py`, minus torch: the `.pt`
// path is not here, because `load_codec` sends those to the torch
// backend, which this port does not have and never will.
//
// **Published filenames are immutable, and everything below follows
// from that.** `DEFAULT_REVISION` names a specific artifact rather than
// a moving "latest", because the on-air format is not frozen and a
// codec that silently changed under an operator would break
// interoperability with every station still running the old one. That
// is what lets a cache hit be trusted outright rather than revalidated:
// a HEAD on every run costs a round trip, fails needlessly offline, and
// buys nothing when the name already identifies the bytes.
//
// **We speak plain HTTPS to the Hub and keep our own cache**, rather
// than sharing `huggingface_hub`'s. Reading its cache would be easy;
// *writing* it means reproducing an undocumented internal layout --
// `blobs/` keyed by etag, `snapshots/<commit>/` symlinked into them
// (copied instead on Windows), `refs/main`, and the lock files around
// it -- and a near-miss corrupts a cache that another program owns. The
// cost of not sharing is that someone who runs both the Python tools
// and the native app downloads ~9-21 MB twice. That is worth paying to
// keep the failure mode "an extra download" instead of "a broken
// huggingface_hub".
//
// The dependency split is the one used throughout the port: **the part
// that can be wrong has no dependencies.** Filename construction,
// `--model` resolution and the cache lookup are path arithmetic, live
// in `sstvae_core`, and are checked against the reference. Only the
// download needs a network stack, and it arrives as a `Fetcher` seam --
// so a build without one still resolves an explicit `--model` and still
// finds an already-cached artifact, which is every case but the first
// run.

#ifndef SSTVAE_CHECKPOINT_CHECKPOINT_HPP
#define SSTVAE_CHECKPOINT_CHECKPOINT_HPP

#include <array>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sstvae::checkpoint {

inline constexpr std::string_view DEFAULT_REPO = "arodland/sstvae";

// The stem of the checkpoint the artifacts were exported from. Bumping
// it is how a new published codec is adopted, and that must happen in
// the same change as the code needing it.
inline constexpr std::string_view DEFAULT_REVISION = "v3";

inline constexpr std::array<std::string_view, 3> PRECISIONS = {"fp32", "fp16",
                                                               "int8"};

// fp16 ships by default: measured identical to fp32 end to end
// (docs/onnx.md) at half the size. int8 exists but costs ~1 dB of
// effective SNR on the *encoder*, whose error every receiver pays for.
inline constexpr std::string_view DEFAULT_PRECISION = "fp16";

// The decoder's gradient graph, for transmit-time latent optimization
// (docs/latent-optimization.md). Not a codec part: no receiver ever
// loads it, and a station that does not optimize never fetches it.
//
// **fp32 whatever the codec's precision is**, because fp32 is the only
// version published -- the fp16 converter emits a graph onnxruntime
// will not load, and int8 is excluded on principle since
// differentiating `ConvInteger` is not well defined. The override is
// silent rather than an error because `--precision` is a statement
// about the *codec*, and refusing to optimize because someone chose
// int8 for their decoder would answer a question they did not ask.
inline constexpr std::string_view GRAD_PART = "decoder-grad";
inline constexpr std::string_view GRAD_PRECISION = "fp32";

// Revisions that actually ship one. v1 and v2 predate the feature and
// DEFAULT_REVISION is still v2, so this is the difference between a
// clear message and a 404 on a filename the operator has never seen.
inline constexpr std::array<std::string_view, 1> GRAD_REVISIONS = {"v3"};

// Everything an operator has to fix: a bad `--model`, a missing
// artifact, an unreachable Hub. The message is the deliverable here --
// `docs/native-app.md` makes the offline story this phase's
// responsibility, and `checkpoint.py`'s wording is the model to follow
// because it already tells an offline user exactly what to do.
class CheckpointError : public std::runtime_error {
public:
    explicit CheckpointError(const std::string& what) : std::runtime_error(what) {}
};

// ("encoder", "fp16") -> "v1-encoder-fp16.onnx"
std::string onnx_filename(std::string_view part,
                          std::string_view precision = DEFAULT_PRECISION);

// Where downloaded artifacts live. `SSTVAE_MODEL_CACHE` overrides it,
// which is how the tests avoid touching a real one and how a field
// laptop can be pointed at a USB stick.
std::string cache_dir();

// The cached path for an artifact, if it has been downloaded.
std::optional<std::string> find_cached(std::string_view filename);

// The URL an artifact is published at. Exposed so an error message can
// tell an operator exactly what to download by hand.
std::string artifact_url(std::string_view filename);

// Download `filename` into `cache_dir()` and return its path; throws
// CheckpointError if it cannot.
//
// Null is a supported configuration, not a bug: a build with no
// networking still resolves an explicit `--model` and still finds a
// warm cache.
using Fetcher = std::function<std::string(std::string_view filename)>;

Fetcher default_fetcher();
void set_default_fetcher(Fetcher fetcher);

// Locate one ONNX part, given whatever the user passed to `--model`.
//
// The `.pt` checkpoint was one file and the ONNX codec is two, so
// `--model` cannot simply be "the model file" any more. It accepts:
//
//   empty          the published artifacts: cache first, then fetched
//   a directory    `*-{part}-{precision}.onnx` inside it
//   a .onnx file   that part; the sibling is derived from the name by
//                  substitution, and only when actually needed
//
// Throws CheckpointError with text an operator can act on.
std::string resolve_onnx(std::string_view part, std::string_view path = {},
                         std::string_view precision = DEFAULT_PRECISION,
                         const Fetcher& fetcher = nullptr);

}  // namespace sstvae::checkpoint

#endif
