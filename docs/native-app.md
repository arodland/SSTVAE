# Native desktop application (C++ / Qt 6)

**Status: Phases 0 and 1 implemented (2026-07-28). Phases 2–5 are
design.** The scaffolding, the parity harness and the **whole modem**
exist in `native/` — `golay`, `ofdm`, `dsp`, `framing`, `beacon`,
`sync`, `modem` — and the Python suite passes against them. Everything
from the headless app core onwards is still plan. The rest of this
document records that plan, the decisions behind it, and the questions
still open, so none of it has to be re-derived.

**Prerequisite: the ONNX path in `docs/onnx.md` must land first.** The
native app cannot embed torch, so `codec.py`'s ONNX rewrite and the
published `.onnx` artifacts are a hard dependency, not a parallel track.

## Scope

The **application** is rewritten. Everything else stays Python:

| Stays Python | Becomes C++ |
|---|---|
| `scripts/train.py`, `latent_channel.py`, `waveform_channel.py`, `data.py` | `sstvae/gui/` |
| `sstvae_encode.py` / `decode` / `simulate` / `listen` CLIs | `sstvae/rx/`, `sstvae/tx/` |
| `hfchannel.py` (simulation) | `sstvae/modem/`, `codec.py`, `images.py` |
| The whole `sstvae` package as the **reference implementation** | `audio.py`, `rig/`, `overlay/`, `gui/settings.py`, `checkpoint.py` |

`sstvae/gui/` is **deleted** when the native app is packaged (decision
1), taking the `gui` extra and `pyside6-essentials` with it; it is
frozen — bug fixes only — from the moment the native app reaches
parity.
`checkpoint.py` is ported rather than left behind because the native app
fetches its own artifacts (decision 5); the Python copy stays for the
CLIs.

Python remains the normative definition of the on-air format. When the
two disagree, Python is right until proven otherwise. This is not
sentiment — it is the only way to keep "compatible implementation" a
checkable claim, and it is the same argument `docs/onnx.md` makes for
publishing canonical model artifacts.

## Why do this at all

Be honest about what it buys, because one of the obvious motivations
turns out to be weak.

| Distribution | Installed size |
|---|---|
| Frozen Python (PyInstaller/Nuitka) *after* ONNX | ~110–140 MB |
| **C++ / Qt (this plan)** | **~75 MB** |
| Hypothetical all-Rust + Slint | ~50 MB |

The ~75 MB is Qt (~45), onnxruntime (~15), `libhamlib` with its backends
(~10), PortAudio and the app itself (~5). Model artifacts are **not** in
it — they are fetched on first run (decision 5) and add ~20.7 MB to the
cache, not to the download.

Once ONNX lands, frozen Python is already in the same size class as a
Qt rewrite. **Download size is not a good reason to do this.** The real
gains are:

- **Startup and responsiveness.** No interpreter, no import graph. The
  Python app's cold start is dominated by imports it can't avoid.
- **Install robustness.** One signed, notarized artifact per platform
  with no Python, no PortAudio-from-a-distro-package, no wheel
  resolution. This is the single biggest user-facing win.
- **Native platform integration** — menus, dialogs, HiDPI, IME,
  accessibility — at a fidelity PySide6 gives grudgingly. See commit
  `955e3ad` ("get the File menu to show up on mac", *"this isn't quite
  the right fix but it will work for now"*) for the flavour of the tax
  being paid now.
- **Memory.** numpy + scipy + PySide6 + onnxruntime resident, for an app
  that is idle most of the time.

If the answer to "is that worth a permanent parity burden and an
irreducible hardware-testing tail" is no, the correct move is to stop
after ONNX and ship a signed PyInstaller bundle. That is a legitimate
outcome of reading this document.

## Language and toolkit

**C++20 + Qt 6 LTS (6.8/6.9), LGPLv3, dynamically linked.**

Two properties decided this over the alternatives, and both are specific
to *this* codebase rather than general C++ advocacy:

**~~PortAudio continuity.~~ Superseded 2026-07-28 — see "Audio: use
QtMultimedia" below.** The original argument was that `sounddevice` *is*
PortAudio, so `audio.py`'s hard-won behaviour ports literally. The
behaviour is still worth porting, but PortAudio itself is no longer worth
keeping, and the conclusion now points the same way as the toolkit
choice rather than merely rhyming with it.

**Qt replaces PIL, not just PySide6.** `QImage`/`QPainter`/
`QFontDatabase` cover everything `overlay/render.py` does. That removes
a whole dependency, removes the risk of the renderer and the editor
preview diverging (the same painter code draws both, so
"the preview **is** `render()`'s output" becomes true by construction),
and removes the font-discovery-per-OS problem in `images.py`.

### Alternatives considered

| Option | Why not |
|---|---|
| **Rust + Slint/egui** | ~50 MB and the nicest CI story (`cargo-dist`), but throws away the PortAudio knowledge, means rebuilding the overlay editor's scene graph and drag handles from scratch instead of porting `QGraphicsView` directly, and neither toolkit gives native menus or accessibility without bolt-ons. |
| **Rust core + Qt C++ front end** | Two toolchains and an FFI boundary, to buy memory safety in code that is already covered by a strong test suite. |
| **Tauri** | WebKitGTK fragmentation makes Linux install UX *worse* than today, and a 20 fps waterfall over IPC is self-inflicted. |
| **Avalonia / .NET** | ~40–60 MB runtime floor with no ecosystem advantage for DSP. |
| **Flutter** | Would FFI to C++ for the DSP anyway — i.e. this plan plus a second language. |
| **Frozen Python** | The baseline this must beat, not an alternative implementation. |

### What we accept

- ~75 MB installed vs ~50 MB for the Rust path. Explicitly compromised.
- LGPLv3 obligations. Dynamic linking against unmodified Qt shared
  libraries satisfies them with no further ceremony; **do not statically
  link Qt** without a commercial licence.
- No memory safety net. Mitigated by ASan/UBSan in CI and by the fact
  that the numeric core is buffer-in/buffer-out with fixed shapes.

## Dependency stack

| Need | Choice | Notes |
|---|---|---|
| FFT | **pocketfft** (BSD, header-only) | `scipy.fft` *is* pocketfft. Same algorithm, same rounding — parity vectors match bit-for-bit instead of to an argued tolerance. This is worth more than it sounds. |
| Arrays | Eigen (MPL2) for the `ofdm.py` DFT matrix; `std::vector<std::complex<double>>` elsewhere | The code is overwhelmingly 1-D. Do not over-abstract. |
| `firwin`, `hilbert`, `fftconvolve` | Hand-written, ~150 lines total | Windowed sinc; FFT-based analytic signal; overlap-save. |
| `resample_poly` | Hand-written polyphase, ~80 lines | **The delicate one.** scipy's kernel defaults (Kaiser window, `gcd` handling) must be replicated exactly or capture and playback drift apart. |
| ONNX | onnxruntime C++ API | First-class native API, not a binding layer. |
| Audio | **QtMultimedia** (`QAudioSource`/`QAudioSink`) | See below. Was PortAudio; changed 2026-07-28 for a measured reason, not for tidiness. |
| Rig | **`libhamlib`, linked** | User installs nothing. One `RIG*` on a dedicated thread; model 2 covers sharing a radio via someone else's `rigctld`. Deletes the socket client and the `rigctld -l` parser. See "Bundling Hamlib". |
| Hub fetch | `QNetworkAccessManager` | Ports `checkpoint.py`'s cache-first, immutable-filename model — a cache hit is trusted outright, no revalidating HEAD. |
| Images, text, fonts | Qt (`QImage`, `QPainter`, `QFontDatabase`) | Replaces Pillow entirely. |
| JSON | `QJsonDocument` | Settings and overlay documents; must round-trip the existing files unchanged. |
| Bindings | pybind11 | Both directions — see below. |

**Do not use FFTW.** GPL-2+ unless licensed commercially, which fights
the project's Artistic 2.0 distribution.

### Audio: use QtMultimedia

**Decided 2026-07-28 on evidence, and already implemented in the Python
app** — `sstvae/gui/qtaudio.py`, with `gui/audio_backend.py` dispatching
on `audio.backend`. Qt is the default for capture *and* playback;
PortAudio is retained only because Qt does not enumerate
PulseAudio/PipeWire monitor sources. Validated on real hardware: capture
clean through 800 ms of deliberate GIL starvation, and a full
transmit→receive loopback through Qt on both sides decoding at +27.4 dB
with the callsign recovered. So the C++ port inherits a *working, tested*
design here rather than a plan.

Two costs worth knowing before the C++ work starts. In **Python**,
QtMultimedia means `pyside6-addons`: 232 MB → 648 MB installed, 195 MB of
it Chromium that nothing loads. **In C++ that cost vanishes** — Qt
Multimedia is a small shared library and WebEngine is simply not linked,
so this is a PySide6 packaging artifact, not a Qt one. And PySide6 cannot
marshal `QAudio::State` into a Python slot, which forced polling
`error()` instead of connecting `stateChanged`; C++ has no such problem.

The original argument follows, since the reasoning is what matters. The Python
app's worst bug to date was that its PortAudio capture callback is
*Python*, so it runs on the host's realtime audio thread and must take
the GIL. When the Qt thread held the GIL — converting a 640×480 preview
to a QPixmap and painting it, immediately after every decode poll — the
callback could not run, and on a JACK device, which has no buffer behind
its couple-of-millisecond period, the audio was silently skipped:
200–350 samples per poll, 5 dB of SNR, a mangled picture, with sync
succeeding and every frame reported. It took several rounds to find
because the identical code was clean headless and clean on
`pulse`/`pipewire`.

The obvious fix is to keep Python off the realtime thread by using
PortAudio's blocking API. **That does not work**: `stream.read()`
corrupts the heap on PortAudio's JACK backend at every blocksize and
latency tried. So the Python app is left with a warning
(`audio.warn_if_fragile_host`) and "don't use JACK".

A C++ app does not have this problem *at all* — there is no GIL, so a
callback is simply a callback. That alone settles PortAudio-vs-Qt on
merit rather than convenience. Given that, prefer QtMultimedia:

- `QAudioSource` is pull-based over a buffered `QIODevice`, so the
  realtime side stays inside Qt's C++ backend and the app drains a
  buffer. The structural hazard disappears rather than being tuned
  around.
- One fewer native dependency, and `QMediaDevices` gives device
  hot-plug and default-change notifications PortAudio does not.
- It is the same class in Python and C++, so a QtMultimedia port of the
  *Python* app is not throwaway work — it is Phase 2 done early, on the
  real hardware, with the Python test suite still available to check it.

**Carry these forward regardless of library**, since they are properties
of the problem and not of PortAudio:

- Open at the device's own rate and resample in our own code
  (`StreamResampler`). `QAudioSource` will equally happily accept an
  8 kHz `QAudioFormat` and let its backend convert.
- Capture resampling must be *stateful across blocks*; per-block
  `resample_poly` cost 4.7 dB on a real recording.
- Nothing may hold the ring buffer's lock across a bulk copy.
- Keep the resampling and ring logic out of the Qt layer, so the
  fake-device tests that caught three of these bugs still apply.

### Prior art worth mining

`codec2`'s `src/ofdm.c` is the closest existing thing to `sync.py` —
pilot-based EQ, coarse/fine frequency offset estimation, a sync state
machine with hysteresis — and it is BSD. `freedv-gui` is worth reading
for PortAudio device configuration and Hamlib integration patterns.

Caveat so nobody wastes a day looking: **freedv-gui is wxWidgets, not
Qt.** The value there is DSP and ham-app plumbing, not GUI structure.

## The parity problem

Two implementations of an on-air waveform will drift, and the failure
mode is nasty: not a crash, but a station that decodes 90% of pictures
and mysteriously fails the rest. Three mechanisms address it, covering
different seams.

### 1. Golden vectors

A corpus generated by Python and committed: known inputs and expected
outputs at every module boundary (`golay`, `ofdm`, `dsp`, `framing`,
`beacon`, `sync`, `modem`), plus whole-transmission WAVs with expected
decoded latents. Both test suites run against the same files.

Generated by a committed script so they can be regenerated
deliberately; regenerating them is a reviewable diff, which is the
point.

### 2. pytest runs the C++ modem (pybind11 module)

A pybind11 module wrapping the C++ core, so **the existing Python test
suite becomes the C++ modem's acceptance suite** — `test_modem_e2e.py`,
`test_blind_acquisition.py`, `test_beacon.py`, and the slow
`test_listen_state_machine.py` all apply unchanged. This is the highest
leverage item in the whole plan and costs about a 200-line shim.

Answers: *is the C++ modem numerically right?*

### 3. C++ app embeds the Python modem (dev builds only)

A `SSTVAE_DEV_PYMODEM` CMake target that links libpython
(`pybind11/embed.h`) and runs the real Python modem in-process behind
the same interface the C++ modem implements. **Never shipped.**

Answers a different question: *is the app wired right?* Buffering, frame
handoff, engine state, sink behaviour — validated while the modem
underneath is known-good, so a bug has only one place to hide.

Design notes:

- **Make the seam swappable per module, not just at
  `Modem::demodulate`.** A whole-transmission A/B tells you "these
  disagree" and nothing else. Per-module swapping lets one run of
  C++-sync-plus-Python-everything-else bisect a divergence immediately.
  Nearly free if designed in; expensive to retrofit.
- **The comparator records, it does not merely assert.** On mismatch,
  dump both sides' intermediates to `.npz`. Debugging a CFO estimator
  disagreement means plotting it, not diffing two floats.
- Pass samples as `py::array_t<double>` wrapping the C++ buffer with a
  capsule — no copy, and both implementations see bit-identical input.

**The trap:** this configuration validates data flow and correctness. It
does **not** validate concurrency or timing. `py::gil_scoped_acquire` in
the rx worker serializes calls that run concurrently in the shipping
build, and a ~173 ms demod holding the GIL distorts exactly the property
you would be tempted to check. Do not conclude "the rx threading is
sound" from this build; that verdict comes only from the all-C++ build.

### 4. Interop CI

A job that runs Python TX → C++ RX and C++ TX → Python RX over the
simulated channel at several SNRs and modes. This is the acceptance
test that actually matters, because it is the thing users will do.

## Repository layout

Same repository. The golden vectors, the Python reference, and `pytest`
all need to be one `git checkout` away from the C++ tree or the parity
machinery becomes a submodule-synchronization chore.

```
native/
  CMakeLists.txt
  vcpkg.json
  core/                 # no QtWidgets anywhere below here
    config.hpp          # GENERATED from sstvae/config.py -- never hand-edited
    dsp/  ofdm/  golay/  framing/  beacon/  sync/  modem/
    codec/              # onnxruntime
    overlay/            # QtGui only, no QtWidgets
    audio/              # rate/format conversion, no Qt
      qt/               # QtMultimedia (see "Audio: use QtMultimedia")
    rig/                # the threading design, no external deps
      hamlib.cpp        # libhamlib (see "Bundling Hamlib")
    latents/  util/
    rx/  tx/
  gui/                  # QtWidgets
  bindings/
    module/             # pybind11 module: pytest loads the C++ core
    embed/              # dev-only: C++ app loads the Python modem
  tests/
tools/
  gen_config_header.py
  gen_golden_vectors.py
```

This sketch predates two decisions recorded further down and has been
corrected to match them: audio is QtMultimedia rather than PortAudio,
and rig control links `libhamlib` rather than speaking to `rigctld`
over a `QTcpSocket`. Both are split so that the part with the logic in
it has no external dependency and stays testable without one.

### Layering rules

Mirrors the Python rules, for the same reasons:

- Nothing under `core/` may include QtWidgets.
- `core/overlay/` may include QtGui only — headless rendering works
  under `QGuiApplication` with `QT_QPA_PLATFORM=offscreen`, so overlays
  stay renderable from the command line. This is the C++ restatement of
  "nothing in `sstvae/overlay/` may import Qt".
- Nothing outside `bindings/embed/` links libpython.
- Enforced by a CI grep, not by good intentions.

### `config.hpp` is generated

`CLAUDE.md`: *"All waveform/latent numbers must agree through this
module."* Two hand-maintained copies of `config.py` would be the single
most likely source of a silent on-air incompatibility, and the failure
would be invisible in both test suites if both were edited consistently
but wrongly.

`tools/gen_config_header.py` emits `config.hpp` from `sstvae/config.py`.
CI regenerates and asserts the diff is empty. `sstvae/config.py` remains
the only place a waveform number is written.

## Phases

### How these are sized

Deliberately **not** in developer-weeks. Most of the code here will be
written by an agent, which compresses the mechanical work by a large and
unpredictable factor while leaving the parts gated on hardware, external
services, and human judgement almost untouched. A week estimate would be
wrong in both directions at once.

Each phase is instead described by three things that can be checked:

- **Volume** — lines of Python displaced, and the new harness code
  required. Counted from the current tree, not estimated.
- **Verified by** — what closes the loop. This is the important axis:
  work verifiable by CI iterates in minutes, work verifiable only
  against a radio iterates at the speed of a person with a radio.
- **Needs you for** — the parts that cannot be delegated.

### Phase 0 — Scaffolding and parity harness — **DONE 2026-07-28**

CMake + vcpkg manifest, GHA matrix, `install-qt-action`, ccache,
ASan/UBSan job. `gen_config_header.py` and its CI check. Golden-vector
generator and committed corpus. pybind11 module skeleton. Port `golay`
and `ofdm` — small, completely testable, and they prove the whole loop
end to end before anything hard is attempted.

**Exit:** `pytest` passes with C++ `golay` and `ofdm` substituted via the
pybind11 module, on all three platforms in CI.

**Met.** Locally: 243 fast tests and all 19 `-m slow` tests pass under
`pytest --native`, plus 16 direct A/B parity tests and two C++ binaries
against the corpus. On CI (the repo's first workflows), the first run
failed six of eight jobs for two causes, both since fixed and both worth
recording because they are the kind of thing that recurs:

- **`pytest` as a console script does not put the working directory on
  `sys.path`**, so `import sstvae` failed in every native job. Invisible
  locally, where `python -m pytest` does. The native jobs now install
  the package.
- **The golden corpus is not byte-reproducible**, and demanding that it
  be was a design error — one that took *two* rounds to get right,
  which is the interesting part. The first fix classified vectors by
  whether they reached a `@` (BLAS) or an FFT, on the evidence that
  three such vectors differed while the `exp` tables matched. That
  evidence was one green Linux run: the next round failed `MOD_MATRIX`
  and `DEMOD_MATRIX` on macOS and Windows, because **no standard
  requires a transcendental to be correctly rounded** and x86-64 and
  Apple silicon genuinely disagree. The rule now follows from IEEE 754
  rather than from a sample: bitwise means integer arithmetic,
  `+ - * /`, and seeded numpy RNG draws; everything touching `exp`,
  BLAS or an FFT is compared by value against a tolerance and carries a
  platform-stable fingerprint instead of a churning SHA-256. 11 vectors
  bitwise, 10 by tolerance.
- **Windows CMake picked MinGW GCC over MSVC**, because it is first on
  the runner's `PATH` and nothing said otherwise. It built a perfectly
  good `.pyd` that CPython then could not load, since a MinGW extension
  is ABI-incompatible with an MSVC-built interpreter. Fixed with
  `ilammy/msvc-dev-cmd` before the configure step. The diagnosis cost a
  round only because `conftest.py` caught `ImportError` and reported
  "no extension module" — it now distinguishes *not built* from *built
  but unloadable* and prints the real error.
- **Git rewrote the corpus on checkout.** With everything above fixed,
  Windows still failed — on `manifest.json` alone, with every `.npy`
  passing. Git for Windows converts LF to CRLF on checkout by default,
  and the `.npy` files escaped only because git's binary heuristic left
  them alone. So the corpus being compared was not the corpus that was
  committed. Fixed by a `.gitattributes` marking the golden tree and
  the generated `config.hpp` as `-text`; `--check` additionally
  recognises a line-endings-only difference and says so, because the
  symptom otherwise reads as a content problem rather than a checkout
  one. **Any committed generated artifact needs this**, and the failure
  is invisible on Linux and macOS.
- **The slow suite was vacuous**: `pytest -m slow --native` reported
  "16 skipped" and a green tick in 0.41 s. `test_listen_state_machine.py`
  opened with `pytest.importorskip("torch")` and then never used torch —
  a leftover from before the ONNX migration, which stubs `reconstruct`
  and needs no model at all. This is precisely the hazard `CLAUDE.md`
  names ("those tests `importorskip`, so without torch here they would
  quietly stop running instead of failing, and there is no CI to
  notice"), caught the first time there was a CI to notice. Guard
  removed; those tests now run everywhere.

Three lessons for Phase 1, all cheap to act on and expensive to
retrofit:

- **Decide what is genuinely reproducible before making reproducibility
  a gate**, and derive the rule from the standard rather than from
  whichever platforms happened to agree today. `sync` and `modem` are
  far more FFT- and transcendental-heavy than `ofdm`, so most of their
  vectors will land on the tolerance side.
- **Pin the toolchain explicitly on every platform.** CMake's idea of
  "the compiler" is whatever it finds first, and on a GitHub runner that
  is not what you meant.
- **A byte-exact check is a promise about the whole path to the bytes**,
  not just about the generator: the toolchain that produced them, the
  library that computed them, and git's handling of them on checkout are
  all part of it. Three of the four failures above were somewhere on
  that path rather than in the code being tested.
- **A skip is not a pass.** Audit `importorskip` guards as modules are
  ported — a stale one silently deletes coverage, and the whole parity
  strategy is built on tests that are actually executing.

Deviations from the plan above, all deliberate:

- **No vcpkg manifest.** Phase 0's dependencies are pybind11 (build-time
  only, from pip) and the standard library. A manifest would be an empty
  ceremony until pocketfft, Eigen, onnxruntime and Hamlib arrive in
  Phases 1–2.
- **No `install-qt-action` or ccache.** No Qt code exists yet, and the
  whole native build is under 10 seconds. Both belong with Phase 3.
- **The notarization spike did not happen.** It is still the right
  advice — see "Where the cost actually sits" — but it needs credentials
  and an Apple Developer account, so it is yours to trigger. It remains
  the cheapest way to move the least compressible work earlier.
- **The Qt-version-against-Windows-10 check is still open.** Decision 2's
  one blocking unknown; no Qt dependency has been pinned, so nothing is
  foreclosed.

What exists:

| Path | What it is |
|---|---|
| `tools/gen_config_header.py` | `sstvae/config.py` → `native/core/config.hpp`, `--check` for CI |
| `tools/gen_golden_vectors.py` | the corpus, `--check` for CI |
| `tools/check_layering.py` | the layering rules, as promised, enforced |
| `tools/build_native.sh` | configure + build against the project venv |
| `native/core/{golay,ofdm}/` | the two ported modules |
| `native/core/testing/npy.hpp` | `.npy` reader, ~90 lines, no zip dependency |
| `native/tests/` | two C++ binaries, `check.hpp`, `golden/` (22 files, 431 KB) |
| `native/bindings/module/` | the pybind11 module |
| `tests/test_native_parity.py` | side-by-side A/B, skips if unbuilt |
| `.github/workflows/ci.yml` | four jobs; the project's first CI |

- **Volume** — 135 lines displaced (`golay.py` 53, `ofdm.py` 82), plus
  the codegen, vector generator, and binding shim, which are new code
  rather than ports. Gated on `test_golay.py`, `test_ofdm.py`.
- **Verified by** — CI alone.
- **Needs you for** — Apple and Azure signing credentials, for the
  notarization spike that belongs here. Also confirm the Qt version
  against the Windows 10 and Intel-Mac floors before the matrix is
  fixed; that check is the one blocking unknown left in decision 2.

> Do this **before** `sync.cpp`, not after. Written against a harness,
> a wrong CFO search fails a vector the moment it is written; written
> without one, it fails a whole-transmission decode much later with
> 1,000 lines of untested code beneath it and nothing to bisect.

#### What Phase 0 taught, that Phase 1 should inherit

**Generate format constants; do not port the algorithms that produce
them.** `ofdm.pilot_sequence()` draws from `np.random.default_rng(42)`.
Reimplementing PCG64 and numpy's bounded-integer draw in C++ would make
numpy's RNG internals part of the on-air format, permanently. The
sequence is 24 fixed QPSK symbols, so `config.hpp` carries the quadrant
indices and C++ evaluates the same `pi/4 + pi/2*k` expression. **The
interleaver permutations in `framing.py` are the same problem at much
larger scale** (`INTERLEAVER_SEED + group`) and should get their own
generated artifact rather than a ported RNG.

**Bit-exactness is the wrong target; knowing *why* it fails is the right
one.** The first parity failure was 2.85e-14 on `MOD_MATRIX`. Chasing it
found a real defect — in the *reference*. `ofdm.py` builds phasors on an
unreduced argument reaching 262 rad, where one ulp is 5.7e-14, so it
carries ~3e-14 of error; the C++ reduces `(n*f) mod FS` in integer
arithmetic and is accurate to 1.6e-16, verified against a 70-digit
series expansion. The tolerance in the harness is therefore sized by
**Python's** error, and says so. `docs/todo.md` has the two-line Python
fix, deferred so the corpus does not churn mid-port.

That has a direct consequence for Phase 1: **reduce phase arguments
exactly wherever they appear**, because `sin`/`cos` of a large argument
disagree between glibc, musl and MSVC by far more than they do near
zero, and every tolerance has to hold on three platforms.

**Resolved on the Python side before Phase 1 continued** (2026-07-28),
deliberately, so the corpus and its tolerances moved once rather than
twice. `dsp.to_baseband` turned out to be far the worst case — its
argument grows over a whole recording, reaching 895,000 rad and 1.5e-10
of phase error, ~5000x the matrices — and also the easiest, since
`FCENTER/FS = 3/16` gives only 16 distinct phasors. Both sides now
reduce, `PHASOR_TOL` tightened from 2e-13 to **1e-14**, and the residual
is one ulp of `exp()` rather than anyone's accumulated error. Details
and measurements in `docs/todo.md`.

**Substitution is by attribute assignment, so from-imports need listing
explicitly.** `sync.py` does `from .ofdm import preamble_template`,
which binds at import time and is invisible to patching `ofdm`. The
table in `tests/conftest.py` names those sites; a missed one means a
test that silently keeps exercising Python while reporting itself as a
native run. Check for new from-import sites as each module is ported.

**Keep golden vectors of the reference's mistakes.** The corpus includes
soft-decode cases noisy enough that the reference decoder is *wrong*.
A port that is right where the reference is wrong has diverged, and only
this kind of case catches it — likewise the tie-breaking test, where
`np.argmax` returning the first maximum is arbitrary but observable.

### Phase 1 — Modem core — **DONE 2026-07-28**

`dsp`, `framing`, `beacon`, `sync`, `modem`. The risk is concentrated
here: `sync.py` + `modem.py` are 622 lines containing the CFO bin
search, the Catmull-Rom pilot EQ, and the deliberately over-smoothed
drift tracker, and they will consume a disproportionate share of the
whole schedule.

**Exit:** the full Python suite including `-m slow` passes against the
C++ modem. Golden vectors match bit-for-bit where pocketfft permits, and
every documented tolerance is justified in a comment.

**Met.** 276 fast and all 19 `-m slow` tests pass under `pytest
--native`, plus 50 A/B parity tests and three C++ binaries (108 checks).
Both interop directions work: a C++ transmission decodes in Python and
vice versa.

The feared part turned out not to be the hard part. **`sync` matched on
the first build** — identical timing index, 1.9e-16 on frequency — and a
72-case sweep (SNR 30 to −2 dB, three frequency offsets, with and
without MPP fading) found **zero differing decisions**, worst frequency
delta 7.1e-15 Hz. Two decisions made that possible rather than lucky:

- `fftconvolve` is FFT-based in C++ too, mirroring scipy's real/complex
  split. A direct moving sum would have been *more* accurate and would
  have diverged *more*, because these feed straight into `argmax`.
- pocketfft's `good_size_cmplx` equals scipy's `next_fast_len`, verified
  across sizes. In `acquire_blind` the transform length sets `bin_hz`
  and therefore the entire CFO search grid, so this is structural rather
  than an optimization detail.

What actually cost time was two reference behaviours that were not
deterministic, both found by the port and both fixed in Python:

- **`beacon.find_sync` sorted with `np.argsort(score)[::-1]`.** A clean
  stream ties *by construction* — every superframe correlates perfectly
  — and numpy's default sort is unstable, so which of several equally
  valid superframes `decode()` returned depended on sort internals. Now
  `kind="stable"`, ties by lowest offset.
- **The frozen-format problem**, described under Phase 0's lessons.

And one thing the *reference test suite* caught that no parity check
would have: `conftest.clip_headroom()` disables clipping by patching a
module constant, to measure the modem's ceiling independent of how the
clipper is tuned. A compiled-in constant is unreachable from there, so
the C++ silently reported the clipped floor (10.1 dB) for an assertion
expecting >30. `Modem::modulate` now takes `clip_headroom_db` as a
parameter. **Substituting into the real suite is what found this** — a
harness of purpose-written parity tests would have agreed with itself.

#### On writing the C++-side round-trip test

`native/tests/test_modem_roundtrip.cpp` is corpus-free: it runs the
whole chain and asserts properties, so `ctest` covers the modem where
the Python reference is unavailable. Worth recording how weak the
obvious version of that test was.

Perturbing the code deliberately showed a latent-SNR threshold catches
almost nothing on a *clean* loopback:

| Perturbation | Caught by SNR floor? |
|---|---|
| Beacon carrier off by one | yes (callsign lost) |
| Catmull-Rom phase `u` shifted 0.05 | **no** |
| Drift-tracker EMA 0.02 → 0.9 | **no** |
| Equalizer scale √2 → 1.0 | **no** |

The first two are honest: on a clean channel the pilots are nearly
identical, so interpolation barely matters and there is no drift to
track. Those paths are exercised by the *parity* tests under fading, not
here. The last one was a genuine hole — a 0.707× amplitude error still
measures 10.7 dB, above any threshold the clip-limited ~10 dB loopback
leaves room for. Fixed by checking the best-fit **gain** separately from
the SNR, which is the quantity that is actually wrong.

The general point: **an SNR floor is not a correctness check.** It
cannot see a scale error, and on a clean channel it cannot see the
equalizer at all.

- **Volume** — 1,034 lines displaced (`dsp` 64, `framing` 134, `beacon`
  214, `sync` 198, `modem` 424). Gated on `test_modem_e2e.py`,
  `test_beacon.py`, `test_blind_acquisition.py`, and the slow
  `test_listen_state_machine.py`.
- **Verified by** — CI and golden vectors, completely. This is the
  reassuring property of the riskiest phase: the hardest code in the
  project is also the most mechanically checkable, because Python
  already computes the right answer for any input you care to try.
- **Needs you for** — judgement calls on tolerances where pocketfft and
  scipy legitimately differ, and on anything that turns out to be an
  accident of the Python implementation rather than part of the format.

### Phase 2 — Headless app core — IN PROGRESS

`codec` (onnxruntime), `checkpoint` (Hub fetch), `images`,
`overlay/render`, `audio` (PortAudio), `rig` (linked `libhamlib`),
`settings`, and the `rx`/`tx` engines.

**`codec` is done.** It was taken first deliberately: it is the only
module in the phase with a heavyweight new external dependency, and
finding out late that onnxruntime could not be made to work cleanly on
three platforms would have meant arranging the rest of the phase around
a hole.

What the spike settled, before any structure was committed to:

- Official prebuilt CPU archives exist per platform at 9–80 MB, ship a
  usable CMake package, and are pinned by sha256. Building ORT from
  source — hours, and its own dependency tree — is not necessary.
- **There is no `osx-x64` archive.** The macOS build is Apple silicon
  only. That compounds the `macos-13` retirement noted in the matrix
  below: Intel Mac support now needs cross-compilation *and* a
  cross-built ORT, which is a Phase 4 packaging problem rather than a
  Phase 2 one, but it is worth knowing now.
- Parity is **exact**, not tolerance-bounded, because both sides call
  one runtime on one file. That is a stronger guarantee than the modem
  gets, and it holds only while the versions match — hence the pin.

The one real defect the spike found was a 1-LSB difference in 3 of
921600 subpixels, from doing the output quantisation in float64 where
numpy does it in float32. Worth recording not for the fix but for the
shape of it: **it was well inside any tolerance anyone would have
chosen**, and the only reason it was caught is that the test demanded
byte equality. When a check *can* be exact, making it exact is what
turns a near-miss into a finding.

**`images` is done, with one deliberate inexactness.** `to_array` and
PNG loading are checked byte-for-byte; **`fit_image`'s rescaling is
not**. Pillow's LANCZOS turned out to be exactly reproducible — a port
of `precompute_coeffs` plus the two fixed-point passes matched it on
every subpixel across four source geometries — so an exact version was
available and was *declined*, on the grounds that framing is
transmit-side and cosmetic: it chooses which pixels of an oversized
source get sent, a receiver never runs it, and two stations cropping a
photo one pixel differently is not an interop question. `stb` does the
resize instead, and parity tests feed already-640x480 pictures so the
resampler is off the compared path.

That is the counterpart to the decision above, and the pair is the
useful thing: exactness was bought where it lands in the delivered
picture and skipped where it does not. Deciding that per module beats
either blanket policy.

Vendoring `stb` rather than reaching for `QImage` also keeps `core/`
Qt-free for the whole phase, so the headless CLI links no GUI toolkit
and CI still installs Qt only in Phase 3 — which is what the workflow's
comments already promise.

**The exit criterion is met for the receive path.**
`native/apps/sstvae_decode.cpp` produces a picture **byte-identical** to
`sstvae_decode.py`'s on the same WAV, over a 12 dB AWGN channel — so
sync, CFO, pilot EQ, drift tracking, the beacon and the codec all agreed
exactly, not merely closely. Also byte-identical on three variants that
each exercise something the plain case does not: a 44.1 kHz recording
(the 160/441 resample), a stereo int16 file (the scale-before-mixdown
rule), and a mid-transmission slice decoded blind (position from the
beacon's absolute counter alone). `tests/test_native_cli.py`.

Two things this required that were not in the plan:

- **`dsp::resample_poly`**, matching scipy's design exactly (Kaiser(5.0)
  firwin, `h *= up`, scipy's pre/post padding and slice). Verified
  against scipy to ~1e-14 with exactly matching output lengths. It was
  needed for `wavio` and will be needed again for capture — where
  CLAUDE.md records that getting it wrong cost 4.7 dB while still
  reporting every frame received.
- **`dr_wav`** rather than a hand-rolled RIFF parser, because the files
  this reads were written by whatever the operator had.

The trap worth recording: the slow suite runs under `--native`, which
substitutes the C++ modem *in-process*. The blind test's "Python
reference" was an in-process `Modem().demodulate_blind`, which under CI
would have been the C++ implementation — the test would have compared
C++ against C++ and passed while checking nothing. Every reference-side
decode in `test_native_cli.py` now runs in a **subprocess**, where no
substitution is applied. Any future test that wants a reference
alongside `--native` has the same problem.

Still open in this phase: `checkpoint` (so `--model` can be optional),
`settings`, `overlay`, the rx/tx engines, `audio` and `rig`.

Because artifacts are fetched rather than bundled (decision 5), this
phase owns the **first-run experience**: a progress indication, a
checksum check, and — importantly for a field laptop with no
connectivity — a clear failure message plus a manual model-import path.
`checkpoint.py`'s existing error text is the model to follow; it already
tells an offline user exactly what to do.

`rx/engine.py`'s `decode_loop` is described in `CLAUDE.md` as
load-bearing and unchanged since the slow tests were written against it.
Port it *literally*, including the `RxConfig`/`sink` seams and the rule
that **saving is the sink's job, not the loop's**. Likewise `tx/engine`'s
invariant: PTT always comes back down, via try/finally *plus* the
independent watchdog thread.

**Exit:** a headless CLI that takes a WAV and produces a picture,
byte-comparable to `sstvae_listen.py` on the same input; engine tests
ported; settings and overlay JSON round-trip existing user files
unchanged.

- **Volume** — 2,144 lines displaced (`rx/` 641, `tx/engine` 297,
  `overlay/` 340, `rig/rigctld` 302, `audio` 223, `gui/settings` 194,
  `images` 87, `codec` 60). Gated on `test_audio.py`,
  `test_rigctld.py`, `test_settings.py`, `test_overlay.py`,
  `test_rx_ringbuffer.py`, `test_tx_engine.py`.
- **Verified by** — CI for everything except `audio`, which is the one
  module in this phase whose real behaviour lives in hardware. The
  fake-PortAudio harness covers the known trap; it cannot cover the
  unknown ones.
- **Needs you for** — a real soundcard the first time the PortAudio path
  runs outside a fake, and a rig for the `libhamlib` path. Note that
  `rig/rigctld` 302 is *displaced*, not reimplemented — most of it
  becomes library calls, so this phase's line count overstates its
  rig-control work.

### Phase 3 — GUI

App shell with native menus, waterfall, transmit panel, receive panel,
settings dialog, overlay editor. The 1,802 lines of PySide6 port to Qt
Widgets nearly line-for-line — this is the phase that most rewards
having chosen Qt.

The waterfall already blits a numpy RGB buffer into a `QImage`; that
code barely changes. The overlay editor's `QGraphicsView` with drag and
resize handles ports directly, where any other toolkit means rebuilding
the scene, the hit-testing, and the handles from scratch.

**Can start during Phase 1**, against recorded WAVs and a stubbed codec.
The GUI work is independent of the modem, and running it early is how
you get something demo-able before `sync.cpp` works.

**Exit:** feature parity with `sstvae-gui`, and the loopback equivalent
of `test_app_loopback.py` passes. **Met 2026-07-29**: a picture sent and
received native->native, native->Python and Python->native over a
soundcard loopback. The two mixed directions are the ones with content —
they check the on-air format against the normative implementation end to
end, through the audio path, which no golden vector can do. Parity **freezes** the Python GUI —
bug fixes only from here, every new feature goes to the native app — but
does *not* delete it; that is Phase 4's, under the amended decision 1.
Do not let this phase idle at 95%: a frozen GUI is a bounded cost, a
nearly-finished one is not.

- **Volume** — 2,162 lines displaced (`rx_panel` 417, `settings_dialog`
  400, `tx_panel` 347, `overlay_editor` 309, `waterfall` 252, `app` 229,
  `rig_controller` 195). Gated on `test_waterfall.py`,
  `test_overlay_editing.py`, `test_rx_panel_save.py`,
  `test_gui_sink.py`, `test_rig_controller.py`,
  `test_settings_dialog_rig.py`, `test_app_menu.py`, and the slow
  `test_app_loopback.py`.
- **Verified by** — CI under `QT_QPA_PLATFORM=offscreen` for behaviour;
  **your eyes** for everything that makes a GUI good. Automated tests
  cannot tell you a dialog is laid out badly or a waterfall is
  unreadable, and this is the largest single block of code in the
  project.
- **Needs you for** — look-and-feel review, iterated. Realistically the
  phase with the highest ratio of your attention to lines of code.

### Phase 4 — Packaging, signing, CI

- **Windows:** `windeployqt` → WiX or NSIS → **Azure Trusted Signing**.
  The EV-token era is over for CI purposes; SignPath is the alternative.
- **macOS:** universal2 via `lipo` from `macos-14` + `macos-13` builds,
  Developer ID cert from a base64 secret, hardened runtime. **Sign
  inside-out manually** — `codesign --deep` is deprecated and will bite
  you with Qt frameworks — then `notarytool submit --wait` and
  `stapler`. Prefer a statically linked onnxruntime to avoid signing an
  extra nested dylib.
- **Linux:** build on `ubuntu-22.04` for the glibc floor. AppImage via
  `linuxdeploy`, plus a Flatpak on Flathub — which is realistically the
  best Linux install UX available and worth the extra manifest.
- `.desktop` and AppStream metainfo files.
- **No auto-update** (decision 4). Distribution channels are winget,
  Homebrew cask, Flathub, and direct download.
- `libhamlib` ships as a linked library, so there is no nested
  executable to sign — one of the reasons for choosing it over bundling
  `rigctld`.

**Exit:** a tagged GHA run produces three signed artifacts, each
installed and launched on a clean VM with no developer tooling present.
**That is what triggers decision 1** — in the same change, `sstvae/gui/`
is deleted along with the `gui` extra and `pyside6-essentials`, and
`CLAUDE.md`'s "The application" section is rewritten to describe the C++
app. In the same change and not as a follow-up, for the reason the
original decision gave: a deletion deferred is a deletion that does not
happen.

- **Volume** — almost no Python displaced; this is nearly all new
  configuration. Small in lines, and that is exactly why it will not
  compress the way Phases 0–3 do.
- **Verified by** — Apple's notary service, Azure Trusted Signing, and
  clean VMs. **External systems on their own schedule**, with slow and
  frequently unhelpful error messages. Iteration here is measured in
  round-trips, not in builds.
- **Needs you for** — credentials, Apple Developer and Flathub account
  actions, and the clean-VM installs. Very little of this phase can be
  delegated at all.

### Phase 5 — Hardware and on-air

Audio device matrix (the K4 USB codec case specifically), real-radio PTT
timing against a physical rig — which `CLAUDE.md` still lists as
outstanding for the Python app too — and on-air interop against a
Python-app counterpart.

**Exit:** a picture sent from the C++ app and received by the Python app
over the air, and the reverse.

- **Volume** — no Python displaced. Zero new features. Entirely a
  matter of finding out what is wrong.
- **Verified by** — a physical radio, a set of soundcards including the
  K4's USB codec, and a counterpart station. **Irreducible.** No harness
  substitutes for it, which is why the Python app still has this item
  outstanding.
- **Needs you for** — all of it.

### Where the cost actually sits

| Phase | Python displaced | Closed by |
|---|---|---|
| 0 Scaffolding and parity harness | 135 | CI |
| 1 Modem core | 1,034 | CI + golden vectors |
| 2 Headless app core | 2,144 | CI, except `audio` |
| 3 GUI *(can start during Phase 1)* | 2,162 | CI + your judgement |
| 4 Packaging, signing, CI | ~0 | External services |
| 5 Hardware and on-air | 0 | A radio |
| **Total** | **5,475 lines** | |

Read that table by column, and the shape of the project is the opposite
of what a week-based plan implies.

**Phases 0–3 are 5,475 lines and almost entirely machine-verifiable.**
Every one of them terminates in a test that either passes or does not,
and the oracle already exists — Python computes the right answer for any
input. This is the part that compresses.

**Phases 4–5 are nearly zero lines and barely compress at all.** They
are gated on Apple's notary service, on signing credentials, on a
soundcard's firmware, and on another operator being on frequency. Their
cost is round-trips against systems that do not go faster because the
code was written faster.

So the honest planning statement is: **the bulk of the code is the part
that will go quickly, and the tail is the project.** Anyone budgeting
this by counting lines will underestimate Phase 4 by an embarrassing
margin, and anyone who ships without Phase 5 will find out what is wrong
from a stranger on 20 metres.

One consequence worth acting on: **do a throwaway notarization and
signing spike in Phase 0**, on a hello-world Qt app. It is the cheapest
possible way to move the least compressible work earlier, and Phase 0
already needs the CI matrix stood up.

## Risks

| Risk | Mitigation |
|---|---|
| **`sync.py` is ported wrong in a way that only shows on air** | Golden vectors and the pybind11 harness exist before Phase 1 starts, so every function is validated as it is written rather than at integration. |
| **Silent on-air divergence** | Four mechanisms above, of which the interop CI job is the one that models reality. |
| **`resample_poly` mismatch** | Explicitly called out as delicate; test against scipy over a matrix of rate pairs, not just 8k↔48k. |
| **Audio device quirks** | PortAudio continuity retires most of it. Port `tests/test_audio.py`'s fake-PortAudio harness early — it caught the resample-direction bug without hardware and will catch its C++ twin. |
| **macOS notarization archaeology** | Static-link onnxruntime; sign inside-out; do a throwaway notarization spike in Phase 0 rather than discovering the problems in Phase 4. |
| **Two GUIs to maintain forever** | Bounded by decision 1 as amended: the Python GUI is frozen at parity (bug fixes only) and deleted at packaging. Freezing is what caps the cost — the window between parity and packaging is real, but during it only one GUI is being *developed*. The residual risk moves to a Phase 4 that stalls, so the deletion is part of the packaging change rather than a follow-up to it. |
| ~~**GHA retires the Intel macOS runner mid-project**~~ — **happened, 2026-07-28** | Already gone by the time CI was first stood up: `macos-13` no longer schedules, and a job requesting it **queues indefinitely rather than failing**, which is worse than an error because nothing tells you why. Removed from the matrix. Cross-compiling the `x86_64` slice on an Apple-silicon runner is therefore mandatory, not a fallback — see "Platform floor". Decision 2 is unchanged: Intel Macs are still supported, and the gap is in CI coverage, not in intent. |
| **First run fails offline** | Decision 5 trades installer size for a network dependency at first launch. Phase 2 owes a clear message and a manual model-import path; a field laptop with no connectivity is an ordinary case, not an edge one. |
| **Bundled Hamlib goes stale** | CI bump, as agreed. Note that bundling means a Hamlib CVE or a new-radio backend becomes our release, not the distro's. Pinned in `native/cmake/hamlib.cmake` (4.7.2), sha256 per artifact, built from the release tarball on Linux/macOS and taken from upstream's prebuilt zip on Windows. |
| **A Hamlib backend segfaults and takes the app with it** | Accepted cost of in-process linking. Mitigated by Hamlib's exposure across the ham software ecosystem, and by model 2 as an out for users who want isolation back. |
| **`config.py` drift** | Generated header, CI-enforced. |

## Decisions (Andrew, 2026-07-27)

1. **The Python GUI is retired** once the native app is **packaged** —
   amended 2026-07-29, from "once it reaches parity". `sstvae/gui/` is
   deleted, the `gui` extra and `pyside6-essentials` go with it, and
   Python keeps the CLIs, the reference modem, and training. The
   `listen` extra stays. This is part of the plan, not a follow-up — see
   Phase 4's exit criteria.

   The amendment is about what a user can actually run. Parity means the
   native app *works*; packaging means someone who is not us can install
   it. Between those two points the only way to have a working desktop
   app is to build C++, Qt and Hamlib from source — so deleting the
   Python GUI at parity would leave every operator without one, to
   remove code that costs nothing to keep for another phase. The trigger
   is therefore a signed, installable build on all three platforms.

   To stop that becoming an indefinite reprieve, the Python GUI is
   **frozen at parity**: bug fixes only, no new features. Anything new
   goes to the native app alone. That bounds the duplicated-maintenance
   cost the original decision was written to avoid, without paying it
   with a period where nothing is installable.
2. **Windows 10 and Intel Macs are supported.** Hams are conservative
   about hardware, so the floor is set by users, not by tooling
   convenience. Consequences in "Platform floor" below.
3. **Hamlib is bundled, linked in-process as `libhamlib`.** Users
   install nothing, there is no child process and no nested binary to
   sign, and sharing a radio with other software is covered by Hamlib
   model 2 rather than by a second transport. See "Bundling Hamlib".
4. **Auto-update is out of scope.** Distribution is via winget,
   Homebrew cask, and Flathub, plus direct downloads. Sparkle /
   WinSparkle can be added later without architectural change, so this
   is a deferral rather than a door closing.
5. **Model artifacts are fetched from the Hub on first run**, not
   bundled. Keeps `checkpoint.py`'s immutable-filename model intact and
   takes ~20.7 MB out of the installer. Requires a first-run network
   path and a graceful offline story — see Phase 2. `docs/onnx.md`
   originally said to ship fp16 *in* the packaged distributions; it was
   revised on 2026-07-27 to match this, and fp16 remains the precision
   used — only the delivery changed.
6. **No commercial Qt licence.** LGPLv3, dynamically linked,
   unmodified. Static Qt is off the table permanently.
7. **The overlay editor targets today's feature set.** A richer editor
   with template saving is wanted later but is not specified well enough
   to build against. The obligation this creates is *preservation*, not
   implementation — see "Overlay format" below.

### Platform floor

- **Qt version is constrained by the Windows 10 floor.** Qt has been
  trimming support for older Windows releases in recent versions;
  **verify the exact floor for the chosen Qt against Windows 10 in
  Phase 0**, before the CI matrix is fixed. Qt 6.8 LTS is the
  conservative pick if the newer releases have moved on.
- **macOS ships universal2**, built by `lipo`-ing an `x86_64` and an
  `arm64` slice. The deployment target is set by the oldest Intel Mac
  worth supporting, and it also constrains the Qt version — pick both
  together in Phase 0.
- **The Intel-macOS runner is gone (confirmed 2026-07-28).** This was
  written as a thing to watch for; it had already happened. `macos-13`
  no longer schedules — and note the failure mode, because it costs an
  afternoon if you meet it cold: the job **sits in the queue forever
  instead of erroring**, so the run neither passes nor fails.
  Consequently there is no way to build or test the `x86_64` slice
  natively. Cross-compiling it on an Apple-silicon runner is now the
  only route, which needs a universal Qt install and a tested `lipo`
  step. **Prove that step early**, since Decision 2 commits to Intel
  Macs and nothing in CI currently exercises them at all.
- **Linux builds on the oldest supported GHA Ubuntu image** for the
  glibc floor, for the same conservative-user reason.
- No Windows-on-ARM target.

### Bundling Hamlib

**Link `libhamlib` in-process.** No child process, no IPC, no nested
binary to sign, no per-platform `rigctld` build in CI.

The `rigctld` architecture was chosen in Python for a reason that does
not survive the port: the SWIG `Hamlib` bindings live in the system
site-packages and a virtualenv cannot see them (`CLAUDE.md`). C++ has no
such problem, so the constraint that produced that design is gone and
the design should be re-derived rather than inherited.

This deletes most of `rig/rigctld.py` outright rather than porting it —
`spawn_rigctld`, the socket client, the redial logic, and the `rigctld -l`
parser all have direct library equivalents.

#### Preserving the property that matters

The thing `gui/rig_controller.py` exists to guarantee is that **nothing
on the GUI thread ever blocks on the rig, and keying is never stuck
behind a stale poll.** That is preserved as follows:

- **One `RIG*`, owned by a dedicated `RigThread`.** Every call is
  submitted to it as a job; the GUI thread only ever posts and receives
  signals. The handle is never touched from anywhere else, which also
  sidesteps `RIG*` not being thread-safe.
- **PTT is a priority job, and enqueuing it drops pending polls.** A
  queued poll is stale by definition — its answer is a frequency
  readout, not something worth waiting on. Worst-case keying latency is
  therefore *one in-flight operation*, not a queue drain.
- **Polling suspends during transmit.** The app is already half duplex
  and already emits `transmitStarted` to suspend receive; the same
  signal suspends the poll loop, so in the normal case PTT contends with
  nothing at all.
- **Timeouts are set explicitly** via `rig_set_conf` (`timeout`,
  `retry`), tuned low. This is an *improvement* on today: the current
  stack has an app-side socket timeout and retry layered on top of
  rigctld's own rig timeout and retry, and only the outer pair is under
  our control.
- **`stop()` detaches, it does not join.** A thread stuck in a blocking
  serial read is abandoned and self-cleans when Hamlib's timeout
  expires, so its exit path must be self-contained and own closing the
  handle. Re-opening the port waits on that, which is why the timeout
  configuration above is load-bearing rather than incidental.

Note that the serial port serializes access in **both** designs — one
radio, one port, one command in flight. `rigctld` never removed that
contention; it added a socket layer in front of it, and the
separate-PTT-client trick addressed contention introduced by that layer.
Removing the layer removes the need for the trick.

#### Sharing a radio with other software still works

Hamlib **model 2, "Hamlib NET rigctl"**, is a backend that speaks the
rigctld protocol as a *client*. So connecting to a `rigctld` the user
already runs — the usual way a radio is shared with WSJT-X or fldigi —
is just another model in the same picker, opened on the same `RIG*` code
path with a `host:port` in place of a serial device. No port conflict,
because we are a client and not a server, and no second transport in the
codebase.

#### A bonus: `list_models()` gets much better

`rig_list_foreach()` replaces parsing `rigctld -l`. That deletes the
fixed-width column slicing which `CLAUDE.md` flags as a trap — *"splitting
on whitespace runs looks fine and silently drops rows, because fields
contain single spaces and at least one Model fills its column exactly"* —
along with the entire class of bug it warns about. Model metadata comes
from a struct instead of from text, and model 2 appears in the list for
free.

#### Pinned, not taken from the system

Implemented 2026-07-29, and the pin is load-bearing rather than tidy.
Hamlib's public API moves between minor releases: Ubuntu 24.04 ships
4.5.5, where a configuration token is a `token_t`, and 4.6 renamed it
`hamlib_token_t`. `core/rig/hamlib.cpp` therefore compiled against a
developer's 4.7 and failed on the CI runner. Papering over that with
version `#if`s would mean the app's rig support silently differs by
platform — and for CAT control, "which radios work" differing by
platform is the whole ballgame.

So one version is pinned with a sha256 per artifact, exactly as
onnxruntime is: built from the release tarball on Linux and macOS
(upstream publishes no binaries for them), and taken from upstream's
prebuilt zip on Windows, which ships an MSVC import library beside the
MinGW-built DLL so linking from MSVC is supported rather than a trick.

**Dynamically linked**, following decision 6's reasoning for Qt: Hamlib
is LGPL-2.1+, so a shared library the user can replace avoids the
relinking obligation a static link would carry. That is why the source
build passes `--enable-shared --disable-static` even though a static
link would be easier to ship.

`-DSSTVAE_HAMLIB_SYSTEM=ON` uses pkg-config instead and downloads
nothing, for distro packagers. That configuration is not what CI checks,
so it is on the packager to confirm their Hamlib is >= 4.6.

Windows adds one more constraint, and it shapes the code rather than the
build: `hamlib/rig.h` includes `<pthread.h>` unconditionally (upstream
suggests the NuGet pthread package), MSVC has none, and the types it
wants sit inside `struct rig_state`. A shim supplies them, but its sizes
cannot be guaranteed to match the winpthreads inside the MinGW-built
DLL — so `core/rig/hamlib.cpp` never dereferences a `RIG*`, taking the
manufacturer and model from `rig_get_caps_cptr()` instead. With no
struct layout relied upon, a mismatch cannot corrupt anything.

Two things the source build has to work around, both recorded because
they cost time: the release tarball's files all carry the same mtime, so `make`
cannot tell `configure` is already newer than `configure.ac` and tries
to re-run `aclocal` — which fails without the exact automake the release
was rolled with. Hamlib has no `AM_MAINTAINER_MODE`, so the generated
files are re-stamped in dependency order first. And the install tree
goes under `FETCHCONTENT_BASE_DIR`, not the build directory, so that
whatever caches the former caches the *built* Hamlib: cold 26 s on a
24-core machine, warm 0.26 s, and CI throws its build tree away every
run.

#### What is genuinely given up

**Crash isolation.** A segfault in a Hamlib backend now takes the app
down instead of a child process. Accepted: these backends are widely
exercised, and a *silently hung* child is arguably worse UX than a
visible crash. Users who want the isolation back can run their own
`rigctld` and select model 2.

#### Testing

`tests/test_rig_controller.py`'s value is the scenario, not the
transport: a rig that accepts and never answers. Port it against a fake
Hamlib backend, or against model 2 pointed at the existing
never-replies test server — which keeps the current harness almost
unchanged. The properties under test are the same: the GUI thread never
blocks, `stop()` returns promptly, and keying is not stuck behind a
poll.

### Overlay format

Targeting today's feature set does **not** mean the C++ port may
simplify `overlay/model.py`. The properties that make templates a
UI-only change later are load-bearing and must survive the port
unchanged:

- Coordinates stay normalized 0..1, so a document is
  resolution-independent.
- `ImageItem.source` stays a **late-bound reference** (`"last_rx"` or a
  path), never a pasted bitmap, so a saved document keeps meaning "the
  most recent received picture".
- `item_bbox` stays shared between the renderer and the editor, so
  selection handles cannot drift from what is drawn.
- The JSON round-trips existing files byte-for-byte.

Build the format for the editor you want; ship the editor you have.
