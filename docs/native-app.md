# Native desktop application (C++ / Qt 6)

**Status: design, not implemented.** Nothing in this document exists in
the tree yet. It records the plan, the decisions behind it, and the
questions still open, so none of that has to be re-derived when work
starts.

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

`sstvae/gui/` is **deleted** when the native app reaches parity
(decision 1), taking the `gui` extra and `pyside6-essentials` with it.
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

**PortAudio continuity.** `sounddevice` *is* PortAudio. Staying on it
means `audio.py`'s hard-won behaviour — device enumeration, the
8 kHz-rejected → native-rate + resample fallback, and `resample_ratio`'s
deliberately two-named-rates signature — ports nearly literally. The
gotcha list in `CLAUDE.md` records that this cost a 32-second
transmission going out as 0.9 s of noise, discoverable only on real
hardware (an Elecraft K4's USB codec takes that path; PulseAudio's
`default` does not). Any other audio abstraction re-derives that class
of bug from scratch.

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
| Audio | **PortAudio** | See above. |
| Rig | **`libhamlib`, linked** | User installs nothing. One `RIG*` on a dedicated thread; model 2 covers sharing a radio via someone else's `rigctld`. Deletes the socket client and the `rigctld -l` parser. See "Bundling Hamlib". |
| Hub fetch | `QNetworkAccessManager` | Ports `checkpoint.py`'s cache-first, immutable-filename model — a cache hit is trusted outright, no revalidating HEAD. |
| Images, text, fonts | Qt (`QImage`, `QPainter`, `QFontDatabase`) | Replaces Pillow entirely. |
| JSON | `QJsonDocument` | Settings and overlay documents; must round-trip the existing files unchanged. |
| Bindings | pybind11 | Both directions — see below. |

**Do not use FFTW.** GPL-2+ unless licensed commercially, which fights
the project's Artistic 2.0 distribution.

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
    audio/              # PortAudio
    rig/                # QTcpSocket
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

### Phase 0 — Scaffolding and parity harness

CMake + vcpkg manifest, GHA matrix, `install-qt-action`, ccache,
ASan/UBSan job. `gen_config_header.py` and its CI check. Golden-vector
generator and committed corpus. pybind11 module skeleton. Port `golay`
and `ofdm` — small, completely testable, and they prove the whole loop
end to end before anything hard is attempted.

**Exit:** `pytest` passes with C++ `golay` and `ofdm` substituted via the
pybind11 module, on all three platforms in CI.

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

### Phase 1 — Modem core

`dsp`, `framing`, `beacon`, `sync`, `modem`. The risk is concentrated
here: `sync.py` + `modem.py` are 622 lines containing the CFO bin
search, the Catmull-Rom pilot EQ, and the deliberately over-smoothed
drift tracker, and they will consume a disproportionate share of the
whole schedule.

**Exit:** the full Python suite including `-m slow` passes against the
C++ modem. Golden vectors match bit-for-bit where pocketfft permits, and
every documented tolerance is justified in a comment.

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

### Phase 2 — Headless app core

`codec` (onnxruntime), `checkpoint` (Hub fetch), `images`,
`overlay/render`, `audio` (PortAudio), `rig` (linked `libhamlib`),
`settings`, and the `rx`/`tx` engines.

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
of `test_app_loopback.py` passes. **Parity here is what triggers
decision 1** — `sstvae/gui/`, the `gui` extra, and `pyside6-essentials`
are removed in a single change, and `CLAUDE.md`'s "The application"
section is rewritten to describe the C++ app. Until that change lands,
both GUIs are maintained, so do not let this phase idle at 95%.

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
| **Two GUIs to maintain forever** | Resolved by decision 1: `sstvae/gui/` is deleted at parity, in the same change. The residual risk is a Phase 3 that stalls just short of parity and leaves both alive indefinitely. |
| **GHA retires the Intel macOS runner mid-project** | Build universal2 from Phase 0, so the `lipo` step is already proven when the fallback (cross-compiling `x86_64` on Apple silicon) becomes necessary rather than optional. |
| **First run fails offline** | Decision 5 trades installer size for a network dependency at first launch. Phase 2 owes a clear message and a manual model-import path; a field laptop with no connectivity is an ordinary case, not an edge one. |
| **Bundled Hamlib goes stale** | CI bump, as agreed. Note that bundling means a Hamlib CVE or a new-radio backend becomes our release, not the distro's. |
| **A Hamlib backend segfaults and takes the app with it** | Accepted cost of in-process linking. Mitigated by Hamlib's exposure across the ham software ecosystem, and by model 2 as an out for users who want isolation back. |
| **`config.py` drift** | Generated header, CI-enforced. |

## Decisions (Andrew, 2026-07-27)

1. **The Python GUI is retired** once the native app reaches parity.
   `sstvae/gui/` is deleted, the `gui` extra and `pyside6-essentials`
   go with it, and Python keeps the CLIs, the reference modem, and
   training. The `listen` extra stays. This is part of the plan, not a
   follow-up — see Phase 3's exit criteria.
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
- **Watch the GitHub Actions Intel-macOS runner.** GHA has been retiring
  older runner images, and Intel macOS is on that path. If the `macos-13`
  class disappears, the fallback is cross-compiling the `x86_64` slice
  on an Apple-silicon runner, which Qt supports but which needs a
  universal Qt install and a tested `lipo` step. **Build the universal
  binary from day one in Phase 0** rather than discovering this at
  Phase 4, when it becomes urgent.
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
