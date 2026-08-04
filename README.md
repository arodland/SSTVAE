# SSTVAE

**Send pictures over HF radio using a neural image codec instead of scanlines or file bytes.**

SSTVAE is an experimental amateur-radio image mode. A convolutional
autoencoder compresses your image into a few hundred thousand
*continuous-valued* numbers ("latents"), and those numbers are sent
directly as OFDM carrier amplitudes — no bits, no packets, no JPEG. The
decoder network is trained with a simulated HF channel in the loop, so
noise and fading push the picture quality gently downhill instead of
breaking it.

It is directly inspired by [FreeDV RADE](https://freedv.org/radio-autoencoder/)
(the "radio autoencoder"), which does the same trick for speech.

![Sent and received images over a 3150 km path to Utah](docs/images/ota-20m-irises.png)

![Sent and received images over a 1700 km path to Minnesota](docs/images/ota-20m-airplane.png)

**Actually over the air.** Mode B (64 s) on 20 m, transmitted from New
Jersey. Top row received at the
[Northern Utah WebSDR](http://www.sdrutah.org/) in Corinne, Utah —
**3150 km**. Bottom row received in Minnesota — **1700 km**. In both,
the middle panel is 100 W and the right panel is **10 W**, sent back to
back over the same path. One tenth the power costs about 1 dB of
picture: it gets slightly softer rather than breaking up, which is the
whole idea.

> ### ⚠️ Status: working beta
>
> It works on real HF, as above. But:
>
> - **The on-air format is not frozen.** Expect incompatible changes.
>   Two stations must run the same commit *and* the same model
>   checkpoint to talk to each other.
> - **There is no release download yet.** The builds are signed and
>   notarized now, but they are still published as CI artifacts rather
>   than on a releases page. See [Get the app](#get-the-app).
> - There is no integration with existing SSTV software.
> - Not registered with, or coordinated with, any band-plan authority.
>   Use it thoughtfully and identify per your licence.
>
> If you want something that just works and interoperates with the rest
> of the world today, use MMSSTV or an existing digital SSTV mode. If
> you want to help shake out a new one, read on.

## What makes it different

**Analog SSTV** fails softly — noise looks like noise — but QRM tears a
band across the image, lost sync slants the frame, and tuning in late
means the top of the picture is simply gone.

**HamDRM** and **SSDV** send a compressed file. While the FEC holds, the
picture is pixel-perfect; when it doesn't, you lose whole blocks or the
image outright. That's the digital cliff: excellent, excellent,
excellent, nothing.

SSTVAE aims at the middle. Latents are real numbers, so channel noise
adds to them instead of corrupting a bitstream — there's no threshold to
fall off. And a fixed interleaver scatters every frame's latents across
the whole picture, so losing part of a transmission dims detail
*everywhere* slightly rather than removing a region.

![SSTVAE mode B and analog Scottie 2 received over the same path](docs/images/ota-vs-analog.png)

The same picture between the same two stations, on the same frequency,
100 W both, a few minutes apart — and comparable airtime, 64 s against
71 s. The analog copy shows what the channel did to it: impulse noise as
coloured streaks, speckle through the sky, softened detail. The SSTVAE
copy absorbed the same channel into a slightly softer picture. (One
comparison on one path at one moment isn't a controlled study, and
Scottie 2 is 320×256 *by design* — it isn't failing, it's doing its job
at its own resolution.)

Three modes, **32 / 64 / 95 s**, in **~1200 Hz**, at 640×480 — the same
airtime and bandwidth as the common analog modes. Across the range where
acquisition is reliable, 20 dB down to 3 dB SNR, the picture gives up
just **1.4 dB of PSNR for a 17 dB drop in channel SNR**. That gentle
slope is the entire point of the design. The cliff hasn't been abolished
so much as moved off the picture and onto acquisition: below about 0 dB
you increasingly get *no* picture rather than a poor one — which is a
much better place for it.

Two things it costs you:

**You need the model.** The network *is* the codec, so both stations
need the same ~40 MB checkpoint — an interoperability cost analog SSTV
doesn't have. It's fetched automatically on first use.

**It's a reconstruction, not a photograph.** The decoder produces the
most plausible image consistent with the latents that arrived. Fine
detail, small text especially, can come back subtly wrong rather than
merely blurry — and it looks confident either way. Judge it by how the
pictures look, not the pixel count, and don't use it where exactness
matters.

→ The measurements are in
[Performance](https://github.com/arodland/SSTVAE/wiki/Performance), the
row-by-row table against analog SSTV, HamDRM and SSDV in
[Comparison with other modes](https://github.com/arodland/SSTVAE/wiki/Comparison-with-other-modes),
and the beacon, nested modes and waveform in
[How it works](https://github.com/arodland/SSTVAE/wiki/How-it-works).

## Get the app

**Most people want the desktop app.** It transmits, receives, keys the
rig and remembers how it is set up, and it needs no Python at all. It's
a native C++/Qt program, built for **Linux x86-64, Linux arm64, macOS
(Apple Silicon and Intel) and Windows x64** on every push, as both a
portable archive and a platform installer (AppImage, `.dmg`, NSIS
setup).

> **No releases page yet.** The builds are code-signed — macOS with a
> Developer ID and notarized by Apple, Windows with Azure Trusted
> Signing — but there is still no tagged release, so the only downloads
> are CI artifacts and GitHub requires you to be logged in to fetch
> them. Open the latest green run of [the CI
> workflow](https://github.com/arodland/SSTVAE/actions/workflows/ci.yml),
> scroll to **Artifacts**, and take the one matching your platform
> (`sstvae-linux-x86_64`, `sstvae-macos-arm64`,
> `sstvae-windows-x64-portable`, …) or its installer.
>
> macOS should no longer warn about an unidentified developer: the
> notarization ticket is stapled into the `.dmg`, so Gatekeeper can
> approve it without asking Apple, and without a network connection.
> On Windows the publisher is now named rather than unknown, but
> **SmartScreen may still interrupt the first install** — its
> reputation is earned per certificate over downloads and time, and
> this one is new. "More info" → "Run anyway" is the way past it, and
> it should stop happening on its own.

The published model is fetched on first use and cached, so there is
nothing else to download. Building from source is an ordinary CMake +
Qt 6 build — see
[Development](https://github.com/arodland/SSTVAE/wiki/Development).

## Using it

One window that transmits and receives on a soundcard, keys the rig, and
remembers how it is set up.

- **Receive** — waterfall with the SSTVAE band marked and an input level
  meter, the picture building up as it arrives, autosave or a Save
  button. Filenames follow a template, e.g.
  `2026-07-26_011542Z_14.340MHz_N0CALL.png`; the frequency comes from
  the rig and is simply left out when there is no rig control.
- **Transmit** — pick a picture, then drag station text and an inset of
  the last received image onto it. The preview *is* the render, so what
  you arrange is exactly what goes on the air. Mode A/B/C with their
  durations, a progress bar, and Cancel.
- **Rig control** — PTT and frequency readback through Hamlib, which is
  bundled, so there is nothing to install. Receive pauses while you
  transmit, so your own signal is never decoded back as a reception.
  Optional: leave it off and use VOX or manual PTT.

Route receiver audio to the computer however you normally would for
digital modes — a rig interface, a VAC/VB-Cable loopback, a
`pulse`/PipeWire monitor source, or just a microphone near the speaker.

<details>
<summary>Rig control setup</summary>

Settings → Rig control is modelled on WSJT-X's Radio tab, because that
is the set of knobs a real radio needs and the one you already know:
serial port and baud, data/stop bits, parity, handshake, forced DTR/RTS,
PTT method (VOX/CAT/DTR/RTS) with its own port, and an optional
USB/PKT-USB mode on connect. Everything defaults to **Default**, which
means *leave the backend's own value alone* rather than forcing one.

The **Rig model** box lists every backend the bundled Hamlib supports —
start typing any part of the name (`FT-847`, `IC-7300`) to find it — so
you never look a number up by hand. **Test CAT** and **Test PTT** try
the settings in front of you before you close the dialog.

Two programs cannot both hold the serial port, so sharing a radio means
putting something in front of it that both can talk to — a `rigctld`
from your own Hamlib install, or flrig. Point every program at that
(WSJT-X calls it *Hamlib NET rigctl*; fldigi has flrig built in), and
point this one at it too: model **2 (NET rigctl)** with the daemon's
`host:port` as the device, or model **4 (FLRig)**. They are ordinary
entries in the same picker, so no separate setting is involved. Model
**1** is the dummy rig, handy for exercising the buttons with no radio
attached.

</details>

<details>
<summary>Transmit-time latent optimization</summary>

Settings → Transmit has one switch, **Optimize latents before
transmitting**. The encoder is amortized — trained to do well on
everything — so for any *one* picture there are better inputs to the
same decoder, and a short gradient descent finds them. Worth
**1.4–1.8 dB** of recovered picture, and it is entirely sender-side:
optimized latents are ordinary latents, so the receiving station needs
no change and does not need to know.

It runs speculatively, starting a moment after you stop editing, so the
cost lands in time you were spending anyway; hitting Send early just
takes the best result so far. See
[docs/latent-optimization.md](docs/latent-optimization.md).

</details>

## For more information

There are also command-line tools — encode, decode, listen live, and a
channel simulator that tests the whole path with no radio — plus the
training pipeline. Those live in
[the wiki](https://github.com/arodland/SSTVAE/wiki):

- **[Command-line tools](https://github.com/arodland/SSTVAE/wiki/Command-line-tools)** — install, encode, decode, listen live, use a different model.
- **[Channel simulator](https://github.com/arodland/SSTVAE/wiki/Channel-simulator)** — AWGN, Watterson fading, frequency and clock offsets.
- **[Performance](https://github.com/arodland/SSTVAE/wiki/Performance)** — PSNR against SNR for every mode, and where it breaks.
- **[How it works](https://github.com/arodland/SSTVAE/wiki/How-it-works)** — beacon side-channel, nested modes, waveform table.
- **[Comparison with other modes](https://github.com/arodland/SSTVAE/wiki/Comparison-with-other-modes)** — against analog SSTV, HamDRM and SSDV.
- **[Training](https://github.com/arodland/SSTVAE/wiki/Training)** — the two-stage recipe.
- **[Development](https://github.com/arodland/SSTVAE/wiki/Development)** — tests, code layout, the native C++ port, building from source.

## Credits

The radio-autoencoder concept, the two-stage training recipe and the
PAPR-penalty approach all come from **FreeDV RADE** by David Rowe and
the FreeDV team. SSTVAE applies those ideas to still images; any
mistakes in the translation are mine.

The autoencoder is trained on **[COCO](https://cocodataset.org)**
(Common Objects in Context), Lin *et al.*, 2014 — resized to 640×480 as
`arodland/coco640-sstvae` on the Hugging Face Hub.

The photographs used on this page are my own.

## License

Code: [Artistic License 2.0](LICENSE).

**The application icon is not.** It is licensed artwork, usable in this
application but not sublicensed to anyone who receives this source — so a
fork or a redistributed package must replace it. [NOTICE](NOTICE) lists
the files and says what to do; each of them also carries an SPDX sidecar.
This is the same arrangement Firefox and Chrome have with their
open-source builds.
