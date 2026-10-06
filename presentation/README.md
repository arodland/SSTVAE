# SSTVAE for undergraduates — Manim deck

Twelve short clips, one per build step, meant to be dropped into slides.
No prose, a handful of labels. `style.py` is the visual vocabulary
(one color per *role*, fixed screen positions, shared primitives) and
`scenes.py` the scenes; keep anything new drawing from `style.py` so the
deck stays one picture.

| Scene | Shows |
|---|---|
| `PhasorSine` | rotating phasor, Re/Im beat, projection becomes the scrolling sine |
| `ModAM` `ModPM` `ModFM` `ModQAM` | m(t) × carrier on one fixed layout; FM is a slow phasor whose *rate* varies |
| `OFDM` | six 4-QAM symbols on six carriers at once, summed: one OFDM symbol, then the next |
| `AutoencoderWhole` `AutoencoderSplit` `AutoencoderChannel` | 12-8-4-8-12 network; braces; split with the latent duplicated; the channel box |
| `LatentPairs` `LatentOnCarrier` | latents two-by-two → complex points → the modulator slot from ModQAM |
| `Carriers` `ResourceGrid` | the same bank fed with latent pairs; the carrier × time grid |

Colors: gold = carrier, blue = information (m(t), constellation, latents),
grey = structure, red = the channel and nothing else.

## Rendering

Manim Community (not ManimGL). No LaTeX needed — labels are Pango
`MarkupText`.

```sh
uv venv -p 3.12 .venv && . .venv/bin/activate && uv pip install manim
cd presentation
manim -pql scenes.py PhasorSine     # preview one scene
manim -qh scenes.py                 # all scenes, 1080p, into media/videos/
```

`manimpango` needs pango headers to build where no wheel is available
(`libpango1.0-dev` on Debian/Ubuntu).
