# QRSSTVAE live stream

Runs a second, **receive-only** QRSSTVAE on a virtual X screen, fed by a
WebSDR (the Utah WebSDR on 7066 kHz USB by default), and streams that
screen plus the WebSDR audio to YouTube Live for a fixed time.

It is built to run beside a station's own QRSSTVAE without touching it.
It uses its own build directory, its own settings and data under
`~/.qrsstvae-stream`, private PulseAudio/PipeWire sinks, rig control off
and an empty schedule. The app's audio output goes to a sink nobody
hears, so it can never key a radio.

## How it fits together

```
WebSDR page --(headless system Firefox, websdr_bridge.js)--> s16le 8 kHz
   --pacat--> null sink "websdr_stream" --monitor--> remapped source "WebSDR_capture"
                                                       |-> sstvae-gui (Xvfb :99) -> QRSS listener
                                                       '-> ffmpeg (audio)
Xvfb :99 --x11grab--> ffmpeg --RTMPS--> YouTube
```

- `websdr_bridge.js` opens the WebSDR page in the system's headless
  Firefox, which Playwright drives over WebDriver BiDi. It taps the decoded audio through the page's
  `hook_webaud` hook and writes s16le mono at 8000 Hz to stdout.
- `stream.sh` starts everything: the sinks, Xvfb, the bridge, the app
  and ffmpeg. It stops after `$MINUTES`, then unloads its modules and
  kills its own process group.
- `setup.sh` is the one-time setup.

## Prerequisites

| Need | Arch | Debian/Ubuntu |
|---|---|---|
| Firefox | `firefox` | `firefox` (set `FIREFOX=` if it isn't `/usr/bin/firefox`) |
| Xvfb | `xorg-server-xvfb` | `xvfb` |
| xdotool | `xdotool` | `xdotool` |
| ffmpeg with `x11grab` and `pulse` | `ffmpeg` | `ffmpeg` |
| pactl, pacat | `libpulse` | `pulseaudio-utils` |
| Node.js ≥ 18, npm | `nodejs npm` | `nodejs npm` |
| Build | `cmake ninja qt6-base qt6-multimedia` | `cmake ninja-build qt6-base-dev qt6-multimedia-dev` |
| uv | `uv` | see https://docs.astral.sh/uv/ |

You also need a running PulseAudio or PipeWire (with pipewire-pulse)
session and network access to the WebSDR and to YouTube. Only the
Playwright npm package is installed, into `~/.qrsstvae-stream/node`.
No Playwright browser is downloaded.

## Setup (once)

```sh
YOUTUBE_KEY=xxxx-xxxx-xxxx-xxxx-xxxx scripts/stream/setup.sh
```

The script checks the tools, builds `native/build-stream`, runs
`uv sync --extra cli` and installs the Playwright package into
`~/.qrsstvae-stream/node`. It then writes:

- `~/.qrsstvae-stream/config/sstvae/config.json`: audio in from
  `WebSDR_capture`, out to `QRSS_stream_silent`; rig off; receive
  auto-start; QRSSTVAE view; received, transmit and template folders
  under `~/.qrsstvae-stream`.
- `~/.qrsstvae-stream/youtube_key` (mode 600). The stream key comes from
  YouTube Studio → Go live → Stream. Never commit it.

Re-running setup is safe: an existing config or key is left alone.

## Run

Dry run first, recording to a local file instead of YouTube:

```sh
MINUTES=2 OUT=/tmp/dry.flv setsid -w scripts/stream/stream.sh
```

While it runs, a screenshot of the virtual screen shows what viewers
will see:

```sh
ffmpeg -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 shot.png
```

The app should come up by itself in the QRSSTVAE view showing
"Receiving", "Nothing scheduled" and "Rig control off", with band noise
on the spectrogram.

Live, detached so it outlives the terminal:

```sh
mkdir -p ~/.qrsstvae-stream/logs
setsid nohup scripts/stream/stream.sh > ~/.qrsstvae-stream/logs/run.log 2>&1 &
```

Check `~/.qrsstvae-stream/logs/ffmpeg.log` after a minute, then confirm
in YouTube Studio that the stream is being received.

To stop early, kill the process group. Its id is `stream.sh`'s pid:

```sh
ps -eo pid,pgid,args | grep stream/stream.sh
kill -- -<pgid>
```

### Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `MINUTES` | 180 | length of the stream |
| `URL` | Utah WebSDR, `?tune=7066usb` | any PA3FWM WebSDR page; `tune=` sets frequency and mode |
| `DISP` | `:99` | Xvfb display |
| `OUT` | YouTube RTMPS URL built from the key | e.g. a local `.flv` for a dry run |
| `STREAM` | `~/.qrsstvae-stream` | settings, logs, key, Playwright |
| `FIREFOX` | `/usr/bin/firefox` | browser the bridge drives |
| `SRC` | this checkout | where `native/build-stream` and `qrss_listen.py` are |
| `STALL_S` | 15 | seconds without WebSDR audio before the bridge reloads the page |
| `CALLSIGN` | `AG7EW` | name the WebSDR shows for this listener |

### Logs

`~/.qrsstvae-stream/logs/` holds:

- `run.log`
- `ffmpeg.log`
- `bridge.log`: the WebSDR's sample rate, plus a rate line every minute;
  expect about 8000 samples/s. A "no audio for N s ... reconnecting"
  line followed by "reconnected" means the bridge reloaded the page
  after the audio stopped (see the 2-hour cap below).
- `app.log`
- `xvfb.log`

Two ffmpeg lines on every start are harmless: "not enough frames to
estimate rate" and "Guessed Channel Layout".

## Things that bit, so they are not re-learned

- **Qt does not list monitor sources.** With `input_device` set to
  "Monitor of WebSDR_stream", the app found no match and silently
  recorded the **system default input** instead. That is why
  `stream.sh` adds a `module-remap-source` called `WebSDR_capture`.
  After any change, check the app's status log for
  `[audio in] WebSDR_capture at ...`.
- **The WebSDR page overwrites `hook_webaud`.** Its
  `websdr-soundhooks.js` assigns its own hook and sets
  `webaud_hookdef=0`, which still counts as defined. The bridge pins its
  hook with `Object.defineProperty` and a setter that ignores
  reassignment. Without that, the bridge runs happily and outputs zero
  samples.
- **The WebSDR page stops its own audio after an idle timeout.** On
  2026-10-11 the audio stopped at exactly 57,600,768 samples (7200 s at
  8 kHz) with the websocket still open, so the app just heard silence.
  The page's `timeout_idle_do()` fires after `idletimeout` from the
  site's `tmp/bandinfo.js` (7,200,000 ms on Utah) without tuning or
  other input, and destroys the audio client. The bridge now reloads the page when no
  samples arrive for `STALL_S` seconds, which opens a new session with a
  gap of a few seconds, and backs off (up to 5 min) if a reload brings
  no audio.
- **The page runs in its "blind" view** (cookie `view=3`), which opens
  no waterfall stream, so the bridge fetches audio only. The callsign is
  set as the `username` cookie, which the page sends with the tuning.
- **Don't edit `stream.sh` while a stream is running.** Bash reads a
  script as it executes it. Write a new file and rename it over the old
  one instead.
- No window manager runs on the virtual screen, so `stream.sh` uses
  xdotool to size the app to fill it. Nothing is ever sent to a real
  display.
