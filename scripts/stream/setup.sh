#!/usr/bin/env bash
# One-time setup for stream.sh: checks prerequisites, builds the app in
# its own build directory, prepares the Python env and Playwright (the
# npm package only; the bridge drives the system Firefox), and
# writes the stream's private config and YouTube key. Safe to re-run;
# an existing config.json or key is left alone.
# usage: YOUTUBE_KEY=xxxx-xxxx-xxxx-xxxx scripts/stream/setup.sh
set -eu
SRC=$(cd "$(dirname "$0")/../.." && pwd)
STREAM=${STREAM:-$HOME/.qrsstvae-stream}
PLAYWRIGHT_VERSION=1.56.1

missing=()
for t in Xvfb ffmpeg pactl pacat xdotool node npm cmake ninja uv "${FIREFOX:-firefox}"; do
  command -v "$t" >/dev/null || missing+=("$t")
done
if ((${#missing[@]})); then
  echo "missing: ${missing[*]}" >&2
  echo "  Arch:   sudo pacman -S --needed firefox xorg-server-xvfb xdotool ffmpeg libpulse nodejs npm cmake ninja uv qt6-base qt6-multimedia" >&2
  echo "  Debian: sudo apt install firefox xvfb xdotool ffmpeg pulseaudio-utils nodejs npm cmake ninja-build qt6-base-dev qt6-multimedia-dev  (uv: https://docs.astral.sh/uv/)" >&2
  exit 1
fi
node -e 'process.exit(+process.versions.node.split(".")[0] >= 18 ? 0 : 1)' \
  || { echo "node >= 18 required" >&2; exit 1; }
for dev in x11grab pulse; do
  ffmpeg -hide_banner -devices 2>/dev/null | grep -qw "$dev" \
    || { echo "ffmpeg lacks the $dev device" >&2; exit 1; }
done

# Own build directory, so a build tree belonging to another checkout is
# never reused.
cmake -S "$SRC/native" -B "$SRC/native/build-stream" -G Ninja -DSSTVAE_BUILD_PYMODULE=OFF
ninja -C "$SRC/native/build-stream"

# The QRSS listener runs from this checkout's .venv (the app finds it).
(cd "$SRC" && uv sync --extra cli && uv pip install soundfile)

mkdir -p "$STREAM/node" "$STREAM/logs" "$STREAM/config/sstvae" \
  "$STREAM/received" "$STREAM/templates" "$STREAM/data" "$STREAM/qrsstvae"
if [ ! -d "$STREAM/node/node_modules/playwright" ]; then
  (cd "$STREAM/node" && npm init -y >/dev/null && npm i "playwright@$PLAYWRIGHT_VERSION")
fi
# No Playwright browser download: the bridge drives the system Firefox.

# Private settings: rig control off, no schedule, audio on the private
# devices stream.sh creates, folders kept away from the main install.
CONF=$STREAM/config/sstvae/config.json
if [ ! -f "$CONF" ]; then
  cat > "$CONF" <<EOF
{"audio":{"backend":"qt","input_device":"WebSDR_capture","output_device":"QRSS_stream_silent"},
 "rig":{"enabled":false},
 "folders":{"receive_dir":"$STREAM/received","transmit_dir":"$STREAM/received","template_dir":"$STREAM/templates"},
 "receive":{"auto_start":true},
 "ui":{"mode":"qrsstvae"}}
EOF
  echo "wrote $CONF"
fi

if [ ! -s "$STREAM/youtube_key" ]; then
  if [ -n "${YOUTUBE_KEY:-}" ]; then
    (umask 077; printf %s "$YOUTUBE_KEY" > "$STREAM/youtube_key")
    echo "wrote $STREAM/youtube_key"
  else
    echo "no YouTube key yet: re-run with YOUTUBE_KEY=... or write it to $STREAM/youtube_key (chmod 600)" >&2
  fi
fi
echo "setup done; see scripts/stream/README.md to start a stream"
