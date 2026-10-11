#!/usr/bin/env bash
# Second, receive-only QRSSTVAE on a virtual screen, fed by the Utah
# WebSDR, streamed to YouTube for at most $MINUTES minutes.
# Run scripts/stream/setup.sh once first; see scripts/stream/README.md.
# Start detached so it outlives the shell and leads its own process group:
#   setsid nohup scripts/stream/stream.sh > ~/.qrsstvae-stream/logs/run.log 2>&1 &
# Stop early with: kill -- -<pgid>   (the pgid is this script's pid)
set -u
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
STREAM=${STREAM:-$HOME/.qrsstvae-stream}
MINUTES=${MINUTES:-180}
URL=${URL:-http://websdr1.sdrutah.org:8901/index1a.html?tune=7066usb&lto}
CALLSIGN=${CALLSIGN:-AG7EW}
DISP=${DISP:-:99}
KEY=$(cat "$STREAM/youtube_key")
# OUT overrides the destination (e.g. a local .flv for a dry run).
OUT=${OUT:-rtmps://a.rtmps.youtube.com/live2/$KEY}
LOG=$STREAM/logs; mkdir -p "$LOG"

# Private null sinks: one the WebSDR audio plays into (the app records
# its monitor), one the app's own output goes to (heard by nobody).
IN_MOD=$(pactl load-module module-null-sink sink_name=websdr_stream sink_properties=device.description=WebSDR_stream)
OUT_MOD=$(pactl load-module module-null-sink sink_name=qrss_stream_silent sink_properties=device.description=QRSS_stream_silent)
# Qt does not enumerate monitor sources, so the app captures from a
# remapped copy of the WebSDR sink's monitor with a description it can see.
SRC_MOD=$(pactl load-module module-remap-source source_name=websdr_capture master=websdr_stream.monitor channels=1 source_properties=device.description=WebSDR_capture)
pids=()
# Run under setsid so this script leads its own process group; cleanup
# then stops everything it started (bridge, Firefox, app, listener).
cleanup() {
  trap - EXIT INT TERM
  pactl unload-module "$SRC_MOD" 2>/dev/null; pactl unload-module "$IN_MOD" 2>/dev/null; pactl unload-module "$OUT_MOD" 2>/dev/null
  kill -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM

Xvfb "$DISP" -screen 0 1920x1080x24 -nolisten tcp > "$LOG/xvfb.log" 2>&1 & pids+=($!)
sleep 2

(NODE_PATH="$STREAM/node/node_modules" WEBSDR_NAME="$CALLSIGN" node "$SRC/scripts/stream/websdr_bridge.js" "$URL" 2> "$LOG/bridge.log") \
  | pacat --playback -d websdr_stream --rate=8000 --format=s16le --channels=1 --raw --latency-msec=200 &
pids+=($!)

# Isolated settings and data: own config.json (no rig, empty schedule),
# own listener store. Nothing of the main install is read or written.
export XDG_CONFIG_HOME=$STREAM/config XDG_DATA_HOME=$STREAM/data QRSSTVAE_HOME=$STREAM/qrsstvae
export PULSE_SINK=qrss_stream_silent PULSE_SOURCE=websdr_capture
(cd "$SRC" && DISPLAY=$DISP ./native/build-stream/sstvae-gui > "$LOG/app.log" 2>&1) & pids+=($!)
sleep 15
# No window manager on :99: make the app fill the virtual screen.
DISPLAY=$DISP xdotool search --sync --onlyvisible --name . windowmove 0 0 windowsize 1920 1080 2>/dev/null

# Screen plus the WebSDR audio, low frame rate, to YouTube (RTMPS).
ffmpeg -nostdin -loglevel warning \
  -f x11grab -framerate 2 -video_size 1920x1080 -i "$DISP" \
  -f pulse -i websdr_stream.monitor \
  -c:v libx264 -preset veryfast -tune stillimage -pix_fmt yuv420p -r 2 -g 4 -b:v 1500k -maxrate 1500k -bufsize 3000k \
  -c:a aac -b:a 64k -ar 44100 -ac 2 \
  -t $((MINUTES*60)) -f flv "$OUT" > "$LOG/ffmpeg.log" 2>&1
echo "stream ended $(date -u)" >> "$LOG/ffmpeg.log"
