# Reception aggregation: many stations, one picture

Mirrors how high-altitude balloons are received. A balloon sending SSDV
is heard by a scattered network of listeners, each of which has it in
range for part of the flight and none of which hears the whole thing
well. The pictures that reach the internet are better than any single
receiver decoded, because the *fragments* were pooled.

SSTVAE can do something stronger than pooling fragments, because its
payload is analog. Latents carry a per-latent confidence weight, so two
stations that each heard a transmission badly can be combined into one
that reads well -- not by picking the better copy per frame, but by
weighting every latent by how much each station's estimate deserves to
be trusted. That is `sstvae/modem/diversity.py`, and its measured gain
at two branches is +2.9 to +5.9 dB of latent SNR
(`docs/diversity-reception.md`).

This document is the network protocol built on it: stations upload what
they demodulated, a server combines every copy of each transmission.

**The picture is made server-side, from latents.** A station uploading
a PNG would be uploading the one thing that cannot be combined --
decoding consumes `latents x weights` and discards the weights, which
are exactly what the combining needs.

## What identifies a transmission

The beacon carries the transmitting station's callsign and an absolute
frame counter, but no clock and no transmission ID, so the server has
to decide for itself which uploads describe the same over. It uses:

    same transmitter callsign, and start times within 5 s

**Frequency is recorded and displayed, never matched on.** This is the
central decision, and it goes against the instinct that more matching
criteria are safer. The two failure directions are not symmetric:

- Splitting one transmission into two buckets forfeits the entire
  diversity gain, silently. Everything still looks like it worked --
  both stations get a 200, both appear in the gallery, both have
  pictures. Only the picture that should have existed is missing.
- Merging two transmissions cannot happen by accident, because one
  transmitter cannot start two overs within the window. The shortest
  mode lasts 32 s.

So the rule leans hard toward merging, and the criteria it uses are the
ones a station can actually be sure of. Dial frequency is the *least*
reliable field in the payload: a headless skimmer may have no rig
control at all, and the number is typed in by the operator. A wrong
digit would split a bucket. `--freq-split-khz` exists for a server
operating across bands where that trade is different; it is off.

The 5 s window is five times the ~1 s stations are asked to hold and
six times under the 32 s floor, which is what "lean toward merging"
means in practice.

## Clocks

Since time is what identifies a transmission, a station with a wrong
clock does not fail -- it uploads successfully, is filed as a
transmission of its own, and contributes nothing to the picture it
should have improved. Nothing about that is visible from either end,
which is why it is defended in three places.

**The station dates its own reception properly.**
`RingBuffer.utc_at()` measures back from the most recent write rather
than forward from an epoch captured at startup. Forward from a fixed
epoch assumes the sound card runs at exactly `FS`; at a routine 100 ppm
that is ~1.1 s of error after three hours and ~3.6 s after ten, and a
skimmer runs for days. Measuring backward bounds the error by the
lookback distance, which the ring bounds at its own depth: ~13 ms.

**The server refuses what cannot be true.** A payload claiming a
transmission that has not finished yet is rejected. The station decoded
it before uploading, so the whole transmission is already in its past,
and every delay between there and here pushes the other way. The
opposite direction is not decidable -- a start time that looks old is
exactly what a queued or retried upload looks like -- so it is recorded
and never refused.

**The station is told.** Every upload response carries `server_time`,
and `UploadSink` warns past a second of skew. It costs nothing, riding
a request already being made, and it puts the message in front of the
one person who can fix it.

`sstvae_server.py list-stations` shows the last skew seen per station.
It includes however long the upload took to arrive, so it is a bound
rather than a measurement: large numbers mean something, small ones do
not.

## The payload

One `.npz` (`sstvae/rx/receptionfile.py`), used unchanged as both the
local archive and the upload body:

| key | contents |
| --- | --- |
| `latents` | float32, canonical order, mode-sized (header) or mode-C-sized (blind) |
| `weights` | float32, per-latent confidence 0..1; **0 means erased** |
| `meta` | a JSON string |

`meta` carries `format`/`format_version`, `protocol_version`,
`codec_revision`, `software`, `path` (`header` or `blind`), the
per-path fields (`mode_name` + `frames_received`, or `frame_offset` +
`n_frames`), `snr_db`, `freq_offset`, the transmitting `callsign`, the
`station_callsign`, `dial_freq_hz` and `utc_start`.

Deliberate choices:

- **float32, not float16.** The payload is ~1.3 MB before compression,
  once per 32-95 s transmission, so nothing is bought by narrowing it,
  and float32 makes the round-trip bit-exact and therefore testable.
  The npz is self-describing, so a future narrower writer is a
  `format_version` bump rather than a format break.
- **Reader-side fields are not carried.** `sync_metric`,
  `preamble_start`, `frame0_start` and `beacon` are positions in the
  producing station's own audio buffer. They mean nothing anywhere
  else, and the combining reads none of them.
- **`protocol_version` mismatches are refused.** A different waveform's
  latents do not describe the same thing, and combining them would
  produce a confident wrong picture rather than an error.
- **numpy and stdlib only.** The server imports this module and has no
  audio, no codec and no torch -- and the same restraint keeps the
  format trivially writable from the C++ side when a native uploader is
  wanted.

## Authentication

Per-station API keys, issued by the server operator, sent as
`Authorization: Bearer <key>` over HTTPS. Keys are stored as a sha256
hash, so the database is not a list of credentials, and the plaintext
is shown once at issuance.

**Identity comes from the key, not the payload.** A payload naming a
different station than its key is refused. Otherwise any key would let
its holder file receptions under any callsign, and a station's record
would mean nothing.

Run uvicorn behind a TLS-terminating reverse proxy: a bearer token on
plain HTTP is a token anyone on the path can replay.

## Combining

On each upload the server reloads every reception for that transmission
and re-runs the combine, so a picture improves as stations arrive and a
late upload still helps. This is synchronous in the request handler:
the combine is milliseconds of numpy, the decode ~50 ms, and there is
at most one upload per station per transmission.

Two cases need care.

**A station that disagrees about the mode.** A mode mismatch is a hard
error in the combiner, and rightly -- two modes are two transmissions.
But the server cannot ask anyone, so the plurality mode wins and the
others sit out of that picture, with the reason returned to the station
that sent it. One bad report must not cost every other station theirs.

This is a vote taken when the picture is made, not an admission test,
and the distinction matters: **nothing is ever rejected on arrival.**
Every reception is stored whatever it claims, and since the whole
combine re-runs on each upload, the count is over everything received
so far. A station outvoted at one moment is counted again the instant
later arrivals agree with it. That is what lets the server decide
immediately, with no quorum to declare and nothing to undo -- the
alternative, waiting for some notion of "all the uploads", has no
stopping condition, because a station can upload a queued reception
hours later.

The corollary is that every verdict is provisional, so all of them are
rewritten on each combine rather than just the winners'. A station
counted a moment ago would otherwise keep the share it was last told
and go on claiming to have supplied a picture it is no longer in --
with the shares summing past 1. `excluded_reason` on each reception
says why it is not in the current picture, which a null share alone
cannot distinguish from "no combine has run yet".

**Blind-only transmissions.** If no station heard a header, the mode is
unknown and the picture is a full mode-C decode -- exactly what the
live blind path produces. Blind branches are already aligned to
absolute frame positions by the beacon inside `demodulate_blind`, so
the combining layer does no alignment work at all.

A station that uploads twice for one transmission **replaces** its
previous reception. Combining two copies of one branch breaks the
independence the maximal-ratio weighting assumes and reports
confidence the signal does not have.

## Running it

```sh
pip install -e '.[server]'

sstvae_server.py issue-key --callsign N0CALL     # shown once
sstvae_server.py run --data-dir ./aggregator-data
```

A station:

```sh
sstvae_listen.py --no-gui \
    --upload-url https://aggregator.example \
    --station-call N0CALL --upload-key-file ~/.sstvae-key \
    --frequency 14233000
```

The key comes from a file or `SSTVAE_UPLOAD_KEY`, never a flag: a key
on the command line ends up in shell history. `--frequency` is
optional, per the matching rule above.

Uploading never risks a reception. The local save runs first, the
payload is spooled to disk, and it is deleted only once the server has
acknowledged it -- so a reception heard while the link is down goes out
with the next one, or at the start of the next session.

## Endpoints

| method | path | |
| --- | --- | --- |
| POST | `/api/v1/receptions` | bearer auth, raw npz body |
| GET | `/api/v1/transmissions?limit=` | newest first, with per-station detail |
| GET | `/api/v1/transmissions/{id}` | one |
| GET | `/pictures/{id}.png` | the combined picture |
| GET | `/healthz` | |
| GET | `/` | gallery |

Only POST is authenticated.

The gallery polls the transmissions API every 5 s -- trivial,
proxy-proof, and bounded by transmission durations anyway. Picture URLs
carry a millisecond-resolution cache-buster, because two stations
uploading in the same second would otherwise produce the same URL and a
browser would go on showing the picture the second station just
improved.

## Verified

`scripts/aggregate_demo.py` runs a station's whole path -- the real
`decode_loop`, the real `UploadSink`, the real payload, real HTTP -- so
the demo exercises what a skimmer does rather than calling the
combining function directly.

Measured with two stations, mode A, both at 5 dB AWGN with independent
noise seeds:

| | radio SNR | latent SNR |
| --- | --- | --- |
| STA1 alone | 6.75 dB | +5.70 dB |
| STA2 alone | 6.90 dB | +5.68 dB |
| combined | 9.84 dB | **+7.78 dB** |

+3.1 dB of radio SNR, which is the +3 dB maximal-ratio combining
predicts for two equal branches, and **+2.08 dB in the latent domain**
over the better station. The two stations reported dial frequencies
33 kHz apart and were still combined, which is the frequency rule
above doing its job.

Pass the same `--utc-start` to every station in a demo. A real
receiver's ring buffer is stamped by its audio callback as the audio
arrives, so two stations hearing one transmission agree without
arranging anything; the demo fabricates its capture and would otherwise
stamp each run with its own "now".

## Not done

- **No native or GUI uploader.** The desktop app cannot upload, and the
  format was kept simple partly so it can later. The C++ HTTP code is
  GET-only today.
- **No incremental upload.** A station uploads once, when a reception
  finishes, so the server's picture cannot improve *during* an over.
- **No station reputation or moderation** beyond revoking a key.
- **No retention policy.** Receptions and pictures accumulate.
- **The combining is unchanged from the two-branch experiment.** It
  generalizes to N, but `contribution_image` (the red/blue diversitygram)
  is strictly two-branch and is not exposed here.
