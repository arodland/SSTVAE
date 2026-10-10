# How the QRSSTVAE receiver hears a picture

This is a plain-language tour of what happens between "a faint signal
reaches the antenna" and "a picture appears on the screen". It assumes
no radio or signal-processing background. The technical details are in
[README.md](README.md) and [design.md](design.md).

## First, what is QRSSTVAE?

**SSTVAE** sends pictures over shortwave radio. Instead of sending
pixels, it uses a neural network (an "autoencoder") that squeezes a
picture into a list of about 158,000 numbers, called *latents*. Think of
them as a very compact description of the picture. A matching network
at the other end reads those numbers and redraws the picture. SSTVAE
sends its numbers quickly, in 32 to 95 seconds, over a signal about as
wide as a voice channel. That works when the signal is reasonably strong.

**QRSSTVAE** uses the same two networks, but sends the numbers very
slowly over a very narrow signal. The idea comes from QRSS, where hams
send Morse code so slowly that a computer can pick it out of noise no
human could hear through. QRSSTVAE does the same for pictures:

| | SSTVAE | QRSSTVAE |
|---|---|---|
| Time to send one picture | 32 seconds | about 30 minutes (one *pass*) |
| Width of the signal | about 1,200 Hz | about 50 Hz |
| Signals that fit in one voice channel | 1 | up to about 48, side by side |
| How weak a signal still works | around the strength of the noise | about 12 times weaker than the noise for a good picture in one pass, and weaker still if passes are repeated |
| Repeats | each transmission stands alone | repeats of the same picture add up |

So the trade is time for sensitivity. Going about 50 times slower and
about 25 times narrower lets the picture through when the signal is buried deep in the
noise.

## Why "below the noise" is possible at all

Radio static is random. A signal is not. If you listen to the same
quiet signal for a long time and keep adding up what you hear, the
signal adds up steadily while the random static partly cancels itself
out. Each time you double how long you listen, the signal-to-noise
ratio improves by about a factor of two (3 dB). That's why QRSSTVAE is slow, and why hearing
the same picture again later makes it clearer.

Signal strength here is quoted as *SNR*, signal-to-noise ratio, in
decibels (dB). 0 dB means the signal and the noise in a voice-width
channel are equally strong, and negative numbers mean the signal is
weaker. Every 10 dB is a factor of ten. In simulation, one QRSSTVAE pass
gives a good picture at about -11 dB, a signal a little over a tenth as
strong as the noise. Sixteen passes of the same picture added together
give a good picture at about -26 dB, where the signal is a few hundred
times weaker than the noise.

## The clock: everything happens on the quarter hour

Every station's computer keeps its clock right (to about a second, using
the internet time service every computer already has). Every
transmission starts exactly 1 second after a quarter hour: :00, :15,
:30 or :45. Each of those start times is a **slot**.

Because a pass lasts about 30 minutes, a slot that starts at :00 is still
going when the :15 slot starts. Neighbouring slots overlap. That matters
later.

Knowing *when* a signal must start is a huge help to the receiver. It
doesn't have to guess the timing, only the exact frequency.

## What one pass contains

A pass is like a letter with an envelope:

```
 :00:01                                                        :29:44
   |-- preamble --|-- header --|------------ picture numbers ------------|
       20 s          80 s            about 28 minutes
                                  (with four short Morse callsign
                                   breaks, one every 7.5 minutes)
```

- **Preamble (20 seconds).** A fixed pattern every receiver knows in
  advance, like a knock on the door. The receiver uses it to notice that
  a signal is there and to measure its exact frequency.
- **Header (80 seconds).** The envelope. It says who is sending (the
  callsign and location), which picture this is (the *picture ID*),
  which part of the picture this pass carries, and which version of the
  picture-drawing network to use. It is heavily protected with an
  error-correcting code, so it can be read from a very weak signal.
  A second, much fainter copy of the header also rides on top of the
  picture numbers for the whole pass. The receiver reads the two copies
  together, which works on a weaker signal than the 80-second block
  alone, and the faint copy alone can still be read if the block is
  lost. Once the header is known, the receiver subtracts the faint copy
  exactly, so it costs the picture nothing.
- **Picture numbers (about 28 minutes).** About 50,600 of the
  picture's numbers. They are scrambled and spread out, so a burst of
  interference damages a little of everything rather than wiping out one
  corner of the picture.
- **Morse callsign breaks.** Four times per pass, the signal pauses its
  data for about 12 seconds to send the callsign in Morse code. The law requires a
  station to identify itself regularly, and this also lets a receiver
  check who is sending even when it missed the header.

A picture can be sent in mode A (one pass), mode B (two passes, half an
hour apart) or mode C (three). More passes carry more of the picture's
numbers, so the picture has more detail.

## The picture ID: a fingerprint

The picture ID is a fingerprint computed from the picture's numbers. The
same picture sent again, in the same mode, always gets the same
fingerprint. That's how a receiver knows that a pass it heard tonight
and a pass it hears tomorrow are the same picture and should be added
together. The callsign plus the picture ID name each picture uniquely.
The desktop app's schedule window sends a picture again automatically
(twice in a row, every other hour, and so on), and every one of those
sends has the same fingerprint.

## How the listener hears: step by step

The listener program takes the audio from your radio's speaker or USB
output and does the following.

### 1. It records everything

It keeps the last 35 minutes or so of audio in memory, and the last 48
hours on disk (the *passband store*). The disk copy turns out to be
important: it lets the receiver go back and look again. Each finished
hour is squeezed with FLAC, the lossless format music archives use, to
about half its size; nothing in it changes.

### 2. It listens for the knock

Shortly after each quarter hour, it searches the recording for the
preamble of every signal in the channel. Each signal it finds gets a
**tile** on the screen: a small card with its frequency, its strength
and, once the header is read, the sender's callsign.

It repeats that search every two minutes for the first five minutes, because
a weak signal can take a few minutes of audio to confirm. If nothing has
turned up by then, it searches only every five minutes for the rest of
the slot. On a quiet band that saves most of the listener's work, and a
signal it misses can still be found by the last look at the end (step 4).

### 3. It fills in the picture as the pass arrives

Every two minutes it re-reads everything heard so far for each signal,
draws the picture from the numbers it has, and updates the tile. Each
number comes with a *confidence weight*. A number heard clearly counts a
lot, and a number buried in noise or not yet received counts for little
or nothing. The drawing network is built to cope with missing numbers,
so the picture starts out blurry and sharpens as the pass goes on. It's
rarely recognizable before about a third of the pass.

### 4. It looks again at the end

When a slot's 30 minutes are over, the listener reads the whole slot one
more time with everything it now knows. This also searches for signals
that were too weak for anyone to notice on their preamble alone. That
search is expensive, so it is only done when the listener heard at least
half of the slot. It runs as a separate program on its own processor
core, so the next slot's tiles, already on the air, keep updating while
it works.

### 5. It files the pass

Each pass found goes into the **multi-pass store** on disk, which keeps
one running total per picture. Passes of the same picture are added in,
each number weighted by its confidence. A picture you heard three times
looks better than one you heard once.

## When things go wrong

Real radio is messy. Here is what the receiver does in the common cases.

### The header didn't come through

The header might be lost to a burst of static, or the signal might be
right at the edge of what can be read. The numbers still arrive; the
receiver just doesn't know for certain where in the picture they belong.

The faint copy that rides on the picture numbers makes this rarer. A
burst of static that wipes out the 80-second block leaves the copy
spread over the other 28 minutes, and that copy is enough on its own
when the signal is a little stronger than the weakest readable one.

When neither copy can be read, a sensible guess usually works: it assumes this is the first
(or only) pass of the picture, which is true for every mode A send. The
tile then shows a picture marked "no header yet", and redraws it
properly as soon as a header is read. The guess is only used for
display. It is never added to the stored picture.

If one refresh reads the header and a later refresh misses it, the
receiver keeps the header it already read. A header doesn't change
during a pass, and the error-correcting code makes a false reading
extremely unlikely (far less than one in a hundred million).

### You started listening late

Suppose you start the listener at 10 past the hour, and someone has
been transmitting since :00. You missed the knock and the envelope, so:

- **During the pass** there is no tile for that signal, because live
  tiles are only created from the preamble.
- **At the end of the slot**, you have heard about two thirds of it,
  which is more than half, so the end-of-slot search runs and finds the
  signal. If the signal is strong enough, the faint copy of the header
  riding on the numbers you heard names the picture, and the pass is
  added to it straight away. Otherwise the numbers you heard are stored
  as a *provisional* pass: kept on disk, but not yet attached to any
  picture, because there was no header to say which one.
- **When the same picture is sent again** later and you catch its
  header, the receiver compares the provisional pass's numbers against
  the newly named picture. If they match (they will, if it's the same
  picture and the same part of it), the provisional pass is added in.
  Your late start still counts.

The headers of several weak passes can also be combined, so passes
that couldn't read their header alone sometimes can together.

If you start after about :15, you heard less than half the slot and
the end-of-slot search doesn't run. The audio is still on disk for 48
hours, though, and once you have the picture from another pass, a
separate tool (`qrss_decode.py --retro`) can search those old recordings
for it.

### You stopped the listener, or it was restarted

When the listener starts, it reloads the recent audio from disk, so a
slot that was already under way picks up where it left off. Any slot
that *ended* while the listener was stopped (in the last 3 hours) is
read back from disk and finished, so stopping the listener just before
a quarter hour doesn't lose the pass it was working on.

### Your own station was transmitting

A radio can't listen while it transmits, so the app stops receiving
during your own passes. It won't hear its own picture. To test, use a
second receiver.

### You changed frequency during a slot

The app can retune the radio, by hand or on a schedule. After a change
the radio hears a different stretch of the band, so whatever now sits
at a signal's audio frequency is somebody else, or nobody. Reading it as
more of the same picture would mix in rubbish. So the app writes every
change of the radio's dial to a small file next to the tiles
(`dial_changes.jsonl`), and the listener reads it:

- The 2 seconds before a change and the 30 seconds after it count as
  not heard, while the radio switches and perhaps tunes its antenna.
- While the slot is on the air, its tiles are updated only from what
  was heard on the dial the slot started on. Everything after the
  change counts as not heard, as if the audio had stopped.
- At the end of the slot, each dial's stretch is received on its own.
  The signals found on the first dial are received as usual. A later
  stretch gets the whole-slot search only if it covers at least half
  the slot, the same rule as starting late.

Each tile remembers its dial, so a signal at 1500 Hz on 30 m and one at
1500 Hz on 20 m are two tiles, never one. The numbers heard before the
change are not wasted: they are stored like any partial pass and add to
the picture when it is sent again. Without rig control there is no
file, and the listener assumes the dial never moved.

### A voice, a whistle or a nearby carrier

On a busy band, a person talking or a steady tone can look a little like
a signal to the search. Two checks keep those out. A real pass that is
strong enough would have had its header read, so a strong signal with
no header and no Morse callsign is thrown away. And the search only
checks a limited number of candidates per slot, strongest first, so a
noisy band can't keep the computer busy for longer than a slot.

### Echoes of the neighbouring slot

Since passes last 30 minutes and slots are 15 minutes apart, the end of
the :00 slot's recording also contains the first half of anything sent
at :15. The search could mistake that signal for one in the :00 slot,
at the wrong timing. So a header-less pass at the same frequency as a
signal the neighbouring slot identified is treated as that signal's
echo and ignored. Only the two immediate neighbours count: the same
picture sent again half an hour or more later is not an echo, and is
exactly what the receiver wants to add up.

## Words used on the screen

- **Slot**: a quarter-hour start time, such as 05:45Z (Z means UTC).
- **SNR**: signal strength against the noise, in dB. Negative means
  weaker than the noise.
- **% of the pass**: how far through its 30 minutes the pass is.
- **heard**: how much of the pass the listener actually recorded.
- **% of latents**: how many of the picture's numbers have arrived with
  any confidence at all.
- **Mean weight**: how confident those numbers are on average. Higher
  (closer to 0 dB or above) is better.
- **The coloured band** under a picture: the same confidence, moment by
  moment through the pass, left to right. Black means that stretch was
  no better than noise; blue is weak, then red, yellow, and white for a
  strong, clear signal. Grey means there was nothing to measure there
  (the knock at the start, or audio the listener didn't hear). A black
  stretch in the middle is usually a fade or a burst of interference.
- **Mode A / B / C** and **passes**: how many passes the picture is sent
  in, and how many of them are in the picture shown.

## Trying it

Start the listener from the desktop app (View > QRSS signals > Start
listener), or from a terminal by piping audio into `qrss_listen.py`.
[README.md](README.md) has the commands. Everything it learns is saved in
your home folder, under `~/.local/share/qrsstvae`.
