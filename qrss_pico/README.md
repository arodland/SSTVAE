# QRSS beacon on a Pico W / Pico 2 W

A QRSSTVAE waveform CE transmitter: a Raspberry Pi Pico W or Pico 2 W
driving an Si5351A over I2C. It sends stored beacon files on a schedule,
keeps UTC from SNTP, and is run from a web page. The phase and the
Si5351 steps come from `../qrss_beacon_c/` (the parity-tested C
reference), built in as a library, not copied.

Nothing here has run on hardware yet. Everything below is built and
host-tested only.

## Build and flash

```sh
pio run                       # both boards
pio run -e picow -t upload    # or pico2w; hold BOOTSEL for the first flash
pio test -e native            # host tests: schedule, Si5351 plan, reached arithmetic
pio device monitor            # the USB console
```

The platform is maxgerhardt's fork with the earlephilhower Arduino core,
pinned in `platformio.ini`. PlatformIO's own `raspberrypi` platform has
no WiFi core and no Pico 2 W.

## First start

With no WiFi configured, or none in reach after 30 s, it opens an
access point `qrss-xxxxxx` serving the page at the address it prints.
The login is `admin` / `qrsstvae` until you change it. You can also set
WiFi over USB:

```
wifi MyNetwork my password here
reboot
```

Other console commands: `show`, `set {"base_hz":10140150}` (any
settings key, same checks as the page), `scan`, `i2c`, `sync`, `carrier <s>`, `abort`,
`bench`, `log`, `reboot`.

## Pictures

Make one beacon file per segment on the desktop, then upload them all:

```sh
qrss_encode.py photo.jpg photo.qrsp --mode B
qrss_beacon.py photo.qrsp seg0.bin --callsign K1ABC --grid FN42 --segment 0
qrss_beacon.py photo.qrsp seg1.bin --callsign K1ABC --grid FN42 --segment 1
```

The callsign, grid and keying (FSK, or OOK with `--ook`) are the file's.
The beacon has no callsign setting. A schedule entry needs every
segment its mode sends. Each file is 50,970 bytes. The file system
holds about 19 on a Pico W and 58 on a Pico 2 W.

## Schedule

An entry sends one picture `count` times (0 = until removed). It starts
on a quarter hour and repeats either back to back or at a fixed
interval. A send is one pass per segment, half an hour apart. This
follows the desktop scheduler (`native/gui/qrss_schedule.hpp`) with two
parts dropped: the encode lead and chaining, and rig retuning. Which
pass comes next is computed from the clock alone. A pass missed while
powered off is just in the past.

A pass is refused, and logged, when:

- there is no carrier frequency set;
- the clock has never synced;
- the last sync is older than `sync_max_age_s`.

TINY is a 53 s test frame with no callsign windows and no header. It is
for a dummy load or the bench, not the air.

## Hardware settings

- **I2C.** GP numbers, not physical pin numbers (GP2/GP3 are pins 4/5).
  Any SDA/SCL pair of one I2C block, e.g. GP4/5 (I2C0), GP2/3 or
  GP26/27 (I2C1). GP23-25 belong to the radio module.
- **Carrier.** The RF carrier in Hz, 8 kHz to 150 MHz. Spread
  randomizes each pass by up to +- that many Hz.
- **T/R GPIO.** Asserted `tr_lead_ms` before keying (lead-in included)
  and released `tr_tail_ms` after.
- **Crystal.** Calibrate it with the test carrier. Key it, measure,
  then set `xtal_ppm` = (measured / set - 1) x 1e6 with the sign that
  corrects it. The spec asks for about 1 ppm. The crystal load is set
  to 10 pF, as Adafruit-style 25 MHz boards expect. The ppm correction
  absorbs any mismatch.

The modulation steps the PLL numerator. One step is the frequency
resolution, and it grows with frequency (25 MHz crystal):

| Carrier | 1.8 MHz | 7 MHz | 14 MHz | 28 MHz | 50 MHz | 144 MHz |
|---|---|---|---|---|---|---|
| Step | 0.05 Hz | 0.19 Hz | 0.39 Hz | 0.80 Hz | 1.5 Hz | 4.0 Hz |

The design's M6 measurement (43 dB at 990 Hz updates) used 0.4 Hz. The
coarser steps of 6 m and 2 m have not been measured.

## What core 1 does

Core 1 runs only the transmitter (`src/tx.cpp`). It busy-waits on
`time_us_64()` deadlines, `update_hz` times a second, mapped from UTC
through the clock as it stood at the start of the pass. Each update
writes only the PLL registers that changed.

A flash write on core 0 (an upload or a save) pauses core 1. The page
warns before every write during a pass and does not refuse it. An
update that is missed entirely is skipped, and the error feedback is
told the chip held the old frequency, so the phase recovers. The status
line counts late updates.

`bench` on the console times one update on the chip. Compare it with
the update period before trusting 990 Hz.

## Not done yet

- Not run on hardware. The order to check things is `bench`, the test
  carrier, then a TINY pass into a dummy load, captured and decoded with
  `qrss_receive`.
- One update's register bytes land one at a time, so the PLL can briefly
  see a mixed value. Measure this on an SDR before fixing it.
- No config UF2. Bootstrap is the console or the access point.
- HTTP basic auth over plain HTTP, so the LAN can read the password.
