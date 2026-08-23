# Patches carried against usb-serial-for-android

The vendored `java/` tree is upstream tag **3.11.0**, and was pristine
until 2026-08-23. Every deviation from upstream is listed here and is
marked in the source with a `// SSTVAE PATCH` comment naming this file,
so a future re-vendor cannot silently drop one.

Keep this list short. A patch here is a maintenance cost paid on every
upgrade; prefer fixing things in `core/rig/android/java/`, which is
ours, whenever the behaviour can be reached from outside the library.

---

## 1. `Cp21xxSerialDriver.openInt()` — do not write SET_FLOW for "none"

**Upstream:**

```java
setConfigSingle(SILABSER_IFC_ENABLE_REQUEST_CODE, UART_ENABLE);
setConfigSingle(SILABSER_SET_MHS_REQUEST_CODE, ...);
setFlowControl(mFlowControl);
```

**Here:** the last call is guarded by `if (mFlowControl !=
FlowControl.NONE)`.

**This did not fix the bug it was written for, and is kept anyway.** An
IC-9700 still does not answer, so the CP2102N's flow configuration was
not what was stopping it. It stays because both reference
implementations avoid this write and one of them is the kernel, and
because Silicon Labs' own erratum says the structure is misread on some
parts — a blind write of zeroes over four registers is wrong on its own
terms whether or not it was Andrew's fault. Treat it as a candidate for
removal at the next re-vendor if upstream has adopted a
read-modify-write.

**Why.** An Icom IC-9700 would not answer a single CI-V frame from this
app on a phone, while `rigctl -m 3081 -r /dev/ttyUSB0 -s 19200` on the
same cable answered instantly. The app's own trace showed a correct
frame reaching the transport (`-> rig 6: fe fe a2 e0 03 fd`) against a
CP2102N at 19200 8N1, one vendor-class interface, two endpoints,
`Cp21xxSerialDriver` port 0 of 1 — every control transfer returning 0,
every bulk write returning its length, and nothing ever coming back.

Setting a CP210x's flow control to `NONE` is not a no-op: it sends the
chip a 16-byte `SET_FLOW` structure of zeroes, clobbering
`ulControlHandshake`, `ulFlowReplace`, `ulXonLimit` and `ulXoffLimit` in
one blind write. Two independent implementations that work with this
chip do not do that:

* **FT8TW** drives the same radio on the same phone. Its copy of this
  file has no `SILABSER_SET_FLOW_REQUEST_CODE` constant and no
  `setFlowControl` method at all — `openInt` is `IFC_ENABLE` plus
  `SET_MHS` and stops. (It is an *older snapshot*, from before upstream
  added flow control, rather than a deliberate removal: its
  `openInt(UsbDeviceConnection)` signature and two-argument `read` both
  predate 3.11. Its `setParameters` is behaviourally identical to
  this one, so with this guard in place the two now program the chip
  the same way — and the Icom still fails, which is what rules the
  chip's *configuration* out entirely.)
* **The Linux `cp210x` kernel driver**, which is what the working
  `rigctl` run goes through, never writes the structure blind: it does
  `GET_FLOW`, modifies the handshake bits, and writes it back,
  preserving `ulXonLimit`/`ulXoffLimit` — it only ever sets them itself
  (to 128) when IXOFF is actually requested.

The kernel also carries erratum **CP2102N_E104**: firmware `<= 0x10004`
interprets `ulXonLimit` as `ulFlowReplace`, i.e. the chip reads the
structure one word out of alignment, which makes a blind 16-byte write
land partly on fields the sender never meant to set — and makes the
chip's own `ulXoffLimit` come from past the end of the buffer. Linux's
response is to declare flow control unsupported on those parts.

**Guarded rather than deleted** so RTS/CTS and XON/XOFF still work
where an operator asks for them: the write is only skipped in the case
where it has nothing to say. A chip opened fresh is already in its
configured default, which is no flow control, so there is nothing to
undo. `SerialBridge.applyFlowControl` carries the matching early
return; both are needed, because `openInt` does this write before
anything of ours is asked.

**On upgrade:** check whether upstream has adopted a read-modify-write
`SET_FLOW` (or a CP2102N firmware-version quirk like the kernel's). If
it has, drop this patch and re-test against an Icom.
