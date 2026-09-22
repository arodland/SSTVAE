# Design: overlay templates, and the overlay on Android

> Written 2026-09-14, revised the same day after Andrew's review (the
> report is the measured SNR rather than a typed RSV, and templates may
> declare **custom fields**). **Approved and not implemented**: every
> decision below is settled as of 2026-09-14, so this is the spec to
> build from, in the order "Sequencing" gives.

## Why now

The beacon carrier identifies *this* station to every receiver, and the
CW ID identifies it to a human by ear. That is why the Android app
transmits the picture unmodified and does not even burn a callsign into
it (`docs/android.md`, Tier 1, 2026-08-09). Neither mechanism can say
**whom the transmission is addressed to**, and a reply in a QSO is
addressed: "W1XYZ de KC2G" is the thing that makes an over a
communication with a particular station rather than a broadcast. The
only place that text can go is the pixels, so a Part 97-shaped QSO from
a phone needs overlay text after all.

That does not reverse the earlier decision. No automatic caption, no
callsign requirement, no identification duty taken on by the app. What
is added is the ability to *address*, and it should cost the operator
about two taps mid-QSO, because on a phone anything more than that will
not be done.

## The one idea: separate the fields from the layout

The desktop composes an overlay per picture: add text, type it, drag it.
On a 6-inch screen, with a transmission to answer inside a minute, that
is the wrong shape. The shape that works is qsstv's: a **template** is a
layout with holes in it, the holes are filled from a **handful of
fields** the operator types once, and choosing a template is one tap.

- A template is an `OverlayDoc`, exactly the existing format, whose text
  items may contain **placeholders** such as `{theircall}`.
- The per-over UI is a **form derived from the template**: a CQ template
  has no holes and asks nothing; a reply template asks for their call.
- Free-form editing (dragging things about) is the *rare* path, kept on
  its own screen, and on a phone reduced to what a phone can do.

Both design choices the overlay model was built with are what make this
a UI-only change: normalized coordinates mean one template fits every
picture, and `ImageItem.source = "last_rx"` means "the picture I am
replying to" without the template naming a file.

## The format

**A template file is an overlay document with a name.** One optional
top-level key, `"name"`, beside `"version"` and `"items"`. The item
schema does not change and `DOC_VERSION` stays 1; both loaders skip keys
they do not know, so a template opens as a plain document in any build
that has the model, and a plain document is a template with no name.

**Placeholders are `{name}` inside `TextItem.text`.** The built-in set
is closed and small, and every one but `{theircall}` is filled in
without the operator typing anything:

| Placeholder   | Source                                              |
|---------------|-----------------------------------------------------|
| `{mycall}`    | station setting (exists: `station/callsign`)        |
| `{grid}`      | station setting (new, optional)                     |
| `{name}`      | station setting (new, optional)                     |
| `{theircall}` | the reply target's beacon callsign; editable        |
| `{snr}`       | the reply target's measured SNR, from its sidecar   |
| `{utc}`       | time at Send, HH:MM                                 |
| `{date}`      | date at Send, ISO                                   |
| `{mode}`      | the transmit mode                                   |

**`{snr}` is the report, and it is not typed.** An RSV is a judgement
the operator has to form and enter; the SNR is a number the receiver
already measured and wrote into the sidecar beside the callsign
(`snr_db`, which `PictureList` already exposes). It skips the *visual*
part of a report, deliberately: a reply that says "SNR 12 dB" is more
useful to the other station than a "5" typed by someone reading a
phone at arm's length, and it costs nothing to fill in. Rendered as an
integer with the unit, `12 dB`; empty when there is no reply target or
the sidecar has none.

**`{field Label}` declares a custom field.** Anything after `field`
up to the closing brace is the label — `{field Comment}`,
`{field Their QTH}`, `{field Visual quality}` — and it is presented to
the operator as a text box with that label. Two uses, and the second is
why it is not just a comment box: it is the way to drop a line of
free-form reply text into a transmission **without editing a
template**, and it is how a more structured template asks for things
the app cannot derive, such as a visual quality figure for whoever
still wants one. The same label twice is one field, substituted in
both places. Custom fields are **never mandatory**: leave one empty
and the dropped-line rule below removes its line, so a template
carrying three optional fields costs nothing when none is filled.

Three rules, all in one pure function `substitute(doc, fields) -> doc`:

1. **An unknown placeholder is left literally in the text.** A typo
   must be visible in the preview, not silently deleted from the air.
   `{{` and `}}` are literal braces, for the one person who wants one.
2. **A line whose placeholders are all empty is dropped, literal text
   and all.** This is the single piece of cleverness and it is what
   makes optional fields optional: `{theircall} de {mycall}` newline
   `SNR {snr}` newline `{field Comment}` loses its second line on a
   reply with no measured SNR and its third when no comment is typed.
   The literal `SNR ` goes with its placeholder, since a label with
   nothing after it is worse than no line. *Only* whole lines, never
   partial text, so the rule cannot produce a half-sentence; a line
   with no placeholders at all is never touched.
3. **The stored template is never mutated.** Substitution runs on a
   copy at preview and again at Send; the file keeps its holes.

Alongside it, `placeholders(doc)` lists what a template uses: the
built-ins it names, and the custom fields with their labels in order
of first appearance. That is what derives the form: no schema, no
per-template metadata, nothing to keep in sync with the text.

These live in `native/core/overlay/template.{hpp,cpp}` — string
processing on the model, **Qt-free, in `sstvae_core`**, so it builds
and is tested with `--no-overlay` and on every CI job — and in
`sstvae/overlay/template.py` as the reference, with the pair diffed in
`tests/test_native_overlay.py` the way the model already is. The
renderer is untouched: it receives an ordinary document.

**Built-in templates ship with both apps** (Android `assets/`, desktop
installed data), listed first and read-only, with *Duplicate* as the way
to get an editable copy. Four is the whole set:

- **None** — the picture goes out unmodified. Today's behaviour, and the
  default on Android, so nothing changes for anyone who never touches
  this.
- **CQ** — `CQ SSTVAE` / `de {mycall}` / `{grid}`, top-left, large,
  with `{field Comment}` beneath.
- **Reply** — `{theircall} de {mycall}` / `SNR {snr}` /
  `{field Comment}`, top-left.
- **Reply with picture** — Reply plus a `last_rx` inset, bottom-right.

Every built-in carries a Comment line, so the free-form path exists
from the first run without anyone opening an editor, and costs no
pixels until something is typed.

User templates are JSON files in `config_dir()/templates/` on the
desktop and the app's files directory on Android. The same bytes work
on both, which is deliberate: the realistic way an elaborate template
reaches a phone is by making it on the desktop and sharing the file.

## The reply binding: which picture, whose call

`last_rx` currently means "the newest complete reception". On the
desktop that is right, since the operator is looking at it. On a phone
the reception being answered may be three back in the Pictures list,
and — the important part — **its callsign is already decoded**: the
beacon put it in the sidecar `.json` beside every saved picture, and
`PictureList` exposes it as a role. So a reply should never involve
typing a callsign the app already knows.

`Composition` gains a **reply target**: a reception's path, its
callsign and its SNR. While one is set, `last_rx` resolves to that
picture, `{theircall}` is prefilled from it (editable — the beacon can
be absent or garbled, in which case the field is empty and the operator
types it) and `{snr}` is its measured figure. With none set, `last_rx`
is the newest reception, as today, and `{snr}` is empty.
The target is runtime state and is not stored in the template, which is
what keeps a template a template.

## The flows

### Android, mid-QSO (the case that matters)

1. A picture from W1XYZ completes. Its card in Pictures, and the viewer,
   carry a **Reply** button beside Share. The Listen tab shows the same
   button while the reception is recent, for the case where the
   operator is watching it come in (settled, Andrew 2026-09-14). There
   is room: while a picture is decoding the tab already shrinks the
   waterfall and shows the live image beneath it, so the button goes on
   the status row under that image, "Reply to W1XYZ", and is absent
   otherwise — no new row, nothing taken from the waterfall's height.
   One thing to get right: `rx/engine` wipes the live image and its
   metadata from shared state about two seconds after a reception, so
   `Listener` has to **keep the last delivered reception itself** —
   path, callsign, SNR, from the same sink call that writes the
   sidecar — for the button to have something to bind to. That is the
   desktop's last-reception card again, for the same reason.
2. Reply switches to Send with the last-used reply template selected
   ("Reply" or "Reply with picture"), *their call and the SNR filled
   in*, and the inset bound to that picture. The preview already shows
   the composite.
3. The operator keeps the picture already loaded, or picks one, and
   taps Send.

Two taps plus the picture. A CQ is one fewer: Send tab, template chip
"CQ", picture, Send.

### The Send screen

Today: crop view, Choose/Camera, Mode, Send. It gains, in this order:

- **A template row** — horizontally scrolling chips: None, CQ, Reply,
  Reply with picture, then the user's own. One tap selects; the choice
  persists in QSettings as the last-used template.
- **The fields the template uses**, and only those. `theircall` gets
  its own line under the chips (uppercase keyboard, with the callsigns
  of recent receptions offered as completions, since those are the
  stations one is likely answering), because it is the one field that
  can block Send. The **custom fields live in a pop-up**: a "Fields…"
  button, labelled with how many are filled ("Fields (1 of 2)"), opens
  a sheet of labelled text boxes, one per `{field}` in the order they
  appear, with Done to close it. A pop-up rather than inline because a
  template may declare several, the screen is already the picture plus
  a button, and an empty custom field is the normal state rather than
  something to keep in view. The button is absent when the template
  declares none; the built-in None shows neither it nor `theircall`.
- **The preview is the composite.** `Composition::preview()` becomes
  `render(fit(source, framing), substitute(template, fields), last_rx)`,
  and the crop view already displays `preview()` through the image
  provider, so the rule from every other preview in the project holds
  by construction: what is on screen is what `TxRequest.picture`
  carries. The crop gesture keeps operating on the base picture
  underneath; the overlay is drawn on the framed result.

**An empty `{theircall}` blocks Send with a visible reason**, the same
tier as `cwIdProblem`: "Template needs their callsign". The whole point
of a reply template is the address, and "  de KC2G" on the air is a
broken picture. It is not a callsign requirement in the sense the app
rejects — pick None or CQ and Send is enabled again — it is the
template refusing to render a hole. It is also the **only** field that
can block: `{snr}` and every custom field are optional by the
dropped-line rule, and the pop-up never has to be opened.

**Substitution of `{utc}` happens at staging.** `send()` stages
`Composition::preview()`, which is the moment the operator committed
to, so the time on the picture is the time it went out to within the
VOX leader.

### Editing a template on the phone

A separate screen, reached from Settings > Templates, because it is done
occasionally and never mid-QSO. It lists built-ins and user templates
with Duplicate, Rename, Delete, Import (a file) and Share (the JSON,
through the system sheet). The editor for one template:

- **The rendered preview on top**, with a sample picture and sample
  field values so the holes read as text. It is `render()`'s output,
  as everywhere.
- **The item list below** — "Text: `{theircall} de {mycall}`", "Inset:
  last received" — with Add text / Add last received / Add image.
- **A property sheet per item**, deliberately smaller than the
  desktop's: the text, with an **Insert field** menu so nobody types
  braces on a phone keyboard; size as a slider; a row of color
  swatches; and **position as a 3x3 anchor grid** — corners, edges,
  centre — which writes the document's existing `anchor` plus an `x`/`y`
  at that corner with a margin. Drag on the preview for fine placement,
  with the same normalized-coordinate arithmetic the desktop editor
  uses. No rotation, no font, no stroke width: the JSON keeps them if a
  desktop wrote them, the phone just does not offer to change them.

The anchor grid is the phone's answer to "hard to edit live on a small
screen": nine positions cover every template anyone has actually
wanted, and the model already has the field.

### The desktop

Less new than it looks, and it fixes a real gap: **the desktop persists
no overlay at all today** — restart the app and the composition is
gone. The transmit panel gets a **Template** combo, **Save as
template…**, and the same derived fields: their call, and the custom
fields as a pane of labelled text boxes rather than a pop-up, since a
desktop has the room and `FlowLayout` already exists for exactly this
shape. The
existing editor edits the *template* — the text box shows the raw
`{theircall}`, the preview shows it substituted with the current fields,
which is the same preview-is-output rule with substitution in the
path. A drag or an edit changes the document and, if the operator
chooses to save, the template file; the speculative optimizer's
`documentChanged` hookup is unchanged, since a substituted document is
just a document.

## What this does not do

- No automatic caption and no callsign requirement. "None" is the
  default and the first-transmit prompt is unchanged.
- No macro language beyond the table and `{field}`. Not conditionals,
  not arithmetic, not a logbook. The one rule with any logic in it is
  the dropped empty line, and a custom field is a text box, never a
  choice list or a number.
- No template sync or cloud anything. A template is a file; files can
  be shared.
- The phone editor does not aim at the desktop's feature set. Anything
  it cannot express is made on the desktop and shared over.

## Build and layering

`SSTVAE_BUILD_OVERLAY` goes **ON** in `native/android-app/CMakeLists.txt`.
The app already links Qt Gui and runs a `QGuiApplication`, so
`render.cpp` — QtGui-only by `check_layering.py`'s rule — compiles
there as is; the "not planned" in `docs/android.md`'s table was a
decision, not a constraint. The default face on Android is Roboto and
on the desktop whatever Qt finds, so the *same* template previews
slightly differently on the two platforms. That is harmless: the
sending station's render is what goes on the air, so both ends of a
QSO see identical pixels. Bundling one font in both apps would make
previews match too; worth doing later, not needed first.

`template.cpp` goes in `sstvae_core` with `test_overlay_template.cpp`
covering: each substitution rule, `{{` escaping, unknown placeholders
kept, the dropped-line rule taking a line's literal text with its
empty placeholders and leaving a placeholder-free line alone, a line
with one filled and one empty placeholder kept intact, `{field}`
parsing (a label with spaces, the same label twice being one field,
`{field}` with no label left literal as an unknown placeholder),
`placeholders()` enumerating custom fields in first-appearance order,
`{snr}` formatting and its empty case, a named template round-tripping
through `to_json`/`from_json`, and a nameless document loading as a
template.
The Python mirror gets the same cases in `tests/test_overlay.py`, and
`tests/test_native_overlay.py` diffs the two on a corpus of documents.
New settings keys (`grid`, `name`, the last-used template, the field
values if they persist) go through the non-default fixture discipline in
`tests/test_native_settings.py` on the desktop, and its QSettings
counterpart on Android.

## Sequencing

1. **`template` module, both implementations, plus the four built-ins.**
   Pure functions and files; fully tested without Qt. **Done
   2026-09-14**: `sstvae/overlay/template.py` (reference),
   `native/core/overlay/template.{hpp,cpp}` in `sstvae_core`, `name` on
   the document in both, the three template files in
   `sstvae/overlay/templates/` (shipped as package data; the C++ test
   reads the same files), `test_overlay_template.cpp`, the rule tests in
   `tests/test_overlay.py` and a corpus parity test in
   `tests/test_native_overlay.py`. Two details the spec left open and
   the code settled: a whitespace-only value counts as empty, and a
   label is normalized by trimming and collapsing whitespace, so
   `{field  Comment }` is `{field Comment}`.
2. **Desktop: Template combo, Save as, fields row.** **Done
   2026-09-14.** `TransmitPanel` gained a Template combo (None plus the
   three built-ins plus whatever is in `config().folders.template_dir`,
   read by the new `overlay::load_templates`/`load_builtin_templates`
   in `native/core/overlay/template_catalog.{hpp,cpp}`), a "Save as
   template..." button, a "Reply fields" box (Their call, plus a
   "Custom fields..." pop-up), and new `grid`/`name` station settings
   feeding `{grid}`/`{name}`. `OverlayEditor` gained `set_fields`:
   the *document* it edits stays the raw template (what the text box
   shows and what Save writes), while paint, hit-testing and the
   selection handle all substitute first, since that is what is
   actually on screen — the design's own "what you arrange is what
   goes on the air" rule, extended to a template's holes.
   Two departures from the sketch above, both forced by a real
   constraint rather than chosen freely. **Custom fields were a
   pop-up on the desktop too, and are not any more** (superseded
   2026-09-15, see the rectangles/z-order/palette bullet below): the
   real constraint stands — `PaneContainer::equalise_strips` does not
   re-run when a strip's content changes *shape* (the exact bug
   `test_tx_panel.cpp` already guards) — but it turns out to admit a
   fixed *number* of live rows rather than only a pop-up button. The
   fields box is a fixed shape, always, whatever the template needs —
   only `setEnabled` (plus a label's text and a value), matching that
   file's own rule; a template asking for more than fit inline still
   gets a pop-up, now for the overflow only. **This does not close the
   overlay-lost-on-restart gap**
   as this bullet used to promise: an ad-hoc composition never saved
   as a template is still gone on restart, exactly as before — what
   changed is that saving one is now possible. And **there is no
   hard Send block on an empty `{theircall}`**, unlike the phone
   flow below: the substituted canvas already shows the gap live (a
   filled-in line with nothing where the call should be), which is
   the same "must be visible" property the phone's block exists to
   guarantee, so the block itself was left for later rather than
   risking the panel's existing Send-enablement logic sight unseen.

   **Follow-on desktop work, 2026-09-15 — not in the original design,
   requested afterward.** Four additions to the same panel and editor,
   all documented in CLAUDE.md's `sstvae/overlay/` and desktop-status
   bullets rather than repeated here in full:
   - A third overlay item, `RectItem` — filled and/or stroked, each
     independently "none"/"solid"/"gradient" (linear only: two colors
     and an angle). Lives in the *core* model/render, both languages,
     not just the desktop, because that is where `TextItem`/`ImageItem`
     live — so it is available to a future Android editor (step 4)
     without re-deriving it there.
   - Stacking-order controls (Raise/Lower/To front/To back), acting on
     whatever is selected regardless of item type.
   - The Add-item row became an icon palette instead of sentence
     buttons.
   - **Custom fields moved inline**, which is the change the
     departure note above records — up to `MAX_INLINE_CUSTOM_FIELDS`
     (4) live, always-present, live-updating rows in `fields_box_`
     itself, with the pop-up demoted to overflow-only. This is a
     genuine second look at the "pop-up rather than a pane" call step
     2 made: the constraint that forced it (a strip's shape may not
     change after construction) turned out to allow a fixed *number*
     of rows just as well as a fixed *two* rows did — the pop-up was
     never load-bearing for the constraint, only for not having
     thought of the fixed-count version yet.
   - "Save as template..." now defaults its name prompt to the loaded
     template's own name.

   **A second look at the same panel, hours later, same day.** The
   first pass put scale, rotation, color/fill/stroke and stacking order
   all in the "Selected item" box — "way too many buttons" (Andrew).
   Reworked: scale and rotation are a drag handle plus a keyboard
   shortcut on the item itself now (`OverlayEditor`'s new rotate handle
   and `+`/`-`/`[`/`]`), and color/gradient/stroke/order/Remove moved
   into a floating panel that appears next to the selection and only
   while something is selected (`TransmitPanel::build_selection_palette`,
   parented to the editor rather than to the fixed-shape control strip,
   which is what lets its rows actually show and hide by item type
   instead of merely disabling). The text editor stays where it was,
   out of line below the canvas — inline is still not solved, because
   of template field substitution: showing the substituted text while
   editing the raw `{placeholder}` underneath it needs more thought.
   See CLAUDE.md's desktop-status bullet for the fuller account.
3. **Android: overlay ON, template chips and fields on Send, Reply from
   Pictures and Listen, the reply binding in `Composition`.** This is
   the step that delivers an addressed reply from a phone. **Done
   2026-09-14, with the same caveat every other Android C++/QML change
   in this project carries: written and reviewed with no NDK toolchain
   available, so it has not been compiled** (`docs/android.md` records
   the identical situation for the Hamlib cross-build and the Java).
   What exists:
   - `Composition` gained `set_template`/`template_doc`,
     `set_fields`/`fields`, and `set_reply_target`/`has_reply_target`/
     `reply_target_callsign`/`reply_target_snr_db`; `preview()` is now
     `render(fit(source, framing), substitute(template, fields),
     last_rx)`, with `last_rx` resolving to the reply target's picture
     when one is set and to `Session`'s newest reception otherwise —
     cached by path so a drag or a keystroke does not re-decode a PNG
     every frame.
   - `Session` gained `LastReception` (path, callsign, SNR) and
     `last_reception()`, set from the same `save_reception()` call that
     writes the sidecar — durable, unlike the ~2 s-lived shared state
     `rx/engine` publishes, which is what the Listen-tab button above
     needs to still have something to bind to after that state is gone.
   - `Transmitter` gained the template chip list, `theirCall`, the
     custom-field map, `templateFieldProblem` (blocks Send exactly like
     `cwIdProblem`), and `replyTo()`. The built-ins load from a Qt
     resource (`RESOURCES` on the `sstvae_android` qml module,
     aliased to `templates/<name>.json` so the runtime path matches
     what `QFile` needs — a `qrc:` URL, which is what QML's own
     `source` properties take, opens nothing there); the operator's own
     saved templates would load from the app's files directory the same
     way the desktop reads `folders.template_dir`, except nothing writes
     one there yet (that is step 4). Custom field values are cleared on
     Send; `theircall` and `{snr}` are not, and change only on the next
     Reply.
   - Reply buttons on Pictures' list delegate, the full-screen viewer,
     and the Listen tab's status row (gated on `hasLiveImage`, per the
     "if there's room" flow above) all call the same
     `Transmitter::replyTo(path, callsign, snrDb)`.
   - `{grid}`/`{name}` have no station setting on Android yet, unlike
     the desktop's new Settings > Transmit fields — left empty, which
     drops a template line that uses only them (rule 2) rather than
     failing to build one at all. CQ still works with no grid set; it
     just prints one fewer line. Worth adding if the built-in CQ
     template turns out to be used on the phone in practice.
   - Caught while re-verifying the desktop build after this step:
     `sstvae_copy_builtin_templates` (step 2) resolved its template
     source path from `CMAKE_CURRENT_SOURCE_DIR`, which inside a CMake
     function is the *caller's* directory, not the defining file's — so
     the call from `tests/CMakeLists.txt` (added in step 2, for
     `test_tx_panel`) looked one directory short and failed the
     function's own existence check outright. Fixed by capturing the
     path once in `native/CMakeLists.txt`'s own scope instead of
     recomputing it inside the function; `native/build-gui`'s full
     `ctest` (33/33) and `pytest --native` (407 passed) were re-run
     after the fix and are unaffected otherwise, since nothing else in
     this step touched `native/core/` or `native/gui/`.
   **Step 3 was compiled for the first time on 2026-09-21**, when a
   machine with an NDK came to hand, and it did not build. Two link
   errors, both of the kind only a compiler finds: the app calls
   `overlay::render` (`Composition::preview`, which is the rule this
   design is built on) while `android-app/CMakeLists.txt` still forced
   `SSTVAE_BUILD_OVERLAY OFF` from when this app had no overlay at all,
   and it did not link `sstvae_overlay_render` either. Both fixed
   there; nothing in the step's own logic moved, and it is still
   untested on a device.

4. **Sharing a template between stations.** **Done 2026-09-21**, and
   not in the original design — a phone with no editor (step 4) can
   only use templates it shipped with, and the desktop had no way to
   hand it one.

   There is deliberately **no share format**: the payload is
   `overlay::to_json(doc, -1)`, the template's own file contents
   written compactly, so anything that already understands a template
   understands this and an operator can read it. The desktop's
   `ShareDialog` (Share... beside Save/Delete) draws it as a QR code
   and offers the same text to copy; Android's Send screen has an
   "Import…" chip that pastes it back, via
   `Transmitter::importTemplate`.

   Four things this settled.
   - **The code carries the whole thing with room to spare.** The
     largest shipped template is 447 compact bytes, which is 73 modules
     (version 15) at ECC-L; the ceiling is 2953 bytes, roughly a dozen
     decorated items, and past it the dialog says so and points at the
     text rather than drawing something unreadable.
   - **The quiet zone is 8 modules, not the specification's 4, and
     that was measured.** At 4, OpenCV's detector could not find the
     code at all until the image was given a wider border; at 6 and 8
     it read all 447 bytes first time. The margin is painted *into*
     the image for a second reason: the app's palette is dark, so a
     code relying on the dialog behind it would have a dark quiet zone.
   - **An imported document is not handed the local filesystem**
     (`overlay::sanitize_imported`). A `TextItem::font` and a
     non-`last_rx` `ImageItem::source` name a file on the machine that
     wrote them; they cannot mean anything on the machine that reads
     them, and on a similar machine honouring one lets a document the
     operator did not write put a file the operator did not pick into a
     transmission. Both are cleared — text falls back to its
     `font_family` request, which travels correctly being a name rather
     than a path, and an inset falls back to `last_rx`.
   - **Encoding is vendored; decoding is Google Play services'**
     (`native/third_party/qrcodegen/`, `TemplateScanner.java`). Encoding
     is a few hundred lines with an exact answer. Decoding is image
     processing — perspective, lighting, blur — so the phone uses ML
     Kit's *unbundled* code scanner (Andrew, 2026-09-21: the module is
     already on almost every phone, since any app that scans a QR code
     pulled it in), which needs no camera permission and adds nothing
     to the APK. The manifest asks for the module at install time. The
     clipboard stays as the path for a phone with no camera, or one
     whose Play services has not got the module yet, and both hand
     their result to the same `importTemplate`.

   Verified end to end rather than structurally: a screenshot of the
   real dialog (`sstvae-gui-shot --share`) decodes with OpenCV back to
   the exact shipped template, byte for byte. `test_share.cpp` pins
   what a test can pin without a decoder — the payload round-trip, the
   sanitizer, the finder patterns and the quiet zone.

5. **Android: the template editor screen.** Polish; steps 3 and 4 work
   with the built-ins, desktop-made files and shared templates before
   this exists.

## Settled on review (2026-09-14)

- **The report is `{snr}`, not a typed RSV.** Measured, already
  recorded, and never a chore. Anyone who wants the visual half writes
  `{field Visual quality}` into their own template.
- **Custom fields exist, are never mandatory, and are a pop-up on the
  phone.** They are the free-form path and the structured path at once.

- **Reply is on the Listen tab too** (Andrew, 2026-09-14: "if there's
  room" — there is, on the row the live image already opens; see the
  flow above). It costs the tuning instrument nothing because it only
  appears in the layout that has already made space for a picture.

- **What persists between overs** (Andrew, 2026-09-14): `theircall`
  and `{snr}` follow the reply target, so they change only when a
  different reception is replied to, and survive a rotation but not a
  relaunch — a stale callsign on a reply is exactly the failure this
  feature exists to avoid. Custom field values are **cleared on
  Send**: a comment is written for one over, and the one thing worse
  than retyping it is transmitting last over's comment again.

Nothing is open.
