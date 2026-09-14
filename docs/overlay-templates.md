# Design: overlay templates, and the overlay on Android

> Written 2026-09-14. **Not implemented.** A proposal for the UI flow,
> the storage format and the sequencing; the decisions marked *open* at
> the end are Andrew's to make before any of it is built.

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

**Placeholders are `{field}` inside `TextItem.text`.** The set is
closed and small:

| Placeholder   | Source                                    |
|---------------|-------------------------------------------|
| `{mycall}`    | station setting (exists: `station/callsign`) |
| `{grid}`      | station setting (new, optional)           |
| `{name}`      | station setting (new, optional)           |
| `{theircall}` | per-over field                            |
| `{rsv}`       | per-over field, the SSTV picture report   |
| `{utc}`       | time at Send, HH:MM                       |
| `{date}`      | date at Send, ISO                         |
| `{mode}`      | the transmit mode                         |

Three rules, all in one pure function `substitute(doc, fields) -> doc`:

1. **An unknown placeholder is left literally in the text.** A typo
   must be visible in the preview, not silently deleted from the air.
   `{{` and `}}` are literal braces, for the one person who wants one.
2. **A line whose only content is empty placeholders is dropped.** This
   is the single piece of cleverness and it is what lets one "Reply"
   template serve with and without a report: `{theircall} de {mycall}`
   newline `RSV {rsv}` loses its second line when no report is typed.
   *Only* whole lines, never partial text, so the rule cannot produce
   a half-sentence.
3. **The stored template is never mutated.** Substitution runs on a
   copy at preview and again at Send; the file keeps its holes.

Alongside it, `placeholders(doc) -> set` says which fields a template
uses. That is what derives the form: no schema, no per-template
metadata, nothing to keep in sync with the text.

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
- **CQ** — `CQ CQ CQ` / `de {mycall}` / `{grid}`, top-left, large.
- **Reply** — `{theircall} de {mycall}` / `RSV {rsv}`, top-left.
- **Reply with picture** — Reply plus a `last_rx` inset, bottom-right.

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

`Composition` gains a **reply target**: a reception's path and its
callsign. While one is set, `last_rx` resolves to that picture and
`{theircall}` is prefilled from it (editable — the beacon can be
absent or garbled, in which case the field is empty and the operator
types it). With none set, `last_rx` is the newest reception, as today.
The target is runtime state and is not stored in the template, which is
what keeps a template a template.

## The flows

### Android, mid-QSO (the case that matters)

1. A picture from W1XYZ completes. Its card in Pictures, and the viewer,
   carry a **Reply** button beside Share. The Listen tab shows the same
   button on its completion line while the reception is recent, for the
   case where the operator is watching it come in.
2. Reply switches to Send with the last-used reply template selected
   ("Reply" or "Reply with picture"), *their call filled in*, and the
   inset bound to that picture. The preview already shows the composite.
3. The operator keeps the picture already loaded, or picks one, and
   taps Send.

Two taps plus the picture. A CQ is one fewer: Send tab, template chip
"CQ", picture, Send.

### The Send screen

Today: crop view, Choose/Camera, Mode, Send. It gains, in this order:

- **A template row** — horizontally scrolling chips: None, CQ, Reply,
  Reply with picture, then the user's own. One tap selects; the choice
  persists in QSettings as the last-used template.
- **The fields the template uses**, and only those: a `theircall` field
  (uppercase keyboard, with the callsigns of recent receptions offered
  as completions, since those are the stations one is likely answering)
  and an `rsv` field when the template names them. A CQ template shows
  no fields at all.
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
template refusing to render a hole. `{rsv}` empty is fine, by the
dropped-line rule.

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
  braces on a phone keyboard; size as a slider; a row of colour
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
template…**, and the same derived fields row (their call, RSV). The
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
- No macro language beyond the table. Not conditionals, not
  arithmetic, not a logbook. The one rule with any logic in it is the
  dropped empty line.
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
kept, the dropped-line rule not firing on a line with any literal text,
`placeholders()` enumeration, a named template round-tripping through
`to_json`/`from_json`, and a nameless document loading as a template.
The Python mirror gets the same cases in `tests/test_overlay.py`, and
`tests/test_native_overlay.py` diffs the two on a corpus of documents.
New settings keys (`grid`, `name`, the last-used template, the field
values if they persist) go through the non-default fixture discipline in
`tests/test_native_settings.py` on the desktop, and its QSettings
counterpart on Android.

## Sequencing

1. **`template` module, both implementations, plus the four built-ins.**
   Pure functions and files; fully tested without Qt.
2. **Desktop: Template combo, Save as, fields row.** Small, and it
   closes the overlay-lost-on-restart gap on its own.
3. **Android: overlay ON, template chips and fields on Send, Reply from
   Pictures and Listen, the reply binding in `Composition`.** This is
   the step that delivers an addressed reply from a phone.
4. **Android: the template editor screen.** Polish; step 3 works with
   the built-ins and desktop-made files before this exists.

## Open

- **Is `{rsv}` worth a field?** It is the one thing SSTV operators
  exchange besides callsigns, and the dropped-line rule makes it free
  to leave blank. Dropping it makes the form one field on every
  template.
- **Does Reply belong on the Listen tab as well as in Pictures?** The
  proposal says yes, for the operator watching a picture arrive, but it
  is one more control on the screen that is supposed to be the tuning
  instrument.
- **Should the per-over fields persist across launches?** Their call
  probably should within a session and probably should not overnight;
  a wrong stale callsign on a reply is exactly the failure this feature
  exists to avoid. Clearing `theircall` when a *different* reception is
  replied to, and otherwise keeping it, is the proposed compromise.
