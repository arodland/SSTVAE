# nlohmann/json (vendored)

`json.hpp` v3.12.0 from https://github.com/nlohmann/json (the
`single_include` amalgamation), fetched 2026-07-28. MIT — see `LICENSE`.

Vendored rather than fetched at configure time, for the same reason as
the other `third_party/` entries: a single header with no build system,
and a `FetchContent` would make every CI job depend on GitHub being
reachable during `cmake`.

## Why not QJsonDocument

Qt has a perfectly good JSON parser and arrives in Phase 3 anyway. But
`settings` and `overlay` are both **read by headless code** — the
config is loaded by anything that talks to a radio, and
`sstvae/overlay/` is explicitly designed to stay renderable from the
command line. Using Qt's parser would put a GUI toolkit under both, and
the layering rule permits QtGui only in `core/overlay/`, never QtCore
generally in `core/`.

It is also the reason the config format is plain JSON in the first
place, rather than `QSettings`: readable, diffable, portable between
machines, and loadable without starting Qt.
