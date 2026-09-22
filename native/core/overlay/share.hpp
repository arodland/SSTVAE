// Handing a template to another station.
//
// A template is already a small JSON document, so there is no share
// *format* here beyond the one the files are written in: the payload is
// `to_json(doc, -1)`, which is what the desktop draws as a QR code and
// what both apps accept on the clipboard. Anything that can carry a few
// hundred characters -- a camera, a chat window, an email -- carries a
// template, and a pasted payload is the same text a `.json` in the
// template folder holds.
//
// What needs a function is the other direction: a document that arrives
// from somewhere else must not be trusted with the local filesystem.

#ifndef SSTVAE_OVERLAY_SHARE_HPP
#define SSTVAE_OVERLAY_SHARE_HPP

#include <string>

#include "overlay/model.hpp"

namespace sstvae::overlay {

// A template from another station, made safe to render here.
//
// Two fields in a document name a file on the machine that wrote it: a
// `TextItem::font` and an `ImageItem::source` that is not `last_rx`.
// Neither can mean anything on the machine that receives it -- the
// paths are another computer's -- and on the machine that *sent* it,
// or any machine with a similar layout, honouring one means a document
// the operator did not write can name a file the operator did not pick
// and put it in a transmission. Both are cleared: a text item falls
// back to its `font_family` request (which travels correctly, being a
// name rather than a path), and an image item falls back to `last_rx`,
// which is the only picture reference that means the same thing
// everywhere.
//
// Everything else is ordinary data -- coordinates, colours, the text
// itself -- and is kept as written.
Doc sanitize_imported(Doc doc);

// Whether `text` parses as a template at all. `from_json` throws on
// malformed input, which is the right answer for a file the app wrote
// and the wrong one for whatever was on the clipboard.
bool is_share_payload(const std::string& text);

}  // namespace sstvae::overlay

#endif
