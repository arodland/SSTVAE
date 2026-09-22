// Templates: an overlay document with holes in it.
//
// The C++ half of `sstvae/overlay/template.py`, which is the reference;
// `tests/test_native_overlay.py` holds the two to identical output. A
// template is an ordinary `Doc` whose text items contain placeholders,
// and this is the pure string processing that fills them in. Qt-free
// and in `sstvae_core`, so it builds and is tested with `--no-overlay`
// and on every CI job; the renderer never sees a placeholder.
//
// Placeholders are `{name}` inside `TextItem::text`: the built-in
// names in `BUILTIN_FIELDS`, or `{field Label}` for a custom field the
// operator is offered as a text box. Three rules, the whole contract:
//
// 1. An unknown placeholder is left literally in the text -- a typo is
//    visible in the preview rather than deleted from the air. `{{` and
//    `}}` are literal braces.
// 2. A line whose placeholders are all empty is dropped, literal text
//    and all: `SNR {snr}` vanishes whole when there is no SNR. Only
//    whole lines, never partial text; a line with no placeholders is
//    never touched.
// 3. The stored template is never mutated: `substitute` returns a copy.

#ifndef SSTVAE_OVERLAY_TEMPLATE_HPP
#define SSTVAE_OVERLAY_TEMPLATE_HPP

#include <array>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "overlay/model.hpp"

namespace sstvae::overlay {

// The closed set of built-in placeholder names. Adding one is a format
// change for every template in the field -- an older build renders the
// new name literally (rule 1), which is the intended behaviour, but it
// is still what happens.
inline constexpr std::array<std::string_view, 8> BUILTIN_FIELDS = {
    "mycall", "grid", "name", "theircall", "snr", "utc", "date", "mode"};

// What to fill the holes with. `builtin` is keyed by placeholder name,
// `custom` by label; anything missing is empty.
struct Fields {
    std::map<std::string, std::string> builtin;
    std::map<std::string, std::string> custom;
};

// What a template asks for, in order of first appearance and without
// repeats. This derives the per-over form: no schema, nothing to keep
// in sync with the text.
struct Placeholders {
    std::vector<std::string> builtin;
    std::vector<std::string> custom;
};

// Trim and collapse internal whitespace, so `{field  Comment}` and
// `{field Comment}` are one field. ASCII whitespace, like the rest of
// this file; the reference's `str.split()` is wider, and a label with
// non-ASCII whitespace in it is the one input the two disagree on.
std::string normalize_label(std::string_view label);

Placeholders placeholders(const Doc& doc);

// Rules 1 and 2 on one text item's content.
std::string substitute_text(const std::string& text, const Fields& fields);

// A copy of `doc` with every hole filled (rule 3). Image items pass
// through unchanged.
Doc substitute(const Doc& doc, const Fields& fields);

// The `{snr}` value: an integer with its unit, "12 dB"; empty for
// nothing measured. `nearbyint`, half to even, matching Python's
// `round` -- so both implementations print the same figure for the
// same sidecar.
std::string format_snr(std::optional<double> snr_db);

}  // namespace sstvae::overlay

#endif
