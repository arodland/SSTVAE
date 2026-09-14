// Templates on disk: the built-ins shipped with the app, and the
// operator's own saved to a folder.
//
// Loading never fails, the same rule `settings::load` follows: a
// directory that does not exist, or one file in it that fails to
// parse, must not take the whole picker down. `sstvae/overlay/
// template.py`'s `builtin_templates()` is the reference for the
// built-in set and its order (CQ, Reply, Reply with picture); this is
// the C++ counterpart, reading the *files* rather than embedding them,
// because the files are what both apps and the Python CLI ship.

#ifndef SSTVAE_OVERLAY_TEMPLATE_CATALOG_HPP
#define SSTVAE_OVERLAY_TEMPLATE_CATALOG_HPP

#include <filesystem>
#include <string>
#include <vector>

#include "overlay/model.hpp"

namespace sstvae::overlay {

struct LoadedTemplate {
    Doc doc;
    // Absolute path it was read from -- so "overwrite this one" and
    // "this name is taken" can be answered without re-deriving a
    // filename from the document's own `name`, which the operator is
    // free to change after the fact.
    std::filesystem::path path;
};

// Every `*.json` in `dir`, sorted by filename, skipping anything that
// is not valid JSON or not an overlay document -- one bad file must
// cost that file, not the picker. `dir` need not exist.
std::vector<LoadedTemplate> load_templates(const std::filesystem::path& dir);

// The three built-ins, in the fixed order the UI lists them
// (docs/overlay-templates.md): CQ, Reply, Reply with picture. Reads
// `cq.json` / `reply.json` / `reply-picture.json` from `dir`; a missing
// or unparseable one is skipped rather than failing the whole list, so
// a build that could not stage the data files still shows the
// operator's own templates from `load_templates`.
std::vector<Doc> load_builtin_templates(const std::filesystem::path& dir);

// A filesystem-safe stem for `name`, for choosing where to save a new
// template: lower-cased, runs of anything but `[a-z0-9]` collapsed to a
// single `-`, trimmed of leading/trailing `-`. Empty input (or input
// that is entirely punctuation) yields "template" rather than an empty
// path.
std::string slugify(const std::string& name);

}  // namespace sstvae::overlay

#endif
