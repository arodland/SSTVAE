#include "overlay/model.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace sstvae::overlay {

using json = nlohmann::json;

namespace {

// Same shape as the settings reader: a wrong type keeps the default and
// is reported, rather than throwing out the whole document over one bad
// field.
struct Reader {
    const json& obj;
    std::string where;
    std::vector<Note>* notes;

    void note(const std::string& key, const std::string& problem) const {
        if (notes) notes->push_back({where + "." + key, problem});
    }

    const json* find(const std::string& key) const {
        const auto it = obj.find(key);
        if (it == obj.end() || it->is_null()) return nullptr;
        return &*it;
    }

    void get(const std::string& key, std::string& dst) const {
        if (const json* v = find(key)) {
            if (v->is_string()) dst = v->get<std::string>();
            else note(key, "expected a string");
        }
    }

    void get(const std::string& key, double& dst) const {
        if (const json* v = find(key)) {
            if (v->is_number()) dst = v->get<double>();
            else note(key, "expected a number");
        }
    }

    void report_unknown(std::initializer_list<std::string_view> known) const {
        for (const auto& item : obj.items()) {
            const std::string_view key = item.key();
            if (key == "type") continue;
            if (std::find(known.begin(), known.end(), key) == known.end()) {
                note(std::string(key), "not a field this build knows about (ignored)");
            }
        }
    }
};

TextItem read_text(const Reader& r) {
    TextItem t;
    r.get("text", t.text);
    r.get("x", t.x);
    r.get("y", t.y);
    r.get("size", t.size);
    r.get("color", t.color);
    r.get("stroke_color", t.stroke_color);
    r.get("stroke_width", t.stroke_width);
    r.get("font", t.font);
    r.get("anchor", t.anchor);
    r.get("align", t.align);
    r.get("line_spacing", t.line_spacing);
    r.get("rotation", t.rotation);
    r.report_unknown({"text", "x", "y", "size", "color", "stroke_color", "stroke_width",
                      "font", "anchor", "align", "line_spacing", "rotation"});
    return t;
}

ImageItem read_image(const Reader& r) {
    ImageItem i;
    r.get("source", i.source);
    r.get("x", i.x);
    r.get("y", i.y);
    r.get("width", i.width);
    r.get("border", i.border);
    r.get("border_color", i.border_color);
    r.get("opacity", i.opacity);
    r.get("rotation", i.rotation);
    r.get("anchor", i.anchor);
    r.report_unknown({"source", "x", "y", "width", "border", "border_color", "opacity",
                      "rotation", "anchor"});
    return i;
}

}  // namespace

Doc from_json(const std::string& text, std::vector<Note>* notes) {
    json root = json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded()) throw std::runtime_error("overlay document is not valid JSON");
    if (!root.is_object()) throw std::runtime_error("overlay document is not a JSON object");

    Doc doc;
    if (const auto it = root.find("version"); it != root.end() && it->is_number_integer()) {
        doc.version = it->get<int>();
    }
    if (doc.version > DOC_VERSION) {
        throw std::runtime_error(
            "overlay document version " + std::to_string(doc.version) +
            " is newer than this build understands (max " + std::to_string(DOC_VERSION) +
            ")");
    }

    const auto items = root.find("items");
    if (items == root.end() || items->is_null()) return doc;
    if (!items->is_array()) throw std::runtime_error("overlay 'items' is not an array");

    int index = 0;
    for (const json& raw : *items) {
        const std::string where = "items[" + std::to_string(index++) + "]";
        if (!raw.is_object()) {
            if (notes) notes->push_back({where, "is not an object (ignored)"});
            continue;
        }
        std::string kind = "text";
        if (const auto t = raw.find("type"); t != raw.end() && t->is_string()) {
            kind = t->get<std::string>();
        }
        const Reader r{raw, where, notes};
        if (kind == "text") {
            doc.items.emplace_back(read_text(r));
        } else if (kind == "image") {
            doc.items.emplace_back(read_image(r));
        } else if (notes) {
            // Forward compatibility: a later version's item kind.
            notes->push_back({where, "unknown item type '" + kind + "' (ignored)"});
        }
    }
    return doc;
}

std::string to_json(const Doc& doc, int indent) {
    json items = json::array();
    for (const Item& item : doc.items) {
        if (const auto* t = std::get_if<TextItem>(&item)) {
            items.push_back({{"text", t->text},
                             {"x", t->x},
                             {"y", t->y},
                             {"size", t->size},
                             {"color", t->color},
                             {"stroke_color", t->stroke_color},
                             {"stroke_width", t->stroke_width},
                             {"font", t->font.empty() ? json(nullptr) : json(t->font)},
                             {"anchor", t->anchor},
                             {"align", t->align},
                             {"line_spacing", t->line_spacing},
                             {"rotation", t->rotation},
                             {"type", "text"}});
        } else {
            const auto& i = std::get<ImageItem>(item);
            items.push_back({{"source", i.source},
                             {"x", i.x},
                             {"y", i.y},
                             {"width", i.width},
                             {"border", i.border},
                             {"border_color", i.border_color},
                             {"opacity", i.opacity},
                             {"rotation", i.rotation},
                             {"anchor", i.anchor},
                             {"type", "image"}});
        }
    }
    const json root = {{"version", doc.version}, {"items", items}};
    return indent >= 0 ? root.dump(indent) : root.dump();
}

}  // namespace sstvae::overlay
