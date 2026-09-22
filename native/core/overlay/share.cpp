#include "overlay/share.hpp"

#include <exception>
#include <variant>

namespace sstvae::overlay {

Doc sanitize_imported(Doc doc) {
    for (Item& item : doc.items) {
        if (auto* text = std::get_if<TextItem>(&item)) {
            text->font.clear();
        } else if (auto* image = std::get_if<ImageItem>(&item)) {
            if (image->source != SOURCE_LAST_RX) image->source = SOURCE_LAST_RX;
        }
    }
    return doc;
}

bool is_share_payload(const std::string& text) {
    try {
        from_json(text);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace sstvae::overlay
