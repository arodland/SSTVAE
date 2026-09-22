#include "overlay/template_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

namespace sstvae::overlay {

namespace {

namespace fs = std::filesystem;

bool try_load(const fs::path& path, Doc* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::stringstream buf;
    buf << in.rdbuf();
    try {
        *out = from_json(buf.str());
    } catch (const std::exception&) {
        return false;  // malformed JSON, or a version this build refuses
    }
    return true;
}

}  // namespace

std::vector<LoadedTemplate> load_templates(const fs::path& dir) {
    std::vector<LoadedTemplate> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() == ".json") files.push_back(entry.path());
    }
    // A directory iterator's order is unspecified; a fixed order is
    // what makes the picker's list reproducible between runs and
    // between platforms.
    std::sort(files.begin(), files.end());

    for (const fs::path& path : files) {
        Doc doc;
        if (try_load(path, &doc)) out.push_back({std::move(doc), path});
    }
    return out;
}

std::vector<Doc> load_builtin_templates(const fs::path& dir) {
    std::vector<Doc> out;
    for (const char* stem : {"cq", "reply", "reply-picture"}) {
        Doc doc;
        if (try_load(dir / (std::string(stem) + ".json"), &doc)) {
            out.push_back(std::move(doc));
        }
    }
    return out;
}

std::string slugify(const std::string& name) {
    std::string out;
    bool need_dash = false;
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            if (need_dash && !out.empty()) out += '-';
            need_dash = false;
            out += static_cast<char>(std::tolower(c));
        } else {
            need_dash = true;
        }
    }
    return out.empty() ? "template" : out;
}

}  // namespace sstvae::overlay
