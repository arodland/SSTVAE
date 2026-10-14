#include "qrss/pypath.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sstvae::qrss::pypath {

std::pair<std::string, std::vector<std::string>> split(std::string_view text) {
    std::string root;
    std::string_view rest = text;
    if (!rest.empty() && rest[0] == '/') {
        // POSIX keeps exactly two leading slashes (implementation-defined
        // there); one, or three and more, are one.
        if (rest.size() >= 2 && rest[1] == '/' && !(rest.size() >= 3 && rest[2] == '/')) {
            root = "//";
            rest.remove_prefix(2);
        } else {
            root = "/";
            rest.remove_prefix(1);
        }
    }
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= rest.size()) {
        const std::size_t j = std::min(rest.find('/', i), rest.size());
        const std::string_view part = rest.substr(i, j - i);
        if (!part.empty() && part != ".") parts.emplace_back(part);
        i = j + 1;
    }
    return {root, parts};
}

std::string join(const std::string& root, const std::vector<std::string>& parts) {
    std::string out = root;
    for (std::size_t k = 0; k < parts.size(); ++k) {
        if (k) out += '/';
        out += parts[k];
    }
    return out.empty() ? "." : out;
}

std::string pathlib_str(std::string_view text) {
    const auto [root, parts] = split(text);
    return join(root, parts);
}

std::string pathlib_name(std::string_view text) {
    const auto [root, parts] = split(text);
    return parts.empty() ? std::string() : parts.back();
}

std::string pathlib_suffix(std::string_view text) {
    const std::string name = pathlib_name(text);
    const std::size_t dot = name.rfind('.');
    // 0 < i < len(name) - 1: a leading dot is not a suffix, nor a trailing one.
    if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) return {};
    return name.substr(dot);
}

std::string child(std::string_view dir, const std::string& name) {
    auto [root, parts] = split(dir);
    parts.push_back(name);
    return join(root, parts);
}

std::string with_name(std::string_view p, const std::string& name) {
    auto [root, parts] = split(p);
    if (parts.empty()) throw std::invalid_argument(pathlib_str(p) + " has an empty name");
    parts.back() = name;
    return join(root, parts);
}

}  // namespace sstvae::qrss::pypath
