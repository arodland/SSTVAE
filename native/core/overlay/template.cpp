#include "overlay/template.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sstvae::overlay {

namespace {

enum class Kind { Text, Builtin, Custom };

// `value` is literal text with escapes resolved for Text, the name for
// Builtin, the normalized label for Custom.
struct Token {
    Kind kind;
    std::string value;
};

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

bool is_blank(std::string_view s) {
    return std::all_of(s.begin(), s.end(), is_space);
}

bool is_builtin(std::string_view inner) {
    if (inner.empty()) return false;
    for (char c : inner) {
        if (c < 'a' || c > 'z') return false;
    }
    return std::find(BUILTIN_FIELDS.begin(), BUILTIN_FIELDS.end(), inner) !=
           BUILTIN_FIELDS.end();
}

// `{field Label}`: the word, at least one space, then a label with
// something in it. `{field}` and `{field }` are not custom fields and
// fall through to rule 1 as unknown placeholders.
bool is_custom(std::string_view inner, std::string& label) {
    static constexpr std::string_view WORD = "field";
    if (inner.size() <= WORD.size() || inner.substr(0, WORD.size()) != WORD) return false;
    if (!is_space(inner[WORD.size()])) return false;
    label = normalize_label(inner.substr(WORD.size()));
    return !label.empty();
}

// Placeholders never span a line, so the tokenizer works a line at a
// time; `substitute_text` decides per line what to do with the result.
std::vector<Token> tokenize_line(std::string_view line) {
    std::vector<Token> tokens;
    std::string literal;
    auto flush = [&] {
        if (!literal.empty()) {
            tokens.push_back({Kind::Text, literal});
            literal.clear();
        }
    };
    const std::size_t n = line.size();
    std::size_t i = 0;
    while (i < n) {
        const char c = line[i];
        if (c == '{') {
            if (i + 1 < n && line[i + 1] == '{') {
                literal += '{';
                i += 2;
                continue;
            }
            const std::size_t end = line.find('}', i + 1);
            if (end != std::string_view::npos) {
                const std::string_view inner = line.substr(i + 1, end - i - 1);
                if (inner.find('{') == std::string_view::npos) {
                    std::string label;
                    if (is_builtin(inner)) {
                        flush();
                        tokens.push_back({Kind::Builtin, std::string(inner)});
                        i = end + 1;
                        continue;
                    }
                    if (is_custom(inner, label)) {
                        flush();
                        tokens.push_back({Kind::Custom, label});
                        i = end + 1;
                        continue;
                    }
                }
            }
            // No closing brace, or not a placeholder we know: literal.
            literal += '{';
            i += 1;
            continue;
        }
        if (c == '}') {
            literal += '}';
            i += (i + 1 < n && line[i + 1] == '}') ? 2 : 1;
            continue;
        }
        literal += c;
        i += 1;
    }
    flush();
    return tokens;
}

std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (true) {
        const std::size_t nl = text.find('\n', start);
        if (nl == std::string_view::npos) {
            lines.push_back(text.substr(start));
            return lines;
        }
        lines.push_back(text.substr(start, nl - start));
        start = nl + 1;
    }
}

void push_unique(std::vector<std::string>& v, const std::string& s) {
    if (std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
}

}  // namespace

std::string normalize_label(std::string_view label) {
    std::string out;
    bool pending = false;
    for (char c : label) {
        if (is_space(c)) {
            pending = !out.empty();
        } else {
            if (pending) out += ' ';
            pending = false;
            out += c;
        }
    }
    return out;
}

Placeholders placeholders(const Doc& doc) {
    Placeholders out;
    for (const Item& item : doc.items) {
        const auto* text = std::get_if<TextItem>(&item);
        if (text == nullptr) continue;
        for (const std::string_view line : split_lines(text->text)) {
            for (const Token& t : tokenize_line(line)) {
                if (t.kind == Kind::Builtin) push_unique(out.builtin, t.value);
                else if (t.kind == Kind::Custom) push_unique(out.custom, t.value);
            }
        }
    }
    return out;
}

std::string substitute_text(const std::string& text, const Fields& fields) {
    std::string out;
    bool first = true;
    for (const std::string_view line : split_lines(text)) {
        const std::vector<Token> tokens = tokenize_line(line);
        bool holes = false;
        bool any_filled = false;
        std::string built;
        for (const Token& t : tokens) {
            if (t.kind == Kind::Text) {
                built += t.value;
                continue;
            }
            holes = true;
            const auto& source = t.kind == Kind::Builtin ? fields.builtin : fields.custom;
            const auto it = source.find(t.value);
            // Whitespace-only counts as empty: a stray space in a
            // comment box is not a comment.
            if (it == source.end() || is_blank(it->second)) continue;
            any_filled = true;
            built += it->second;
        }
        if (holes && !any_filled) continue;  // rule 2
        if (!first) out += '\n';
        first = false;
        out += built;
    }
    return out;
}

Doc substitute(const Doc& doc, const Fields& fields) {
    Doc out = doc;
    for (Item& item : out.items) {
        if (auto* text = std::get_if<TextItem>(&item)) {
            text->text = substitute_text(text->text, fields);
        }
    }
    return out;
}

std::string format_snr(std::optional<double> snr_db) {
    if (!snr_db) return "";
    const long long whole = static_cast<long long>(std::nearbyint(*snr_db));
    return std::to_string(whole) + " dB";
}

}  // namespace sstvae::overlay
