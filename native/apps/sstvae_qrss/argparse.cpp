#include "argparse.hpp"

#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qrss/pynum.hpp"
#include "qrss/pytext.hpp"

namespace sstvae::qrss::cli {

namespace {

// argparse's ArgumentError: the message, and the action it names (if any).
struct ArgumentError {
    std::string message;
};

std::string upper(std::string s) {
    for (char& c : s)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return s;
}

std::string dest_of(const std::string& flag) {
    std::string d = flag.substr(flag.find_first_not_of('-'));
    for (char& c : d)
        if (c == '-') c = '_';
    return d;
}

bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

// argparse's _negative_number_matcher: '^-\d+$|^-\d*\.\d+$' (ASCII digits;
// re's `$` also matches before one final newline).
bool negative_number(std::string s) {
    if (!s.empty() && s.back() == '\n') s.pop_back();
    if (s.size() < 2 || s[0] != '-') return false;
    std::size_t i = 1;
    std::size_t before = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i, ++before;
    if (i == s.size()) return before > 0;
    if (s[i] != '.') return false;
    ++i;
    std::size_t after = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i, ++after;
    return i == s.size() && after > 0;
}

// Word-wrap `text` to `width` columns, each line prefixed with `indent`
// spaces except the first (which the caller has already placed).
std::string wrap(const std::string& text, std::size_t first_col, std::size_t indent, std::size_t width) {
    std::istringstream words(text);
    std::string out, word;
    std::size_t col = first_col;
    bool first = true;
    while (words >> word) {
        if (!first && col + 1 + word.size() > width) {
            out += "\n" + std::string(indent, ' ');
            col = indent;
        } else if (!first) {
            out += ' ';
            ++col;
        }
        out += word;
        col += word.size();
        first = false;
    }
    return out;
}

}  // namespace

// --- Args -------------------------------------------------------------------------------

const std::optional<std::string>& Args::text(const std::string& dest) const {
    const auto it = values_.find(dest);
    if (it == values_.end()) throw std::out_of_range("no argument " + dest);
    return it->second;
}

bool Args::on(const std::string& dest) const {
    const auto it = flags_.find(dest);
    if (it == flags_.end()) throw std::out_of_range("no flag " + dest);
    return it->second;
}

std::optional<pynum::PyInt> Args::integer(const std::string& dest) const {
    const std::optional<std::string>& t = text(dest);
    if (!t) return std::nullopt;
    auto v = pynum::py_int(*t, 10);
    if (!v) throw std::logic_error("argument " + dest + " is not an int");
    return v;
}

std::optional<double> Args::real(const std::string& dest) const {
    const std::optional<std::string>& t = text(dest);
    if (!t) return std::nullopt;
    auto v = pynum::py_float(*t);
    if (!v) throw std::logic_error("argument " + dest + " is not a float");
    return v;
}

// --- ArgParser ----------------------------------------------------------------------------

std::string ArgParser::Action::name() const {
    if (positional) return spec.metavar.empty() ? dest : spec.metavar;
    std::string out;
    for (std::size_t k = 0; k < flags.size(); ++k) out += (k ? "/" : "") + flags[k];
    return out;
}

std::string ArgParser::Action::metavar() const {
    if (!spec.metavar.empty()) return spec.metavar;
    if (!spec.choices.empty()) {
        std::string out = "{";
        for (std::size_t k = 0; k < spec.choices.size(); ++k) out += (k ? "," : "") + spec.choices[k];
        return out + "}";
    }
    return positional ? dest : upper(dest);
}

ArgParser::ArgParser(std::string prog, std::string description)
    : prog_(std::move(prog)), description_(std::move(description)) {
    Action h;
    h.help = true;
    h.store_true = true;
    h.flags = {"-h", "--help"};
    h.dest = "help";
    h.spec.help = "show this help message and exit";
    actions_.push_back(std::move(h));
}

ArgParser& ArgParser::positional(const std::string& name, const std::string& help) {
    Action a;
    a.positional = true;
    a.dest = name;
    a.spec.help = help;
    a.spec.required = true;
    actions_.push_back(std::move(a));
    return *this;
}

ArgParser& ArgParser::option(const std::string& flag, ArgSpec spec) {
    Action a;
    a.flags = {flag};
    a.dest = dest_of(flag);
    a.spec = std::move(spec);
    actions_.push_back(std::move(a));
    return *this;
}

ArgParser& ArgParser::flag(const std::string& flag, const std::string& help) {
    Action a;
    a.store_true = true;
    a.flags = {flag};
    a.dest = dest_of(flag);
    a.spec.help = help;
    actions_.push_back(std::move(a));
    return *this;
}

std::string ArgParser::usage() const {
    std::string out = "usage: " + prog_;
    for (const Action& a : actions_) {
        if (a.positional) continue;
        std::string part = a.flags.front();
        if (!a.store_true) part += " " + a.metavar();
        out += " " + (a.spec.required ? part : "[" + part + "]");
    }
    for (const Action& a : actions_)
        if (a.positional) out += " " + a.metavar();
    return out;
}

std::string ArgParser::help() const {
    constexpr std::size_t width = 78, column = 24;
    std::string out = usage() + "\n";
    if (!description_.empty()) out += "\n" + description_ + "\n";
    auto row = [&](const std::string& head, const std::string& help) {
        std::string line = "  " + head;
        if (help.empty()) return line + "\n";
        if (line.size() + 2 > column) line += "\n" + std::string(column, ' ');
        else line += std::string(column - line.size(), ' ');
        return line + wrap(help, column, column, width) + "\n";
    };
    out += "\npositional arguments:\n";
    for (const Action& a : actions_)
        if (a.positional) out += row(a.metavar(), a.spec.help);
    out += "\noptions:\n";
    for (const Action& a : actions_) {
        if (a.positional) continue;
        std::string head;
        for (std::size_t k = 0; k < a.flags.size(); ++k)
            head += (k ? ", " : "") + a.flags[k] + (a.store_true ? "" : " " + a.metavar());
        out += row(head, a.spec.help);
    }
    return out;
}

std::optional<Args> ArgParser::parse(const std::vector<std::string>& args, std::ostream& out, std::ostream& err,
                                     int& status) const {
    // One classified argument: a value ('A'), the '--' marker, an option
    // (possibly ambiguous: several matches), or an unknown option.
    struct Match {
        std::size_t action;
        std::string flag;
        std::optional<std::string> explicit_arg;
        bool sep = false;   // the explicit argument came after '='
    };
    struct Token {
        enum Kind { value, dashdash, option } kind = value;
        std::vector<Match> matches;   // option: empty is an unknown option
    };

    auto lookup = [&](const std::string& flag) -> std::optional<std::size_t> {
        for (std::size_t k = 0; k < actions_.size(); ++k)
            for (const std::string& f : actions_[k].flags)
                if (f == flag) return k;
        return std::nullopt;
    };

    // _parse_optional, for every argument before any is consumed.
    std::vector<Token> tokens;
    bool after_dashdash = false;
    for (const std::string& a : args) {
        Token t;
        if (after_dashdash) {
            t.kind = Token::value;
        } else if (a == "--") {
            t.kind = Token::dashdash;
            after_dashdash = true;
        } else if (a.empty() || a[0] != '-') {
            t.kind = Token::value;
        } else if (const auto k = lookup(a)) {
            t.kind = Token::option;
            t.matches.push_back({*k, a, std::nullopt});
        } else if (a.size() == 1) {
            t.kind = Token::value;
        } else {
            const std::size_t eq = a.find('=');
            std::optional<std::size_t> exact;
            if (eq != std::string::npos) exact = lookup(a.substr(0, eq));
            if (exact) {
                t.kind = Token::option;
                t.matches.push_back({*exact, a.substr(0, eq), a.substr(eq + 1), true});
            } else {
                // _get_option_tuples (allow_abbrev): every option string the
                // part before '=' prefixes. A single-dash argument also
                // matches the short option of its first two characters, with
                // the rest as its explicit argument ("-hX"). "-=..." prefixes
                // every option, so it is ambiguous, as there.
                const std::string prefix = a.substr(0, eq);
                const bool sep = eq != std::string::npos;
                std::optional<std::string> explicit_arg;
                if (sep) explicit_arg = a.substr(eq + 1);
                const bool single = a[1] != '-';
                for (std::size_t k = 0; k < actions_.size(); ++k)
                    for (const std::string& f : actions_[k].flags) {
                        if (single && f == a.substr(0, 2)) t.matches.push_back({k, f, a.substr(2), false});
                        else if (starts_with(f, prefix)) t.matches.push_back({k, f, explicit_arg, sep});
                    }
                t.kind = Token::option;
            }
            if (t.matches.empty() && (negative_number(a) || a.find(' ') != std::string::npos))
                t.kind = Token::value;
        }
        tokens.push_back(std::move(t));
    }

    Args result;
    for (const Action& a : actions_) {
        if (a.help) continue;
        if (a.store_true) result.flags_[a.dest] = false;
        else result.values_[a.dest] = a.spec.default_text;
    }

    std::vector<bool> seen(actions_.size(), false);
    std::vector<std::string> extras;
    std::size_t next_positional = 0;
    auto positional_index = [&](std::size_t nth) -> std::optional<std::size_t> {
        std::size_t n = 0;
        for (std::size_t k = 0; k < actions_.size(); ++k)
            if (actions_[k].positional && n++ == nth) return k;
        return std::nullopt;
    };

    auto refuse = [&](const std::string& message) {
        err << usage() << "\n" << prog_ << ": error: " << message << "\n";
        status = 2;
    };

    // Which arguments a positional took, for the '--' below.
    std::vector<bool> to_positional(tokens.size(), false);

    try {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const Token& t = tokens[i];
            if (t.kind == Token::dashdash) {
                // argparse drops the '--' only where a positional's pattern
                // (-*A-*) takes it: straight after a positional's value, or
                // before the value of a positional still to fill. Otherwise
                // it is an unrecognized argument like any other ("a b --opt
                // x --" refuses "--"; everything after it is a value anyway).
                const bool taken = (i > 0 && to_positional[i - 1]) ||
                                   (positional_index(next_positional) && i + 1 < tokens.size());
                if (!taken) extras.push_back(args[i]);
                continue;
            }
            if (t.kind == Token::value) {
                if (const auto k = positional_index(next_positional)) {
                    ++next_positional;
                    seen[*k] = true;
                    to_positional[i] = true;
                    result.values_[actions_[*k].dest] = args[i];
                } else {
                    extras.push_back(args[i]);
                }
                continue;
            }
            if (t.matches.empty()) {   // an option this parser does not have
                extras.push_back(args[i]);
                continue;
            }
            if (t.matches.size() > 1) {
                std::string flags;
                for (std::size_t k = 0; k < t.matches.size(); ++k) flags += (k ? ", " : "") + t.matches[k].flag;
                throw ArgumentError{"ambiguous option: " + args[i] + " could match " + flags};
            }
            const Match& m = t.matches.front();
            const Action& a = actions_[m.action];
            seen[m.action] = true;
            if (a.store_true) {
                if (m.explicit_arg) {
                    // A long flag takes no argument at all. A short flag's tail is
                    // more short flags ("-hx" is -h, then an unknown -x), unless it
                    // is empty, came after '=' or starts with a dash.
                    const std::string& x = *m.explicit_arg;
                    const bool shortflag = m.flag.size() >= 2 && m.flag[1] != '-';
                    if (!shortflag || x.empty() || m.sep || x[0] == '-')
                        throw ArgumentError{"argument " + a.name() + ": ignored explicit argument " +
                                            pytext::repr(x)};
                }
                if (a.help) {
                    out << help();
                    status = 0;
                    return std::nullopt;
                }
                result.flags_[a.dest] = true;
                continue;
            }
            std::string value;
            if (m.explicit_arg) {
                value = *m.explicit_arg;
            } else if (i + 1 < tokens.size() && tokens[i + 1].kind == Token::value) {
                value = args[++i];
            } else {
                throw ArgumentError{"argument " + a.name() + ": expected one argument"};
            }
            if (a.spec.type == ArgSpec::Type::int_ && !pynum::py_int(value, 10))
                throw ArgumentError{"argument " + a.name() + ": invalid int value: " + pytext::repr(value)};
            if (a.spec.type == ArgSpec::Type::float_ && !pynum::py_float(value))
                throw ArgumentError{"argument " + a.name() + ": invalid float value: " + pytext::repr(value)};
            if (!a.spec.choices.empty()) {
                bool ok = false;
                for (const std::string& c : a.spec.choices) ok = ok || c == value;
                if (!ok) {
                    std::string choices;
                    for (std::size_t k = 0; k < a.spec.choices.size(); ++k)
                        choices += (k ? ", " : "") + pytext::repr(a.spec.choices[k]);
                    throw ArgumentError{"argument " + a.name() + ": invalid choice: " + pytext::repr(value) +
                                        " (choose from " + choices + ")"};
                }
            }
            result.values_[a.dest] = value;
        }

        std::vector<std::string> missing;
        for (std::size_t k = 0; k < actions_.size(); ++k)
            if (!seen[k] && actions_[k].spec.required) missing.push_back(actions_[k].name());
        if (!missing.empty()) {
            std::string names;
            for (std::size_t k = 0; k < missing.size(); ++k) names += (k ? ", " : "") + missing[k];
            throw ArgumentError{"the following arguments are required: " + names};
        }
        if (!extras.empty()) {
            std::string names;
            for (std::size_t k = 0; k < extras.size(); ++k) names += (k ? " " : "") + extras[k];
            throw ArgumentError{"unrecognized arguments: " + names};
        }
    } catch (const ArgumentError& e) {
        refuse(e.message);
        return std::nullopt;
    }
    status = 0;
    return result;
}

}  // namespace sstvae::qrss::cli
