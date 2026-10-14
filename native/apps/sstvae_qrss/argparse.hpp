// The part of Python's argparse the qrss_*.py CLIs use, parsing and refusing
// as argparse does, so `sstvae-qrss <sub>` accepts exactly the command lines
// its script accepts and refuses the rest with the same message (exit 2).
//
// What is reproduced, against CPython 3.13's argparse:
//
// - positionals (each exactly one value, or as the last one nargs='?': at
//   most one value, None without) and long options taking one value
//   (`--opt V` or `--opt=V`) or none (store_true); `-h`/`--help`;
// - every argument classified before any is consumed, as argparse does: an
//   exact option, `--opt=V`, a unique prefix of a long option (allow_abbrev;
//   "ambiguous option: --m could match --model, --mode" for several, raised
//   when that argument is reached), a negative number or anything containing
//   a space as a value, any other `-x` as an unknown option; everything after
//   the first `--` a value;
// - that first `--` dropped only where a positional takes it (next to a
//   positional's value): after the positionals are filled it is an
//   unrecognized argument, so "in out --callsign K --" refuses "--"; a
//   positional of nargs='?' still to fill takes it even as the last
//   argument (its pattern -*A?-* matches the "--" alone, and argparse
//   drops a "--" inside a positional's match), and is never a required
//   argument;
// - an option's value must be the next argument and must not look like an
//   option ("argument --model: expected one argument"); a store_true given
//   one is "ignored explicit argument 'x'";
// - type=int and type=float as int() and float() read text (pynum.hpp),
//   "invalid int value: 'x'"; choices checked after the type, "invalid
//   choice: 'Z' (choose from 'A', 'B', 'C')";
// - mutually exclusive groups of options (add_mutually_exclusive_group):
//   "argument --b: not allowed with argument --a" raised at the second, once
//   its value has passed type and choices (take_action); the usage line
//   shows a group whose options were added one after another as one
//   bracket, "[--a | --b]" ("(--a | --b)" when required, which a group of
//   one prints bare), and a required group none of whose options was given
//   is "one of the arguments --a --b is required";
// - after the arguments, "the following arguments are required: a, b,
//   --callsign" in the order the arguments were added, then a required
//   group's refusal, then "unrecognized arguments: ..." for unknown options
//   and surplus values;
// - a refusal is "usage: ..." and "<prog>: error: <message>" on stderr;
// - -h's single-dash forms: "-hx" is help (argparse reads x as more short
//   flags, and help exits before an unknown one is reported), "-h=x" and
//   "-h-x" are "ignored explicit argument". The scripts have no other
//   short option, so no other combination arises.
//
// Not reproduced: argparse wraps a long usage line at the terminal width
// and lays out --help with its own formatter (this prints the usage on one
// line, the description as written and the options in argparse's columns,
// wrapped at 78).

#ifndef SSTVAE_APPS_SSTVAE_QRSS_ARGPARSE_HPP
#define SSTVAE_APPS_SSTVAE_QRSS_ARGPARSE_HPP

#include <cstddef>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "qrss/pynum.hpp"

namespace sstvae::qrss::cli {

// One add_argument's keywords, set fluently:
//   ArgSpec().integer().otherwise("0").described("...")
struct ArgSpec {
    enum class Type { str, int_, float_ };
    Type type = Type::str;
    std::vector<std::string> choices;          // empty: any value
    bool required = false;                     // options (a positional is, unless nargs='?')
    std::optional<std::string> default_text;   // as text of `type`; nullopt is None
    std::string metavar;                       // default: DEST, or {a,b,c} with choices
    std::string help;

    ArgSpec& integer() { type = Type::int_; return *this; }                       // type=int
    ArgSpec& real() { type = Type::float_; return *this; }                        // type=float
    ArgSpec& one_of(std::vector<std::string> c) { choices = std::move(c); return *this; }
    ArgSpec& must() { required = true; return *this; }                            // required=True
    ArgSpec& otherwise(std::string d) { default_text = std::move(d); return *this; }   // default=
    ArgSpec& shown_as(std::string m) { metavar = std::move(m); return *this; }
    ArgSpec& described(std::string h) { help = std::move(h); return *this; }
};

// The parsed namespace. Values are the text given (or the default's text),
// already checked against the argument's type and choices.
class Args {
public:
    // The value, nullopt for None. `dest` as argparse names it ("codec_id").
    const std::optional<std::string>& text(const std::string& dest) const;
    // A store_true argument.
    bool on(const std::string& dest) const;
    // A type=int / type=float argument (nullopt for None).
    std::optional<pynum::PyInt> integer(const std::string& dest) const;
    std::optional<double> real(const std::string& dest) const;

private:
    friend class ArgParser;
    std::map<std::string, std::optional<std::string>> values_;
    std::map<std::string, bool> flags_;
};

class ArgParser {
public:
    ArgParser(std::string prog, std::string description);

    ArgParser& positional(const std::string& name, const std::string& help = {});
    // add_argument(name, nargs="?"): the value or None, "[name]" in the
    // usage line. Only as the last positional (std::logic_error for a
    // positional added after it).
    ArgParser& optional_positional(const std::string& name, const std::string& help = {});
    // A long option taking one value ("--codec-id"; dest "codec_id").
    ArgParser& option(const std::string& flag, ArgSpec spec = {});
    // action="store_true".
    ArgParser& flag(const std::string& flag, const std::string& help = {});
    // add_mutually_exclusive_group(required=...) holding these options
    // (already added; flags as given to option() / flag()). Its options are
    // the group's in the order they were added, as argparse's are. Refuses
    // (std::logic_error) what argparse refuses or this does not model: an
    // empty group, a positional, a required option, an unknown flag.
    ArgParser& exclusive(const std::vector<std::string>& flags, bool required = false);

    // parse_args(args): nullopt after printing help (status 0) or a usage
    // error (status 2) to `out` / `err`; `status` says which.
    std::optional<Args> parse(const std::vector<std::string>& args, std::ostream& out, std::ostream& err,
                              int& status) const;

    std::string usage() const;   // "usage: prog [-h] [--model MODEL] ... image output"
    std::string help() const;

private:
    struct Action {
        bool positional = false;
        bool store_true = false;
        bool help = false;
        std::vector<std::string> flags;   // "-h", "--help"; one long flag otherwise
        std::string dest;
        ArgSpec spec;
        std::string name() const;         // argparse's _get_action_name
        std::string metavar() const;
    };
    struct Group {
        std::vector<std::size_t> actions;   // indices into actions_, ascending
        bool required = false;
    };
    std::string prog_;
    std::string description_;
    std::vector<Action> actions_;   // add order, -h/--help first
    std::vector<Group> groups_;     // add order
};

}  // namespace sstvae::qrss::cli

#endif  // SSTVAE_APPS_SSTVAE_QRSS_ARGPARSE_HPP
