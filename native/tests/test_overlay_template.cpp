// Template substitution: the three rules of docs/overlay-templates.md,
// the placeholder grammar, and the shipped templates themselves.
//
// The reference is `sstvae/overlay/template.py`, and the parity suite
// diffs this implementation against it on a corpus; what is here is the
// same set of cases stated against the *spec*, so that a rule can be
// wrong in both implementations at once and still fail. The shipped
// templates are read from the Python package directory (argv[1]) rather
// than from a copy, because a copy is a second source of what "Reply"
// says.

#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "check.hpp"
#include "overlay/model.hpp"
#include "overlay/template.hpp"

using namespace sstvae;
using overlay::Fields;

namespace {

std::string sub(const std::string& text, Fields fields = {}) {
    return overlay::substitute_text(text, fields);
}

Fields reply_fields() {
    Fields f;
    f.builtin = {{"theircall", "W1XYZ"}, {"mycall", "KC2G"}, {"snr", "12 dB"}};
    return f;
}

// --- rule 1: unknown placeholders stay literal ---------------------------

void test_builtins_are_replaced() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    check::equal(sub("de {mycall}", f), std::string("de KC2G"), "builtin substituted");
}

void test_unknown_placeholder_is_left_literally() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    // A typo in a template must show in the preview, not vanish on air.
    check::equal(sub("de {mycal}", f), std::string("de {mycal}"), "typo kept");
    check::equal(sub("{MYCALL}", f), std::string("{MYCALL}"), "case-sensitive");
    check::equal(sub("{my call}", f), std::string("{my call}"), "space in name");
}

void test_doubled_braces_are_literal() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    check::equal(sub("{{mycall}}", f), std::string("{mycall}"), "escaped placeholder");
    check::equal(sub("a {{ b }} c", f), std::string("a { b } c"), "escaped braces");
}

void test_unclosed_brace_is_literal() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    check::equal(sub("{mycall", f), std::string("{mycall"), "no closing brace");
    check::equal(sub("} {mycall}", f), std::string("} KC2G"), "stray closing brace");
    check::equal(sub("{{mycall}", f), std::string("{mycall}"), "escaped open, plain close");
}

// --- rule 2: a line whose placeholders are all empty is dropped ----------

void test_line_with_only_empty_placeholders_is_dropped_with_its_literal_text() {
    Fields f;
    f.builtin = {{"theircall", "W1XYZ"}, {"mycall", "KC2G"}};
    // The literal "SNR " goes with its placeholder: a label with nothing
    // after it is worse than no line.
    check::equal(sub("{theircall} de {mycall}\nSNR {snr}\n{field Comment}", f),
                 std::string("W1XYZ de KC2G"), "two empty lines dropped");
}

void test_line_without_placeholders_is_never_touched() {
    check::equal(sub("CQ CQ CQ\n\nde {mycall}"), std::string("CQ CQ CQ\n"),
                 "literal lines kept, including the empty one");
}

void test_line_with_one_filled_and_one_empty_placeholder_is_kept() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    // Partial text is never dropped; the empty hole simply contributes
    // nothing.
    check::equal(sub("{theircall} de {mycall}", f), std::string(" de KC2G"),
                 "line kept when any placeholder is filled");
}

void test_whitespace_only_value_counts_as_empty() {
    Fields f;
    f.custom = {{"Comment", "   "}};
    check::equal(sub("{field Comment}", f), std::string(""), "blank comment dropped");
    check::equal(sub("x\n{field Comment}", f), std::string("x"), "and only that line");
}

void test_missing_field_counts_as_empty() {
    check::equal(sub("SNR {snr}"), std::string(""), "no fields at all");
}

void test_all_lines_dropped_gives_empty_text() {
    check::equal(sub("{theircall}\n{snr}"), std::string(""), "empty, not a newline");
}

// --- custom fields ------------------------------------------------------

void test_custom_field_is_substituted_by_label() {
    Fields f;
    f.custom = {{"Comment", "TNX FER PIC"}};
    check::equal(sub("{field Comment}", f), std::string("TNX FER PIC"), "custom filled");
}

void test_label_may_contain_spaces_and_is_normalized() {
    Fields f;
    f.custom = {{"Their QTH", "Boston"}};
    check::equal(sub("QTH {field Their QTH}", f), std::string("QTH Boston"), "space in label");
    check::equal(sub("QTH {field  Their   QTH }", f), std::string("QTH Boston"),
                 "extra whitespace collapsed");
    check::equal(overlay::normalize_label("  a \t b  "), std::string("a b"), "normalize_label");
}

void test_same_label_twice_is_one_field() {
    overlay::Doc doc;
    overlay::TextItem t;
    t.text = "{field Comment}\n{field Comment} again";
    doc.items.emplace_back(t);
    const auto p = overlay::placeholders(doc);
    check::equal(p.custom.size(), std::size_t{1}, "one custom field");
    Fields f;
    f.custom = {{"Comment", "hi"}};
    check::equal(sub(t.text, f), std::string("hi\nhi again"), "both occurrences filled");
}

void test_malformed_field_declarations_are_unknown_placeholders() {
    Fields f;
    f.custom = {{"", "x"}};
    check::equal(sub("{field}", f), std::string("{field}"), "no label");
    check::equal(sub("{field }", f), std::string("{field }"), "blank label");
    check::equal(sub("{fieldx}", f), std::string("{fieldx}"), "no separator");
}

void test_a_builtin_name_as_a_label_is_a_custom_field_not_a_builtin() {
    Fields f;
    f.builtin = {{"mycall", "KC2G"}};
    f.custom = {{"mycall", "custom"}};
    check::equal(sub("{mycall} {field mycall}", f), std::string("KC2G custom"),
                 "the two namespaces do not collide");
}

// --- placeholders() ---------------------------------------------------

void test_placeholders_in_order_of_first_appearance_without_repeats() {
    overlay::Doc doc;
    overlay::TextItem a;
    a.text = "{snr} {mycall}\n{field Comment} {nonsense}";
    overlay::TextItem b;
    b.text = "{mycall} {field QTH} {field Comment}";
    doc.items.emplace_back(a);
    doc.items.emplace_back(overlay::ImageItem{});
    doc.items.emplace_back(b);
    const auto p = overlay::placeholders(doc);
    check::equal(p.builtin.size(), std::size_t{2}, "two builtins");
    check::equal(p.builtin[0], std::string("snr"), "first builtin");
    check::equal(p.builtin[1], std::string("mycall"), "second builtin");
    check::equal(p.custom.size(), std::size_t{2}, "two custom fields");
    check::equal(p.custom[0], std::string("Comment"), "first custom");
    check::equal(p.custom[1], std::string("QTH"), "second custom");
}

// --- rule 3 and the document ------------------------------------------

void test_substitute_returns_a_copy_and_leaves_images_alone() {
    overlay::Doc doc;
    doc.name = "Reply";
    overlay::TextItem t;
    t.text = "{theircall} de {mycall}";
    doc.items.emplace_back(t);
    doc.items.emplace_back(overlay::ImageItem{});

    const overlay::Doc out = overlay::substitute(doc, reply_fields());
    check::equal(std::get<overlay::TextItem>(doc.items[0]).text, t.text, "template untouched");
    check::equal(std::get<overlay::TextItem>(out.items[0]).text, std::string("W1XYZ de KC2G"),
                 "copy substituted");
    check::is_true(std::holds_alternative<overlay::ImageItem>(out.items[1]), "image kept");
    check::equal(out.name, std::string("Reply"), "name carried");
}

void test_name_round_trips_and_is_omitted_when_empty() {
    overlay::Doc doc;
    doc.name = "CQ";
    const std::string text = overlay::to_json(doc);
    check::is_true(text.find("\"name\"") != std::string::npos, "name written");
    check::equal(overlay::from_json(text).name, std::string("CQ"), "name read back");

    overlay::Doc plain;
    check::is_true(overlay::to_json(plain).find("\"name\"") == std::string::npos,
                   "unnamed document serializes as before");
    check::equal(overlay::from_json(overlay::to_json(plain)).name, std::string(""),
                 "nameless loads with an empty name");
}

void test_a_non_string_name_costs_the_name_not_the_document() {
    std::vector<overlay::Note> notes;
    const auto doc = overlay::from_json(R"({"version": 1, "name": 7, "items": []})", &notes);
    check::equal(doc.name, std::string(""), "bad name ignored");
    check::equal(notes.size(), std::size_t{1}, "and reported");
}

// --- format_snr -------------------------------------------------------

void test_format_snr() {
    check::equal(overlay::format_snr(12.4), std::string("12 dB"), "rounds down");
    check::equal(overlay::format_snr(12.5), std::string("12 dB"), "half to even");
    check::equal(overlay::format_snr(13.5), std::string("14 dB"), "half to even, up");
    check::equal(overlay::format_snr(-2.6), std::string("-3 dB"), "negative");
    check::equal(overlay::format_snr(-0.3), std::string("0 dB"), "no negative zero");
    check::equal(overlay::format_snr(std::nullopt), std::string(""), "nothing measured");
}

// --- the shipped templates --------------------------------------------

overlay::Doc load(const std::string& dir, const std::string& stem) {
    std::ifstream in(dir + "/" + stem + ".json");
    check::is_true(in.good(), "template file readable: " + stem);
    std::stringstream buf;
    buf << in.rdbuf();
    std::vector<overlay::Note> notes;
    const auto doc = overlay::from_json(buf.str(), &notes);
    check::is_true(notes.empty(), "template loads without notes: " + stem);
    return doc;
}

const std::string& first_text(const overlay::Doc& doc) {
    return std::get<overlay::TextItem>(doc.items.at(0)).text;
}

void test_shipped_templates(const std::string& dir) {
    const auto cq = load(dir, "cq");
    const auto reply = load(dir, "reply");
    const auto picture = load(dir, "reply-picture");

    check::equal(cq.name, std::string("CQ"), "cq name");
    check::equal(reply.name, std::string("Reply"), "reply name");
    check::equal(picture.name, std::string("Reply with picture"), "reply-picture name");

    // Every built-in carries a Comment line, so free-form text needs no
    // editor; only Reply asks for their call.
    for (const auto* doc : {&cq, &reply, &picture}) {
        const auto p = overlay::placeholders(*doc);
        check::is_true(p.custom == std::vector<std::string>{"Comment"},
                       doc->name + " has exactly a Comment field");
    }
    check::is_true(overlay::placeholders(cq).builtin == std::vector<std::string>{"mycall", "grid"},
                   "cq asks for mycall and grid");
    check::is_true(overlay::placeholders(reply).builtin ==
                       std::vector<std::string>{"theircall", "mycall", "snr"},
                   "reply asks for theircall, mycall, snr");

    // Reply with picture is Reply plus a last_rx inset, and nothing else.
    check::equal(first_text(picture), first_text(reply), "same text as reply");
    check::equal(picture.items.size(), std::size_t{2}, "one inset");
    check::equal(std::get<overlay::ImageItem>(picture.items[1]).source,
                 std::string(overlay::SOURCE_LAST_RX), "inset is the last reception");

    // What they say when filled in the ordinary way.
    check::equal(first_text(overlay::substitute(reply, reply_fields())),
                 std::string("W1XYZ de KC2G\nSNR 12 dB"), "reply, no comment");
    Fields cq_fields;
    cq_fields.builtin = {{"mycall", "KC2G"}};
    cq_fields.custom = {{"Comment", "QRZ?"}};
    check::equal(first_text(overlay::substitute(cq, cq_fields)),
                 std::string("CQ CQ CQ\nde KC2G\nQRZ?"), "cq with a comment, no grid");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_overlay_template <templates dir>\n");
        return 2;
    }
    try {
        test_builtins_are_replaced();
        test_unknown_placeholder_is_left_literally();
        test_doubled_braces_are_literal();
        test_unclosed_brace_is_literal();
        test_line_with_only_empty_placeholders_is_dropped_with_its_literal_text();
        test_line_without_placeholders_is_never_touched();
        test_line_with_one_filled_and_one_empty_placeholder_is_kept();
        test_whitespace_only_value_counts_as_empty();
        test_missing_field_counts_as_empty();
        test_all_lines_dropped_gives_empty_text();
        test_custom_field_is_substituted_by_label();
        test_label_may_contain_spaces_and_is_normalized();
        test_same_label_twice_is_one_field();
        test_malformed_field_declarations_are_unknown_placeholders();
        test_a_builtin_name_as_a_label_is_a_custom_field_not_a_builtin();
        test_placeholders_in_order_of_first_appearance_without_repeats();
        test_substitute_returns_a_copy_and_leaves_images_alone();
        test_name_round_trips_and_is_omitted_when_empty();
        test_a_non_string_name_costs_the_name_not_the_document();
        test_format_snr();
        test_shipped_templates(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("overlay_template");
}
