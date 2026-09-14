// The template catalog: reading a folder of overlay documents, never
// failing on a bad one, plus the slugify helper "Save as template" uses
// to pick a filename.
//
// argv[1] is `sstvae/overlay/templates/`, the built-ins' real home --
// the same directory `test_overlay_template.cpp` reads directly, so
// there is one source of what ships, not a copy pinned to this file.

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "overlay/template_catalog.hpp"

using namespace sstvae;
namespace fs = std::filesystem;

namespace {

void test_builtins_load_in_the_fixed_order(const std::string& dir) {
    const auto docs = overlay::load_builtin_templates(dir);
    check::equal(docs.size(), std::size_t{3}, "all three built-ins found");
    if (docs.size() == 3) {
        check::equal(docs[0].name, std::string("CQ"), "cq first");
        check::equal(docs[1].name, std::string("Reply"), "reply second");
        check::equal(docs[2].name, std::string("Reply with picture"), "reply-picture third");
    }
}

void test_a_missing_directory_yields_no_builtins_not_a_crash() {
    const auto docs = overlay::load_builtin_templates("/nonexistent/does-not-exist");
    check::is_true(docs.empty(), "missing dir -> empty, not a crash");
}

void test_load_templates_sorts_by_filename_and_skips_bad_ones() {
    fs::path tmp = fs::temp_directory_path() / "sstvae_test_templates";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    {
        std::ofstream(tmp / "b.json") << overlay::to_json([] {
            overlay::Doc d;
            d.name = "B";
            return d;
        }());
        std::ofstream(tmp / "a.json") << overlay::to_json([] {
            overlay::Doc d;
            d.name = "A";
            return d;
        }());
        std::ofstream(tmp / "broken.json") << "{ not json";
        std::ofstream(tmp / "not-a-template.txt") << "ignored, wrong extension";
    }

    const auto loaded = overlay::load_templates(tmp);
    check::equal(loaded.size(), std::size_t{2}, "the two good files, not the broken one or the .txt");
    if (loaded.size() == 2) {
        check::equal(loaded[0].doc.name, std::string("A"), "sorted by filename: a.json first");
        check::equal(loaded[1].doc.name, std::string("B"), "then b.json");
        check::equal(loaded[0].path.filename().string(), std::string("a.json"), "path recorded");
    }

    fs::remove_all(tmp, ec);
}

void test_load_templates_on_a_missing_directory_is_empty() {
    const auto loaded = overlay::load_templates("/nonexistent/does-not-exist");
    check::is_true(loaded.empty(), "missing dir -> empty list");
}

void test_slugify() {
    check::equal(overlay::slugify("Reply with picture"), std::string("reply-with-picture"),
                 "spaces become one dash");
    check::equal(overlay::slugify("  QRZ?!  "), std::string("qrz"), "punctuation trimmed and dropped");
    check::equal(overlay::slugify("CQ---CQ"), std::string("cq-cq"), "runs of punctuation collapse");
    check::equal(overlay::slugify(""), std::string("template"), "empty name falls back");
    check::equal(overlay::slugify("!!!"), std::string("template"), "all-punctuation falls back too");
    check::equal(overlay::slugify("W1AW"), std::string("w1aw"), "lower-cased");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_template_catalog <templates dir>\n");
        return 2;
    }
    try {
        test_builtins_load_in_the_fixed_order(argv[1]);
        test_a_missing_directory_yields_no_builtins_not_a_crash();
        test_load_templates_sorts_by_filename_and_skips_bad_ones();
        test_load_templates_on_a_missing_directory_is_empty();
        test_slugify();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("template_catalog");
}
