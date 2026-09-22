#!/usr/bin/env python3
"""Generates overlap-glyph.ttf: a one-glyph TrueType font whose only
glyph (U+E000, a private-use codepoint) is two same-direction
overlapping squares -- the general shape of the Android bug this is a
fixture for (native/core/overlay/render.cpp, "outlined text draws a
stray line through a glyph's self-intersection"). Regenerate with:

    python3 native/tests/fixtures/gen_overlap_glyph_font.py

Needs fontTools (`pip install fonttools`); not a build-time dependency,
since the .ttf it writes is committed.
"""
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

OUT = Path(__file__).parent / "overlap-glyph.ttf"


def main():
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "overlap"])
    fb.setupCharacterMap({0xE000: "overlap"})

    pen = TTGlyphPen(None)
    # Two squares sharing a 200-unit-wide vertical strip, both wound
    # clockwise (TrueType's convention for a solid, not a hole) -- two
    # ordinary same-direction contours whose union is what a real
    # glyph's self-intersecting design looks like. A correct renderer
    # (nonzero winding, as every font rasterizer uses) fills the overlap
    # solid; Qt's QPainterPath default (even-odd) punches it into a hole
    # and, worse for a stroked outline, treats it as "outside" and lets
    # the seam edges -- the parts of each square's boundary that fall
    # inside the other square -- paint as a stray line through the ink.
    for x0, x1 in ((100, 600), (400, 900)):
        pts = [(x0, 100), (x1, 100), (x1, 600), (x0, 600)]
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    glyph = pen.glyph()

    notdef_pen = TTGlyphPen(None)
    notdef_pen.moveTo((0, 0))
    notdef_pen.lineTo((0, 700))
    notdef_pen.lineTo((700, 700))
    notdef_pen.lineTo((700, 0))
    notdef_pen.closePath()

    fb.setupGlyf({".notdef": notdef_pen.glyph(), "overlap": glyph})
    fb.setupHorizontalMetrics({".notdef": (700, 0), "overlap": (1000, 0)})
    fb.setupHorizontalHeader(ascent=1000, descent=-200)
    fb.setupNameTable({"familyName": "SSTVAE Overlap Test", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
