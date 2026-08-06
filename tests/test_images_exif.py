"""EXIF orientation on the Python side of the send path.

The C++ counterpart is `native/tests/test_images.cpp`, and the two are
held to the same result by `tests/test_native_parity.py -k exif`. What
is checked here is the thing neither of those can: that the *framing*
sees the upright picture. Orientation 6 turns a 480x640 portrait file
into a 640x480 landscape one, and `fit_image` computes its cover-crop
from that geometry -- so getting this wrong does not merely rotate the
transmission, it centre-crops a sideways picture and sends a strip out
of the middle of it. That is much worse than a rotation and looks like
a framing bug rather than a metadata one.
"""

import numpy as np
import pytest
from PIL import Image

from sstvae.images import IMG_H, IMG_W, fit_image, load_image, open_image

# The eight orientations, as the transform each one *undoes*, expressed
# with Pillow's primitives. Independent of `exif_transpose`'s own table.
_EXPECTED = {
    1: lambda im: im,
    2: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    3: lambda im: im.transpose(Image.ROTATE_180),
    4: lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    5: lambda im: im.transpose(Image.TRANSPOSE),
    6: lambda im: im.transpose(Image.ROTATE_270),
    7: lambda im: im.transpose(Image.TRANSVERSE),
    8: lambda im: im.transpose(Image.ROTATE_90),
}


def _tagged(tmp_path, orientation, size=(61, 37), fmt="jpeg"):
    """A picture with an orientation tag, saved losslessly by default.

    PNG rather than JPEG wherever the comparison is exact -- Pillow
    writes the tag into a PNG `eXIf` chunk and reads it back, so the
    transform can be checked on the pixels themselves rather than
    through a lossy codec.
    """
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    arr = np.stack([(x * 4) % 256, (y * 6) % 256, (x * 2 + y * 3) % 256], -1)
    img = Image.fromarray(arr.astype(np.uint8))
    exif = Image.Exif()
    exif[0x0112] = orientation
    path = tmp_path / f"probe_{orientation}.{'png' if fmt == 'png' else 'jpg'}"
    img.save(path, exif=exif, quality=95)
    return img, path


@pytest.mark.parametrize("orientation", range(1, 9))
def test_open_image_applies_the_tag(tmp_path, orientation):
    src, path = _tagged(tmp_path, orientation, fmt="png")

    got = np.array(open_image(path).convert("RGB"))
    want = np.array(_EXPECTED[orientation](src).convert("RGB"))

    assert np.array_equal(got, want)


def test_untagged_files_are_untouched(tmp_path):
    """The common case, and the one a bug here would break for everyone."""
    y, x = np.mgrid[0:37, 0:61]
    arr = np.stack([(x * 4) % 256, (y * 6) % 256, (x + y) % 256], -1).astype(np.uint8)
    path = tmp_path / "plain.png"
    Image.fromarray(arr).save(path)

    assert np.array_equal(np.array(open_image(path).convert("RGB")), arr)


def test_the_eight_transforms_are_distinguishable(tmp_path):
    """Asserts the probe's fitness, not the code: on a picture symmetric
    enough for two cases to coincide, a swapped pair would pass."""
    src, _ = _tagged(tmp_path, 1, fmt="png")
    seen = [np.array(_EXPECTED[o](src).convert("RGB")) for o in range(1, 9)]
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            assert a.shape != b.shape or not np.array_equal(a, b)


@pytest.mark.parametrize("orientation", [1, 6])
def test_framing_sees_the_upright_picture(tmp_path, orientation):
    """The consequence that matters: a portrait file must be framed as
    the portrait picture it is, not as the landscape file it is stored
    as.

    A 480x640 portrait tagged 6 is stored 640x480. Framed without the
    rotation it is already the target shape and passes through
    untouched; framed correctly it is 3:4, and covering a 4:3 target
    means scaling up and cropping top and bottom. Those two results have
    almost nothing in common, which is what makes this a sharp check
    rather than a rotation check.
    """
    # Stored landscape, meant to be seen portrait.
    y, x = np.mgrid[0:IMG_H, 0:IMG_W]
    arr = np.stack([(x % 256), (y % 256), ((x + y) % 256)], -1).astype(np.uint8)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = orientation
    path = tmp_path / "portrait.png"
    img.save(path, exif=exif)

    opened = open_image(path)
    if orientation == 6:
        assert opened.size == (IMG_H, IMG_W), "stored landscape, seen portrait"
    else:
        assert opened.size == (IMG_W, IMG_H)

    fitted = fit_image(opened)
    assert fitted.size == (IMG_W, IMG_H)

    # And the whole path agrees with itself.
    got = load_image(path)
    assert got.shape == (3, IMG_H, IMG_W)
    assert np.allclose(got, np.array(fitted, dtype=np.float32).transpose(2, 0, 1) / 255.0)

    if orientation == 6:
        # The untouched-landscape result is what the bug would produce.
        # It must not be what we produce.
        assert not np.array_equal(np.array(fitted), arr)


def test_an_enormous_file_is_refused(tmp_path, monkeypatch):
    """The limit is patched rather than met.

    Writing a gigabyte to prove a comparison would be slow and would
    need a sparse file to be honest about disk use; what is under test
    is the comparison and the refusal, and neither cares what the
    constant is. The constant itself is checked separately, against the
    C++ one, in `tests/test_native_parity.py` -- which is the check that
    matters, since a limit the two implementations disagreed about would
    mean one of them sending a file the other refused.
    """
    from sstvae import images as images_mod

    path = tmp_path / "big.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
    monkeypatch.setattr(images_mod, "MAX_FILE_BYTES", 4)

    with pytest.raises(ValueError, match="limit"):
        images_mod.open_image(path)

    # And at exactly the limit it opens, so the boundary is inclusive on
    # the same side as the C++.
    monkeypatch.setattr(images_mod, "MAX_FILE_BYTES", path.stat().st_size)
    assert images_mod.open_image(path).size == (8, 8)


def test_an_oversized_overlay_inset_draws_nothing(tmp_path, monkeypatch):
    """The refusal must not take the whole composition down with it.

    `_resolve_source` returns None for every other unusable source, and
    a size refusal is one more of those -- an operator mid-transmission
    would rather lose a decorative inset than the picture.
    """
    from importlib import import_module

    from sstvae import images as images_mod

    # `sstvae.overlay` re-exports a *function* named `render`, which
    # shadows the submodule of the same name on the package.
    render_mod = import_module("sstvae.overlay.render")

    path = tmp_path / "inset.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
    monkeypatch.setattr(images_mod, "MAX_FILE_BYTES", 4)

    assert render_mod._resolve_source(str(path), None) is None


def test_corrupt_metadata_does_not_break_the_load(tmp_path):
    """A truncated or nonsense tag costs a rotation, never the picture --
    the same rule `images::exif_orientation` follows in C++."""
    y, x = np.mgrid[0:37, 0:61]
    arr = np.stack([(x * 4) % 256, (y * 6) % 256, (x + y) % 256], -1).astype(np.uint8)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = 99  # outside 1..8
    path = tmp_path / "bogus.png"
    img.save(path, exif=exif)

    assert np.array_equal(np.array(open_image(path).convert("RGB")), arr)
