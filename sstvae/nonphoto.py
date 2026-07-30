"""Procedural non-photographic images at IMG_W x IMG_H.

The content classes are the ones operators actually send that COCO does
not contain (docs/todo.md "Non-photographic content"): test cards,
colour bars, callsign ID cards, rendered text blocks, line art,
gradients and plotted charts. Everything is drawn with PIL and
matplotlib, so the images carry no licence and can be generated in
unlimited quantity — the same code serves the evaluation set
(`scripts/gen_nonphoto.py` / `scripts/eval_nonphoto.py`) and training
(`data.NonPhotoDataset`).

Determinism: `generate(cls, i, salt)` is identical across runs and
machines for the same arguments. Seeding goes through blake2s, not
Python's `hash()` — string hashing is salted per process, which would
make "deterministic" silently false. The `salt` is what keeps the
splits disjoint: "eval" for the measured evaluation set, "train" and
"val" for training, so no training image is ever an evaluation image.

matplotlib is imported lazily inside `gen_chart`, so every other class
works without it (it is a `train`/`listen` extra, not a base dep).

Torch-free on purpose, like `images.py` — the eval scripts run against
the ONNX codec with no torch installed.
"""

import hashlib

import numpy as np
from PIL import Image, ImageDraw

from .images import IMG_H, IMG_W, font

# Saturated primaries plus black/white: the palette of test cards and
# ID cards, and exactly the colours a photograph-trained model rarely
# sees at full saturation over a large flat area.
PALETTE = [
    (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
    (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
]

WORDS = [
    "SSTVAE", "TESTING", "HELLO", "73 DE", "CQ CQ CQ", "RST 599",
    "GREETINGS", "QSL VIA BURO", "HF IMAGE", "OFDM", "GOOD DX",
    "THANKS FOR QSO",
]


def _rng(cls: str, i: int, salt: str) -> np.random.Generator:
    digest = hashlib.blake2s(f"{salt}:{cls}:{i}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _canvas(color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (IMG_W, IMG_H), color)


def gen_testcard(r: np.random.Generator) -> Image.Image:
    """Colour bars over a grid, centre circle, a word in the middle."""
    img = _canvas((int(r.integers(0, 64)),) * 3)
    d = ImageDraw.Draw(img)
    step = int(r.integers(32, 56))
    for x in range(0, IMG_W, step):
        d.line([(x, 0), (x, IMG_H)], fill=(255, 255, 255), width=1)
    for y in range(0, IMG_H, step):
        d.line([(0, y), (IMG_W, y)], fill=(255, 255, 255), width=1)
    bar_h = IMG_H // 3
    y0 = (IMG_H - bar_h) // 2
    w = IMG_W / len(PALETTE)
    order = r.permutation(len(PALETTE))
    for k, ci in enumerate(order):
        d.rectangle([k * w, y0, (k + 1) * w, y0 + bar_h], fill=PALETTE[ci])
    cx, cy, rad = IMG_W // 2, IMG_H // 2, int(r.integers(120, 200))
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
              outline=(255, 255, 255), width=6)
    f = font(int(r.integers(28, 48)))
    word = WORDS[int(r.integers(0, len(WORDS)))]
    d.text((cx, cy), word, fill=(255, 255, 255), font=f, anchor="mm")
    return img


def gen_gradient(r: np.random.Generator) -> Image.Image:
    """Smooth ramps: linear, radial, or corner-blended colour fields."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    xx /= IMG_W - 1
    yy /= IMG_H - 1
    kind = int(r.integers(0, 3))
    if kind == 0:  # linear ramp at a random angle, two random colours
        th = r.uniform(0, np.pi)
        t = xx * np.cos(th) + yy * np.sin(th)
        t = (t - t.min()) / (t.max() - t.min())
    elif kind == 1:  # radial from a random centre
        cx, cy = r.uniform(0.2, 0.8), r.uniform(0.2, 0.8)
        t = np.hypot(xx - cx, yy - cy)
        t /= t.max()
    else:  # bilinear blend of four corner colours
        t = None
    if t is not None:
        c0, c1 = r.integers(0, 256, 3), r.integers(0, 256, 3)
        arr = (c0[None, None] * (1 - t[..., None])
               + c1[None, None] * t[..., None])
    else:
        corners = r.integers(0, 256, (4, 3)).astype(np.float64)
        arr = (corners[0] * ((1 - xx) * (1 - yy))[..., None]
               + corners[1] * (xx * (1 - yy))[..., None]
               + corners[2] * ((1 - xx) * yy)[..., None]
               + corners[3] * (xx * yy)[..., None])
    return Image.fromarray(arr.round().astype(np.uint8))


def gen_text(r: np.random.Generator) -> Image.Image:
    """A block of rendered text: dark on light or light on dark."""
    dark_bg = bool(r.integers(0, 2))
    bg = (int(r.integers(0, 48)),) * 3 if dark_bg else (int(r.integers(208, 256)),) * 3
    fg = (255, 255, 255) if dark_bg else (0, 0, 0)
    img = _canvas(bg)
    d = ImageDraw.Draw(img)
    size = int(r.integers(18, 40))
    f = font(size)
    y = int(r.integers(8, 40))
    while y < IMG_H - size:
        n = int(r.integers(2, 5))
        line = " ".join(WORDS[int(r.integers(0, len(WORDS)))] for _ in range(n))
        d.text((int(r.integers(8, 60)), y), line, fill=fg, font=f)
        y += int(size * r.uniform(1.3, 1.9))
    return img


def gen_callsign(r: np.random.Generator) -> Image.Image:
    """SSTV-style ID card: colour blocks, huge callsign, a status line."""
    bg = PALETTE[int(r.integers(0, len(PALETTE)))]
    img = _canvas(bg)
    d = ImageDraw.Draw(img)
    for _ in range(int(r.integers(2, 5))):
        c = PALETTE[int(r.integers(0, len(PALETTE)))]
        x0, y0 = r.integers(0, IMG_W // 2), r.integers(0, IMG_H // 2)
        x1 = x0 + int(r.integers(IMG_W // 4, IMG_W))
        y1 = y0 + int(r.integers(IMG_H // 4, IMG_H))
        d.rectangle([int(x0), int(y0), x1, y1], fill=c)
    call = "".join([
        "KNWAV"[int(r.integers(0, 5))],
        str(int(r.integers(0, 10))),
        *(chr(int(r.integers(65, 91))) for _ in range(int(r.integers(2, 4)))),
    ])
    fg = (0, 0, 0) if bool(r.integers(0, 2)) else (255, 255, 255)
    outline = (255, 255, 255) if fg == (0, 0, 0) else (0, 0, 0)
    f = font(int(r.integers(90, 150)))
    d.text((IMG_W // 2, IMG_H // 2), call, fill=fg, font=f, anchor="mm",
           stroke_width=4, stroke_fill=outline)
    f2 = font(int(r.integers(28, 44)))
    word = WORDS[int(r.integers(0, len(WORDS)))]
    d.text((IMG_W // 2, IMG_H - 48), word, fill=fg, font=f2, anchor="mm",
           stroke_width=2, stroke_fill=outline)
    return img


def gen_lineart(r: np.random.Generator) -> Image.Image:
    """Black-on-white shapes and polylines, hard edges everywhere."""
    img = _canvas()
    d = ImageDraw.Draw(img)
    for _ in range(int(r.integers(8, 20))):
        kind = int(r.integers(0, 3))
        w = int(r.integers(2, 7))
        if kind == 0:
            pts = [tuple(map(int, (r.integers(0, IMG_W), r.integers(0, IMG_H))))
                   for _ in range(int(r.integers(2, 6)))]
            d.line(pts, fill=(0, 0, 0), width=w, joint="curve")
        elif kind == 1:
            x0, y0 = int(r.integers(0, IMG_W - 60)), int(r.integers(0, IMG_H - 60))
            x1, y1 = x0 + int(r.integers(30, 200)), y0 + int(r.integers(30, 200))
            d.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), width=w)
        else:
            x0, y0 = int(r.integers(0, IMG_W - 60)), int(r.integers(0, IMG_H - 60))
            x1, y1 = x0 + int(r.integers(30, 250)), y0 + int(r.integers(30, 250))
            fill = (0, 0, 0) if r.integers(0, 4) == 0 else None
            d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=w, fill=fill)
    return img


def gen_chart(r: np.random.Generator) -> Image.Image:
    """A matplotlib figure rendered at exactly IMG_W x IMG_H."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    kind = int(r.integers(0, 3))
    n = int(r.integers(8, 30))
    if kind == 0:
        for _ in range(int(r.integers(1, 4))):
            ax.plot(np.cumsum(r.normal(size=n)), marker="o")
    elif kind == 1:
        ax.bar(np.arange(n), r.uniform(0.1, 1.0, n))
    else:
        ax.scatter(r.uniform(0, 1, 5 * n), r.uniform(0, 1, 5 * n),
                   c=r.uniform(0, 1, 5 * n), cmap="viridis")
    ax.set_title(WORDS[int(r.integers(0, len(WORDS)))])
    ax.grid(bool(r.integers(0, 2)))
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    img = Image.fromarray(buf.copy())
    if img.size != (IMG_W, IMG_H):
        img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
    return img


GENERATORS = {
    "testcard": gen_testcard,
    "gradient": gen_gradient,
    "text": gen_text,
    "callsign": gen_callsign,
    "lineart": gen_lineart,
    "chart": gen_chart,
}
CLASSES = tuple(sorted(GENERATORS))


def generate(cls: str, i: int, salt: str = "eval") -> Image.Image:
    """Image `i` of `cls` under `salt`: IMG_W x IMG_H RGB, deterministic."""
    img = GENERATORS[cls](_rng(cls, i, salt))
    assert img.size == (IMG_W, IMG_H)
    return img


def generate_index(i: int, salt: str = "train") -> Image.Image:
    """Image `i` of a flat mixture that cycles through the classes."""
    return generate(CLASSES[i % len(CLASSES)], i // len(CLASSES), salt)
