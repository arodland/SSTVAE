"""Locating the model checkpoint the CLI tools decode with.

The encoder/decoder network *is* the codec, so every tool needs a
checkpoint. Rather than make each user find one, the tools default to a
published checkpoint on the Hub and download it once (huggingface_hub
caches it under ~/.cache/huggingface, so it's a one-time cost).

`DEFAULT_FILE` is deliberately a fixed filename rather than a moving
"latest": the on-air format is not frozen, and a checkpoint that silently
changed under a user would break interoperability with every station
still running the old one. When a new public checkpoint is published,
bump `DEFAULT_FILE` here in the same change as the code that needs it.

Published checkpoints are therefore **immutable**, which is what lets a
cache hit be trusted outright rather than revalidated — see
`default_checkpoint`.
"""

from pathlib import Path

DEFAULT_REPO = "arodland/sstvae"
DEFAULT_FILE = "v4.pt"

# The ONNX artifacts exported from DEFAULT_FILE by scripts/export_onnx.py.
# `DEFAULT_REVISION` is the stem of the checkpoint they came from, so the
# two can never be bumped independently: change DEFAULT_FILE and the
# ONNX names follow automatically. Each artifact also carries its source
# checkpoint's sha256 in its ONNX metadata, so a mismatch is detectable
# rather than merely unlikely.
DEFAULT_REVISION = DEFAULT_FILE.rsplit(".", 1)[0]
PRECISIONS = ("fp32", "fp16", "int8")

# The decoder's gradient graph, for transmit-time latent optimization
# (`docs/latent-optimization.md`). Not a codec part: no receiver ever
# loads it, and a station that does not use the optimizer never fetches
# it.
#
# **It is fp32 whatever the codec's precision is**, because fp32 is the
# only version published -- the fp16 converter emits a graph
# onnxruntime will not load, and int8 is excluded on principle since
# differentiating `ConvInteger` is not well defined. The override is
# silent rather than an error because `--precision` is a statement about
# the *codec*, and refusing to optimize because someone picked int8 for
# their decoder would be answering a question they did not ask.
GRAD_PART = "decoder-grad"
GRAD_PRECISION = "fp32"
PARTS = ("encoder", "decoder", GRAD_PART)
# Revisions that actually have one. v1 and v2 predate the feature, so
# this is not hypothetical bookkeeping -- it is the difference between a
# clear message and a 404 for anyone pinning an older revision. Add to
# it when a revision ships the artifact.
GRAD_REVISIONS = frozenset({"v3", "v4"})

# fp16 is the shipped default: measured identical to fp32 end to end
# (docs/onnx.md) at half the size. int8 is available but costs ~1 dB of
# effective SNR on the *encoder*, whose error every receiver pays for.
DEFAULT_PRECISION = "fp16"


def onnx_filename(part: str, precision: str = DEFAULT_PRECISION) -> str:
    """e.g. ("encoder", "fp16") -> "v1-encoder-fp16.onnx".

    `decoder-grad` ignores `precision` and is always fp32 -- see
    `GRAD_PRECISION`.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {', '.join(PARTS)}, not {part!r}")
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {', '.join(PRECISIONS)}, not {precision!r}"
        )
    if part == GRAD_PART:
        precision = GRAD_PRECISION
    return f"{DEFAULT_REVISION}-{part}-{precision}.onnx"


def default_onnx(part: str, precision: str = DEFAULT_PRECISION) -> str:
    """Path to a published ONNX artifact, downloading it once if needed.

    Same immutability argument as `default_checkpoint`: the filename
    names a specific artifact, so a cache hit needs no revalidation.

    **One part at a time, on purpose.** No CLI needs both: encoding uses
    only the encoder, decoding and listening only the decoder. Fetching
    per part means a receive-only station downloads 9 MB rather than the
    21 MB pair.
    """
    return _fetch(onnx_filename(part, precision), "onnxruntime")


def resolve_onnx(part: str, path: str | None = None,
                 precision: str = DEFAULT_PRECISION) -> str:
    """Locate one ONNX part, given whatever the user passed to --model.

    The `.pt` checkpoint is one file and the ONNX codec is two, so
    `--model` cannot simply be "the model file" any more. It accepts:

      omitted            the published artifacts (fetched, cached)
      a directory        `*-{part}-{precision}.onnx` inside it
      a `.onnx` file     that part; the sibling part is derived from the
                         name by substitution, and only when it is
                         actually needed

    A `.pt` path never reaches here -- `codec.load_codec` sends those to
    the torch backend instead.

    `decoder-grad` is forced to fp32 (see `GRAD_PRECISION`), including
    when resolving out of a directory or off a sibling.
    """
    if part == GRAD_PART:
        precision = GRAD_PRECISION
    if path is None:
        if part == GRAD_PART and DEFAULT_REVISION not in GRAD_REVISIONS:
            # The gradient graph arrived with v3; earlier revisions were
            # published without one. Say so, rather than letting the
            # fetch fail with a 404 on a filename nobody recognises --
            # the operator has done nothing wrong and the fix is a flag.
            raise SystemExit(
                f"latent optimization needs the {GRAD_PART} artifact, which "
                f"was not published for {DEFAULT_REVISION} (it exists from "
                f"{sorted(GRAD_REVISIONS)[0]} onward).\n"
                f"Pass --model with a revision that has it, e.g.\n"
                f"  --model {sorted(GRAD_REVISIONS)[0]}-encoder-"
                f"{DEFAULT_PRECISION}.onnx\n"
                "or a directory of exported .onnx files."
            )
        return default_onnx(part, precision)

    p = Path(path)
    if p.is_dir():
        matches = sorted(p.glob(f"*-{part}-{precision}.onnx"))
        if not matches:
            available = sorted(q.name for q in p.glob("*.onnx"))
            raise SystemExit(
                f"no *-{part}-{precision}.onnx in {p}\n"
                + (f"found: {', '.join(available)}" if available
                   else "that directory holds no .onnx files at all")
            )
        if len(matches) > 1:
            raise SystemExit(
                f"ambiguous: {len(matches)} candidates for {part}/{precision} "
                f"in {p} ({', '.join(q.name for q in matches)})"
            )
        return str(matches[0])

    if p.suffix == ".onnx":
        # `-decoder-` is a prefix of `-decoder-grad-`, so a plain
        # substring test would hand back the gradient graph when the
        # decoder was asked for -- and it would load, and produce
        # nonsense, because its first output is the reconstruction.
        is_grad = f"-{GRAD_PART}-" in p.name
        if (f"-{part}-" in p.name) and (is_grad == (part == GRAD_PART)):
            return str(p)
        # Named a different part: derive this one from it, so `--model
        # v1-encoder-fp16.onnx` still works for an operation that turns
        # out to need the decoder too.
        other = GRAD_PART if is_grad else (
            "decoder" if part == "encoder" else "encoder")
        if f"-{other}-" not in p.name:
            raise SystemExit(
                f"cannot tell which part {p.name} is -- expected a name like "
                f"v1-{part}-{precision}.onnx, or pass the directory instead"
            )
        # Rebuild from the stem rather than substituting the part into
        # the given name: the sibling's *precision* need not match the
        # one we were handed. `decoder-grad` is only published at fp32,
        # so `--model v3-encoder-fp16.onnx` has to resolve to
        # `v3-decoder-grad-fp32.onnx`, not to an fp16 name that was
        # never exported.
        stem = p.name.split(f"-{other}-")[0]
        sibling = p.with_name(f"{stem}-{part}-{precision}.onnx")
        if not sibling.exists():
            raise SystemExit(
                f"need the {part} as well as the {other}, but {sibling.name} "
                f"is not next to {p.name}"
            )
        return str(sibling)

    raise SystemExit(
        f"--model {path}: expected a .pt checkpoint, a .onnx artifact, or a "
        "directory containing the exported .onnx files"
    )


def default_checkpoint() -> str:
    """Path to the published checkpoint, downloading it once if needed.

    The cache is consulted **without touching the network** first, and a
    hit is returned as-is. `DEFAULT_FILE` names a specific immutable
    checkpoint (see above), so once it is cached there is nothing to
    revalidate — whereas plain `hf_hub_download` issues a HEAD request on
    every call to check for a newer version. That costs a round trip on
    each run, fails needlessly when offline, and prints a "you are
    sending unauthenticated requests to the HF Hub" warning on any
    machine without Hub credentials. For a long-running GUI that resolves
    the checkpoint at startup, none of that buys anything.
    """
    return _fetch(DEFAULT_FILE, "torch")


def _any_cached_revision(filename: str) -> str | None:
    """The file under *any* cached revision of the repo, or None.

    `hf_hub_download(local_files_only=True)` resolves `refs/main` to one
    commit and looks only inside that snapshot, so a file cached before
    an unrelated commit stops resolving offline the moment anything else
    is downloaded. That is a real failure and not a hypothetical one:
    the codec fetches the encoder and decoder **separately, on demand**,
    so downloading one can strand the other on a machine that then goes
    offline -- a field laptop being exactly the case that matters.

    Falling back like this is safe *because* published filenames are
    immutable (see the module docstring). Any revision's `v1.pt` is byte
    for byte the same `v1.pt`; the commit it happens to be filed under
    carries no information.
    """
    try:
        import huggingface_hub

        cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    except Exception:
        return None  # no real hub installed (or a stand-in, as in tests)

    snapshots = cache / f"models--{DEFAULT_REPO.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        candidate = snapshot / filename
        if candidate.exists():
            return str(candidate)
    return None


def _fetch(filename: str, needed_for: str) -> str:
    """Cache-first fetch of one immutable published file.

    Shared by the `.pt` and `.onnx` paths because the argument for
    trusting a cache hit outright is the same for both: the filename
    names a specific artifact that will never change under us.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise SystemExit(
            f"huggingface_hub is needed to fetch {filename}.\n"
            "Install it (pip install huggingface_hub), or pass --model with a\n"
            "local .pt, .onnx, or directory of exported .onnx files."
        ) from e

    try:
        return hf_hub_download(DEFAULT_REPO, filename, local_files_only=True)
    except Exception:
        pass  # not under the current revision -- try the others, then fetch

    found = _any_cached_revision(filename)
    if found is not None:
        return found

    try:
        return hf_hub_download(DEFAULT_REPO, filename)
    except Exception as e:  # network down, Hub outage, offline machine...
        raise SystemExit(
            f"could not fetch {DEFAULT_REPO}/{filename} (needed by "
            f"{needed_for}): {e}\n"
            "If you're offline, pass --model with a local .pt, .onnx, or\n"
            "directory of exported .onnx files."
        ) from e


def resolve(path: str | None) -> str:
    """An explicit --model wins; otherwise fall back to the published one."""
    return path if path else default_checkpoint()
