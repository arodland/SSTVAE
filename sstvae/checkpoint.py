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
DEFAULT_FILE = "v1.pt"

# The ONNX artifacts exported from DEFAULT_FILE by scripts/export_onnx.py.
# `DEFAULT_REVISION` is the stem of the checkpoint they came from, so the
# two can never be bumped independently: change DEFAULT_FILE and the
# ONNX names follow automatically. Each artifact also carries its source
# checkpoint's sha256 in its ONNX metadata, so a mismatch is detectable
# rather than merely unlikely.
DEFAULT_REVISION = DEFAULT_FILE.rsplit(".", 1)[0]
PRECISIONS = ("fp32", "fp16", "int8")

# fp16 is the shipped default: measured identical to fp32 end to end
# (docs/onnx.md) at half the size. int8 is available but costs ~1 dB of
# effective SNR on the *encoder*, whose error every receiver pays for.
DEFAULT_PRECISION = "fp16"


def onnx_filename(part: str, precision: str = DEFAULT_PRECISION) -> str:
    """e.g. ("encoder", "fp16") -> "v1-encoder-fp16.onnx"."""
    if part not in ("encoder", "decoder"):
        raise ValueError(f"part must be 'encoder' or 'decoder', not {part!r}")
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {', '.join(PRECISIONS)}, not {precision!r}"
        )
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
    """
    if path is None:
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
        if f"-{part}-" in p.name:
            return str(p)
        # Named the other part: derive this one from it, so `--model
        # v1-encoder-fp16.onnx` still works for an operation that turns
        # out to need the decoder too.
        other = "decoder" if part == "encoder" else "encoder"
        if f"-{other}-" not in p.name:
            raise SystemExit(
                f"cannot tell which part {p.name} is -- expected a name like "
                f"v1-{part}-{precision}.onnx, or pass the directory instead"
            )
        sibling = p.with_name(p.name.replace(f"-{other}-", f"-{part}-"))
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
