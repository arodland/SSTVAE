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

DEFAULT_REPO = "arodland/sstvae"
DEFAULT_FILE = "v1.pt"


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
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise SystemExit(
            "huggingface_hub is needed to fetch the default checkpoint.\n"
            "Install it (pip install huggingface_hub), or pass an explicit\n"
            "--model /path/to/checkpoint.pt"
        ) from e

    try:
        return hf_hub_download(DEFAULT_REPO, DEFAULT_FILE, local_files_only=True)
    except Exception:
        pass  # not cached yet (or the cache is unreadable) -- go and fetch it

    try:
        return hf_hub_download(DEFAULT_REPO, DEFAULT_FILE)
    except Exception as e:  # network down, Hub outage, offline machine...
        raise SystemExit(
            f"could not fetch the default checkpoint "
            f"({DEFAULT_REPO}/{DEFAULT_FILE}): {e}\n"
            "If you're offline, pass an explicit --model /path/to/checkpoint.pt"
        ) from e


def resolve(path: str | None) -> str:
    """An explicit --model wins; otherwise fall back to the published one."""
    return path if path else default_checkpoint()
