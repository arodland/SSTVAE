"""Resolving the published checkpoint.

The behaviour under test is that an already-downloaded checkpoint is
returned without any network access. `DEFAULT_FILE` names a specific,
immutable checkpoint, so there is nothing to revalidate — and the
revalidation `hf_hub_download` does by default costs a HEAD request on
every run, breaks when offline, and warns about unauthenticated requests
on machines with no Hub credentials.
"""

import sys
import types

import pytest

from sstvae import checkpoint


class FakeHub:
    """Stands in for `huggingface_hub`, recording how it was called."""

    def __init__(self, cached=None, remote=None):
        self.cached = cached
        self.remote = remote
        self.calls = []

    def hf_hub_download(self, repo, filename, local_files_only=False, **kw):
        self.calls.append({"filename": filename, "local_files_only": local_files_only})
        if local_files_only:
            if self.cached is None:
                raise FileNotFoundError(f"{filename} is not in the cache")
            return self.cached
        if self.remote is None:
            raise ConnectionError("no route to huggingface.co")
        return self.remote

    @property
    def network_calls(self):
        return [c for c in self.calls if not c["local_files_only"]]


@pytest.fixture
def fake_hub(monkeypatch):
    def install(cached=None, remote=None):
        hub = FakeHub(cached=cached, remote=remote)
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = hub.hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return hub

    return install


def test_cached_checkpoint_is_returned_without_touching_the_network(fake_hub):
    hub = fake_hub(cached="/cache/v1.pt", remote="/downloaded/v1.pt")

    assert checkpoint.default_checkpoint() == "/cache/v1.pt"
    assert hub.network_calls == [], (
        "a cached checkpoint must not trigger a Hub request -- that is what "
        "produces the unauthenticated-requests warning on every run"
    )


def test_the_cache_is_consulted_before_the_network(fake_hub):
    hub = fake_hub(cached="/cache/v1.pt", remote="/downloaded/v1.pt")
    checkpoint.default_checkpoint()
    assert hub.calls[0]["local_files_only"] is True


def test_an_uncached_checkpoint_is_downloaded(fake_hub):
    hub = fake_hub(cached=None, remote="/downloaded/v1.pt")

    assert checkpoint.default_checkpoint() == "/downloaded/v1.pt"
    assert len(hub.network_calls) == 1


def test_uncached_and_offline_explains_the_way_out(fake_hub):
    fake_hub(cached=None, remote=None)

    with pytest.raises(SystemExit) as excinfo:
        checkpoint.default_checkpoint()
    assert "--model" in str(excinfo.value), "should point at the manual override"


def test_cached_checkpoint_survives_an_unreachable_hub(fake_hub):
    """The offline case that matters: the file is already here."""
    hub = fake_hub(cached="/cache/v1.pt", remote=None)
    assert checkpoint.default_checkpoint() == "/cache/v1.pt"
    assert hub.network_calls == []


def test_a_file_cached_under_another_revision_is_still_found(fake_hub, tmp_path, monkeypatch):
    """The offline case that a new upstream commit used to break.

    `local_files_only` resolves `refs/main` and looks in that snapshot
    only, so an unrelated commit strands files cached earlier. Published
    filenames are immutable, so any revision's copy will do -- and the
    codec fetching encoder and decoder separately makes this reachable
    rather than theoretical.
    """
    hub = fake_hub(cached=None, remote=None)  # miss under refs/main, offline

    snapshots = tmp_path / f"models--{checkpoint.DEFAULT_REPO.replace('/', '--')}" / "snapshots"
    (snapshots / "oldcommit").mkdir(parents=True)
    stranded = snapshots / "oldcommit" / checkpoint.DEFAULT_FILE
    stranded.write_bytes(b"weights")

    fake_constants = types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path))
    monkeypatch.setattr(sys.modules["huggingface_hub"], "constants",
                        fake_constants, raising=False)

    assert checkpoint.default_checkpoint() == str(stranded)
    assert hub.network_calls == [], "should not need the network for this"


def test_an_explicit_path_short_circuits_everything(fake_hub):
    hub = fake_hub(cached="/cache/v1.pt", remote="/downloaded/v1.pt")
    assert checkpoint.resolve("/my/own.pt") == "/my/own.pt"
    assert hub.calls == []


# --- the decoder-gradient artifact ---------------------------------------
#
# Transmit-time latent optimization (docs/latent-optimization.md) adds a
# third part with two properties nothing else has: it is published at
# fp32 only, and its filename *contains* the decoder's. Both are easy to
# get wrong in a way that still loads a graph and produces a picture.


@pytest.mark.parametrize("asked", checkpoint.PRECISIONS)
def test_gradient_artifact_is_fp32_whatever_precision_is_asked(asked):
    """`--precision` is a statement about the codec, not about this.

    fp16 is unpublished (the converter emits a graph onnxruntime will
    not load) and int8 is excluded on principle, so honouring the
    caller here would name a file that does not exist.
    """
    name = checkpoint.onnx_filename(checkpoint.GRAD_PART, asked)
    assert name.endswith(f"-{checkpoint.GRAD_PART}-fp32.onnx")


def test_gradient_artifact_is_not_mistaken_for_the_decoder(tmp_path):
    """`-decoder-` is a substring of `-decoder-grad-`.

    A plain containment test hands back the gradient graph when the
    decoder was asked for. It loads, and its first output *is* a
    reconstruction, so the mistake survives all the way to a wrong
    picture.
    """
    for n in ("v3-decoder-fp16.onnx", "v3-decoder-grad-fp32.onnx",
              "v3-encoder-fp16.onnx"):
        (tmp_path / n).write_bytes(b"")
    grad = str(tmp_path / "v3-decoder-grad-fp32.onnx")

    assert checkpoint.resolve_onnx("decoder", grad, "fp16") == \
        str(tmp_path / "v3-decoder-fp16.onnx")
    assert checkpoint.resolve_onnx(checkpoint.GRAD_PART, grad) == grad
    # ...and from a directory, where the globs must not collide either.
    assert checkpoint.resolve_onnx("decoder", str(tmp_path), "fp16") == \
        str(tmp_path / "v3-decoder-fp16.onnx")
    assert checkpoint.resolve_onnx(checkpoint.GRAD_PART, str(tmp_path)) == grad


def test_gradient_sibling_uses_its_own_precision_not_the_given_one(tmp_path):
    """Deriving the sibling must rebuild the name, not substitute into it.

    From `v3-encoder-fp16.onnx` the gradient sibling is *fp32*; a
    substitution would look for an fp16 name that was never exported.
    """
    for n in ("v3-encoder-fp16.onnx", "v3-decoder-grad-fp32.onnx"):
        (tmp_path / n).write_bytes(b"")
    got = checkpoint.resolve_onnx(
        checkpoint.GRAD_PART, str(tmp_path / "v3-encoder-fp16.onnx"), "fp16")
    assert got == str(tmp_path / "v3-decoder-grad-fp32.onnx")


def test_a_revision_without_a_gradient_artifact_says_so(monkeypatch):
    """v1 and v2 predate the feature and the default is still v2.

    The failure to avoid is a 404 on a filename the operator has never
    seen, for a flag they used correctly.
    """
    monkeypatch.setattr(checkpoint, "DEFAULT_REVISION", "v2")
    with pytest.raises(SystemExit) as e:
        checkpoint.resolve_onnx(checkpoint.GRAD_PART, None)
    msg = str(e.value)
    assert "v2" in msg and "v3" in msg and checkpoint.GRAD_PART in msg


def test_the_current_revision_ships_a_gradient_artifact():
    """`GRAD_REVISIONS` must contain `DEFAULT_REVISION`.

    These are two hand-maintained lists that have to be bumped together,
    and the failure when they aren't is silent in the worst direction:
    the optimizer's gradient fetch is refused on the *current* codec, so
    a station that never optimizes sees nothing wrong and one that does
    gets a message about an unpublished artifact that was in fact
    published. Caught for real when v4 was published — the Python side
    was updated and the C++ copy in `native/core/checkpoint/` was not.
    """
    assert checkpoint.DEFAULT_REVISION in checkpoint.GRAD_REVISIONS
