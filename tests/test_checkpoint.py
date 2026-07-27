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


def test_an_explicit_path_short_circuits_everything(fake_hub):
    hub = fake_hub(cached="/cache/v1.pt", remote="/downloaded/v1.pt")
    assert checkpoint.resolve("/my/own.pt") == "/my/own.pt"
    assert hub.calls == []
