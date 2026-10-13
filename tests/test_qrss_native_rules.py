"""Parity of native/core/qrss/rules (sstvae_native.qrss.rules) with
sstvae/qrss/rules.py: the same environment variables, read and written so
that Python's os.environ and the C++'s getenv agree in one process (and in
a process either one starts), the same refusal for an unknown rule.
"""

import argparse
import os
import subprocess
import sys

import pytest

from qrss_native import require
from sstvae.qrss import rules


@pytest.fixture(scope="module")
def nr(native):
    return require(native, "rules")


@pytest.fixture
def clean_env(monkeypatch):
    """Both rules unset, and restored afterwards whatever the test wrote
    (monkeypatch records the original values before anything changes)."""
    for r in rules.RULES:
        monkeypatch.setenv(r, "sentinel")
        monkeypatch.delenv(r)
    return monkeypatch


def test_constants(nr):
    assert nr.EDGE_NOISE == rules.EDGE_NOISE == "QRSS_OLD_EDGE_NOISE"
    assert nr.GATING == rules.GATING == "QRSS_OLD_GATING"
    assert nr.RULES == rules.RULES


@pytest.mark.parametrize("value", [None, "", "0", "1", "yes", "00", " ", "false", "0 "])
def test_old_reads_what_the_reference_reads(nr, clean_env, value):
    for r in rules.RULES:
        if value is not None:
            clean_env.setenv(r, value)
        assert nr.old(r) == rules.old(r), (r, value)
    assert nr.old("SOME_OTHER_VARIABLE_THAT_IS_NOT_SET") is False


def test_set_old_writes_through_os_environ(nr, clean_env):
    for r in rules.RULES:
        nr.set_old(r)
        assert os.environ[r] == "1" and rules.old(r) and nr.old(r)
        nr.set_old(r, False)
        assert r not in os.environ and not rules.old(r) and not nr.old(r)
        nr.set_old(r, on=1)
        assert rules.old(r) and nr.old(r)
        nr.set_old(r, on=0)
        nr.set_old(r, on=0)          # unsetting an unset rule is quiet
        assert not rules.old(r) and not nr.old(r)
        rules.set_old(r)             # the reference's writes reach the C++
        assert nr.old(r)
        rules.set_old(r, False)
        assert not nr.old(r)


def test_a_child_process_follows(nr, clean_env):
    nr.set_old(rules.GATING)
    code = "import os; print(os.environ.get('QRSS_OLD_GATING'), os.environ.get('QRSS_OLD_EDGE_NOISE'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["1", "None"]


@pytest.mark.parametrize("rule", ["X", "qrss_old_gating", "", 5, None, ("QRSS_OLD_GATING",)])
def test_unknown_rules(nr, clean_env, rule):
    with pytest.raises(ValueError) as ref:
        rules.set_old(rule)
    with pytest.raises(ValueError) as got:
        nr.set_old(rule)
    assert str(got.value) == str(ref.value)


def test_apply(nr, clean_env):
    nr.apply(argparse.Namespace(old_edge_noise=True))
    assert rules.old(rules.EDGE_NOISE) and not rules.old(rules.GATING)
    nr.apply(argparse.Namespace(old_gating=True, old_edge_noise=False))
    assert rules.old(rules.EDGE_NOISE) and rules.old(rules.GATING)   # a false flag leaves its rule
    nr.apply(object())
    assert rules.old(rules.EDGE_NOISE) and rules.old(rules.GATING)
