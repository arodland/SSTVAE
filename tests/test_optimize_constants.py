"""The optimizer's constants are hand-maintained in two places.

`sstvae/latent_optim.py` and `native/core/optimize/optimize.hpp` each
declare the learning rate, the objective SNR and the channel-sample
count, and unlike the waveform constants there is no generator keeping
them in step -- `tools/gen_config_header.py` covers `sstvae/config.py`
only, and these do not belong there because nothing on the air depends
on them.

That last point is why this is a test rather than a build gate: the
optimizer is **sender-side only**, so a divergence does not break
interoperability with anyone. Both implementations would keep producing
valid unit-RMS latents and every receiver would decode them. It would
just mean the two apps quietly optimize to different qualities, with
nothing to show for it but a dB somewhere -- which is precisely the
failure that survives a long time unnoticed, and precisely why the value
being duplicated needs something watching it.

Parsed out of the header rather than compiled, so this runs with no
build and no toolchain.
"""

import re
from pathlib import Path

import pytest

from sstvae import latent_optim

HEADER = (Path(__file__).resolve().parents[1]
          / "native" / "core" / "optimize" / "optimize.hpp")


def _constant(name: str) -> str:
    text = HEADER.read_text()
    m = re.search(
        rf"^inline\s+constexpr\s+\w+\s+{re.escape(name)}\s*=\s*([^;]+);",
        text, re.MULTILINE)
    assert m, f"{name} not found in {HEADER.name} -- renamed or removed?"
    return m.group(1).strip()


@pytest.mark.parametrize("name, python_value, cast", [
    ("LEARNING_RATE", latent_optim.LEARNING_RATE, float),
    ("OBJECTIVE_SNR_DB", latent_optim.OBJECTIVE_SNR_DB, float),
    ("CHANNEL_SAMPLES", latent_optim.CHANNEL_SAMPLES, int),
])
def test_the_two_implementations_agree(name, python_value, cast):
    assert cast(_constant(name)) == python_value, (
        f"{name} is {python_value} in sstvae/latent_optim.py but "
        f"{_constant(name)} in {HEADER.name}. Python is normative; "
        f"change both in one commit."
    )
