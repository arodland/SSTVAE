"""native/core/qrss/rx_config.hpp is what tools/gen_qrss_rx_config.py
generates from the receiver's modules now (to 4 ulp on a float computed
through libm at import): the C++ port's one copy of the receiver's
calibrations must not drift from the Python it was measured on."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rx_config_header_is_current():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_qrss_rx_config.py"), "--check"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
