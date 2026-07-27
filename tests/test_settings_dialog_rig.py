"""The rig-model picker in the settings dialog.

The value stored in the config is a Hamlib model *number*, but the user
now picks from labels. The invariants worth pinning: the number saved is
the number that comes back, a config naming a model this Hamlib doesn't
list is not silently reset, and a machine with no Hamlib can still edit
the field and is told why the list is empty.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from sstvae.gui import settings_dialog as sd  # noqa: E402
from sstvae.gui.settings import Config  # noqa: E402
from sstvae.rig import RigError, RigModel  # noqa: E402

_APP = None

FAKE_MODELS = [
    RigModel(2, "Hamlib", "NET rigctl", "20250211.0", "Stable", "RIG_MODEL_NETRIGCTL"),
    RigModel(4, "FLRig", "FLRig", "20260130.0", "Stable", "RIG_MODEL_FLRIG"),
    RigModel(1001, "Yaesu", "FT-847", "20230512.0", "Stable", "RIG_MODEL_FT847"),
    RigModel(42001, "Harris", "PRC-138", "1.0.6", "Alpha", "RIG_MODEL_PRC138"),
]


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.fixture
def listed(monkeypatch):
    monkeypatch.setattr(sd, "list_models", lambda: list(FAKE_MODELS))


def _dialog(model: str, spawn: bool = False) -> sd.SettingsDialog:
    cfg = Config()
    cfg.rig.model = model
    cfg.rig.spawn_local = spawn
    return sd.SettingsDialog(cfg)


def test_saved_model_preselects_and_survives_a_round_trip(qapp, listed):
    dlg = _dialog("1001")
    assert dlg.rig_model.currentText() == "Yaesu FT-847 (1001)"

    out = Config()
    dlg.apply_to(out)
    assert out.rig.model == "1001"


def test_choosing_a_different_rig_stores_its_number(qapp, listed):
    dlg = _dialog("1001")
    dlg.rig_model.setCurrentIndex(dlg.rig_model.findData(42001))

    out = Config()
    dlg.apply_to(out)
    assert out.rig.model == "42001"


def test_model_absent_from_this_hamlib_is_not_silently_reset(qapp, listed):
    # A config written where Hamlib listed model 9999; this build doesn't.
    dlg = _dialog("9999")
    assert dlg.rig_model.currentText() == "9999"

    out = Config()
    dlg.apply_to(out)
    assert out.rig.model == "9999"


def test_typed_bare_number_is_accepted(qapp, listed):
    dlg = _dialog("1001")
    dlg.rig_model.setEditText("3073")

    out = Config()
    dlg.apply_to(out)
    assert out.rig.model == "3073"


def test_missing_hamlib_surfaces_the_reason_and_blocks_spawning(qapp, monkeypatch):
    def boom():
        raise RigError("rigctld not found on PATH. Install Hamlib (...).")

    monkeypatch.setattr(sd, "list_models", boom)
    dlg = _dialog("1001", spawn=True)

    assert dlg.rig_model.count() == 0
    assert "Install Hamlib" in dlg.rig_model_note.text()
    # Offering to start a daemon that isn't installed can only fail later.
    assert not dlg.rig_spawn.isEnabled()

    # ...but a missing Hamlib on the machine editing the config must not
    # clear settings the user never touched.
    out = Config()
    dlg.apply_to(out)
    assert out.rig.spawn_local is True
    # The model field is still editable, and still keeps its value.
    assert dlg.rig_model.currentText() == "1001"
    assert out.rig.model == "1001"
