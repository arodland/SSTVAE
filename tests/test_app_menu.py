"""The File menu must survive macOS's menu-bar rewriting.

Qt's Cocoa plugin pattern-matches action text and relocates anything
resembling Preferences or Quit into the application menu. Both of the
File menu's actions match, so Qt emptied it -- and macOS hides an empty
menu, which made Settings unreachable in the shipped app. These tests
pin the roles and the shortcuts that keep it reachable. They can only
check the *inputs* to that behaviour; the relocation itself happens
inside the Cocoa plugin and does not exist on other platforms.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction, QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu  # noqa: E402

from sstvae.gui.app import MainWindow  # noqa: E402

_APP = None


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.fixture
def menu(qapp):
    """The real `_build_menu`, run against a bare QMainWindow.

    A real `MainWindow` starts a model-loading thread and dials the rig;
    the menu code only needs `menuBar`, `open_settings` and `close`.
    """
    window = QMainWindow()
    window.open_settings = lambda: None
    MainWindow._build_menu(window)

    # Reach the menu as a child of the window, not via
    # `menuBar().actions()[0].menu()`: that chain goes through temporary
    # PySide wrappers, and shiboken invalidates the QMenu wrapper when
    # the QAction one it came from is collected. The menu itself is alive
    # either way -- the window owns it.
    file_menu = window.findChildren(QMenu)[0]
    actions = {a.text(): a for a in file_menu.actions() if not a.isSeparator()}
    yield actions
    window.deleteLater()


def test_file_menu_keeps_both_actions(menu):
    assert set(menu) == {"&Settings...", "&Quit"}


def test_actions_opt_out_of_platform_relocation(menu):
    # Without NoRole, macOS moves both into the application menu and
    # hides the emptied File menu.
    for action in menu.values():
        assert action.menuRole() == QAction.MenuRole.NoRole


def test_settings_and_quit_have_platform_shortcuts(menu):
    assert menu["&Settings..."].shortcut() == QKeySequence(
        QKeySequence.StandardKey.Preferences
    )
    assert menu["&Quit"].shortcut() == QKeySequence(QKeySequence.StandardKey.Quit)
    assert not menu["&Settings..."].shortcut().isEmpty()
