// What the Transmit pane's composer and the QRSS Editor share: the tool
// palette's hand-drawn icons, the picture file filter, and where the
// built-in overlay templates live. One copy, so the two windows offer
// the same tools and the same templates.

#ifndef SSTVAE_GUI_COMPOSE_TOOLS_HPP
#define SSTVAE_GUI_COMPOSE_TOOLS_HPP

#include <QIcon>

#include <filesystem>

class QWidget;

namespace sstvae::gui::compose {

extern const char* const IMAGE_FILTER;

QIcon text_tool_icon(const QWidget* metrics);
QIcon image_tool_icon(const QWidget* metrics);
QIcon last_rx_tool_icon(const QWidget* metrics);
QIcon rect_tool_icon(const QWidget* metrics);

// The Sends list's buttons (Jeff, 2026-10-16, option B of the mockup):
// a page with a plus, an open folder, a pencil and a bin, line drawn.
QIcon new_send_icon(const QWidget* metrics, int size);
QIcon open_file_icon(const QWidget* metrics, int size);
QIcon edit_icon(const QWidget* metrics, int size);
QIcon delete_icon(const QWidget* metrics, int size);

// The built-in templates' directory (see the .cpp for where an installed
// app keeps them).
std::filesystem::path builtin_templates_dir();

}  // namespace sstvae::gui::compose

#endif
