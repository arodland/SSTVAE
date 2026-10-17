// The QRSS Editor: prepare a picture for the schedule and keep it as a
// Send (gui/qrss_sends.hpp).
//
// The same composer the Transmit pane has -- `OverlayEditor` for the
// canvas, `ItemMenu` on a right-click, the tool palette, templates and
// the framing dialog -- in a window of its own, so preparing tonight's
// pictures does not disturb what is on the Transmit pane. What it saves
// is the composed 640 x 480 picture plus the base and overlay apart, so
// a Send can be opened here again with its text still editable.

#ifndef SSTVAE_GUI_QRSS_EDITOR_HPP
#define SSTVAE_GUI_QRSS_EDITOR_HPP

#include <QDialog>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "images/images.hpp"
#include "images/types.hpp"
#include "overlay/model.hpp"
#include "overlay/template.hpp"

class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QToolButton;

namespace sstvae::gui {

class ItemMenu;
class OverlayEditor;
class QrssSends;

class QrssEditor : public QDialog {
    Q_OBJECT

public:
    // What the Editor borrows from the rest of the app. All optional.
    struct Context {
        // Values for {mycall}, {grid}, {utc} and the rest, as the
        // Transmit pane fills them.
        std::function<overlay::Fields()> fields;
        // The newest received picture, for a "last received" inset.
        std::function<std::optional<images::Picture>()> last_rx;
        // The text a new text item starts with (the callsign).
        std::string default_text;
        // The operator's own templates, after the built-ins.
        std::filesystem::path template_dir;
        // Where Open looks first.
        std::string picture_dir;
    };

    QrssEditor(QrssSends* sends, Context context, QWidget* parent = nullptr);
    ~QrssEditor() override;

    // A plain black picture to write on.
    void start_blank();
    // A picture already 640 x 480 (the Transmit pane's base) with an
    // overlay on it.
    void start_from(const images::Picture& base, const overlay::Doc& doc,
                    const std::string& label);
    // An existing Send, to change it. Save replaces it.
    bool open_send(const std::string& id);
    // A picture file as the base, framed first unless it is exactly
    // 640 x 480 (the schedule's rule). False if it would not load.
    bool load_picture(const QString& path);
    void choose_framing();

    // The Send being changed, or empty for a new one.
    const std::string& send_id() const { return send_id_; }
    OverlayEditor* editor() const { return editor_; }
    // Save as the Send being edited (or a new one); returns its id, or
    // empty when it could not be saved (the reason is shown).
    std::string save(bool as_new = false);

signals:
    // A Send was written; the window that opened the Editor selects it.
    void saved(const QString& id);

private:
    void build();
    void set_base(const images::Picture& base);
    void apply_framing();
    void refresh_templates();
    void refresh_fields();
    void on_selection(overlay::Item* item);
    void update_buttons();

    QrssSends* sends_ = nullptr;
    Context context_;
    std::string send_id_;

    // The picture under the overlay, framed to 640 x 480.
    images::Picture base_;
    // The file it came from, at its own size, so re-framing starts from
    // the original.
    std::optional<images::Picture> source_;
    QString source_path_;
    images::Framing framing_;

    OverlayEditor* editor_ = nullptr;
    ItemMenu* item_menu_ = nullptr;
    QPushButton* framing_button_ = nullptr;
    QLabel* picture_label_ = nullptr;
    QToolButton* add_rx_button_ = nullptr;
    QComboBox* template_combo_ = nullptr;
    std::vector<overlay::Doc> templates_;
    QPlainTextEdit* text_edit_ = nullptr;
    QComboBox* align_combo_ = nullptr;
    QLineEdit* name_ = nullptr;
    QPushButton* save_ = nullptr;
    QPushButton* save_new_ = nullptr;
    QLabel* status_ = nullptr;
    bool loading_ = false;
};

}  // namespace sstvae::gui

#endif
