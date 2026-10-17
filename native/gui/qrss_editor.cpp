#include "qrss_editor.hpp"

#include <QComboBox>
#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTextDocument>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <exception>
#include <utility>
#include <variant>

#include "compose_tools.hpp"
#include "crop_dialog.hpp"
#include "flow_layout.hpp"
#include "item_menu.hpp"
#include "overlay/template_catalog.hpp"
#include "overlay_editor.hpp"
#include "qrss_sends.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace {

images::Picture black_picture() {
    return images::Picture(images::IMG_W, images::IMG_H);
}

QFrame* vrule(QWidget* parent, int height) {
    auto* rule = new QFrame(parent);
    rule->setFrameShape(QFrame::VLine);
    rule->setFrameShadow(QFrame::Sunken);
    rule->setFixedHeight(height);
    return rule;
}

}  // namespace

QrssEditor::QrssEditor(QrssSends* sends, Context context, QWidget* parent)
    : QDialog(parent), sends_(sends), context_(std::move(context)) {
    setObjectName(QStringLiteral("qrss_editor"));
    setWindowTitle(tr("QRSS Editor"));
    build();
    refresh_templates();
    if (context_.last_rx) {
        if (const auto rx = context_.last_rx()) {
            editor_->set_last_rx(*rx);
            add_rx_button_->setEnabled(true);
            add_rx_button_->setToolTip(tr("Add last received"));
        }
    }
    refresh_fields();
    start_blank();
    resize(760, 760);
}

QrssEditor::~QrssEditor() = default;

void QrssEditor::build() {
    auto* outer = new QVBoxLayout(this);

    editor_ = new OverlayEditor(this);
    editor_->setObjectName(QStringLiteral("editor_canvas"));
    connect(editor_, &OverlayEditor::selectionChanged, this, &QrssEditor::on_selection);
    connect(editor_, &OverlayEditor::documentChanged, this, &QrssEditor::update_buttons);
    item_menu_ = new ItemMenu(editor_, this);
    connect(editor_, &OverlayEditor::contextMenuRequested, item_menu_, &ItemMenu::popup_for);
    outer->addWidget(editor_, 1);

    // --- the Transmit pane's tool row --------------------------------
    auto* tools = new QWidget(this);
    auto* flow = new FlowLayout(tools);
    flow->setContentsMargins(0, 0, 0, 0);
    auto* open = new QPushButton(tr("&Image..."), tools);
    open->setObjectName(QStringLiteral("editor_open"));
    open->setToolTip(tr("Choose the picture to write on"));
    connect(open, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Picture"), QString::fromStdString(context_.picture_dir),
            QString::fromLatin1(compose::IMAGE_FILTER));
        if (!path.isEmpty()) load_picture(path);
    });
    framing_button_ = new QPushButton(tr("&Framing..."), tools);
    framing_button_->setObjectName(QStringLiteral("editor_framing"));
    framing_button_->setToolTip(tr("Choose which part of the picture goes on the air"));
    framing_button_->setEnabled(false);
    connect(framing_button_, &QPushButton::clicked, this, &QrssEditor::choose_framing);
    auto* blank = new QPushButton(tr("&Blank"), tools);
    blank->setToolTip(tr("Start again on a plain black picture"));
    connect(blank, &QPushButton::clicked, this, [this] {
        const std::string keep = send_id_;
        start_blank();
        send_id_ = keep;
        update_buttons();
    });
    flow->addWidget(open);
    flow->addWidget(framing_button_);
    flow->addWidget(blank);
    flow->addWidget(vrule(tools, open->sizeHint().height()));

    auto* add_text = new QToolButton(tools);
    add_text->setObjectName(QStringLiteral("editor_add_text"));
    add_text->setIcon(compose::text_tool_icon(this));
    add_text->setToolTip(tr("Add text"));
    connect(add_text, &QToolButton::clicked, this, [this] {
        editor_->add_text(context_.default_text.empty() ? std::string("TEXT")
                                                        : context_.default_text);
    });
    add_rx_button_ = new QToolButton(tools);
    add_rx_button_->setIcon(compose::last_rx_tool_icon(this));
    add_rx_button_->setEnabled(false);
    add_rx_button_->setToolTip(
        tr("Add last received: available once a picture has been received"));
    connect(add_rx_button_, &QToolButton::clicked, editor_, &OverlayEditor::add_last_rx_inset);
    auto* add_image = new QToolButton(tools);
    add_image->setIcon(compose::image_tool_icon(this));
    add_image->setToolTip(tr("Add image..."));
    connect(add_image, &QToolButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Choose an inset image"), QString::fromStdString(context_.picture_dir),
            QString::fromLatin1(compose::IMAGE_FILTER));
        if (!path.isEmpty()) editor_->add_image_inset(path.toStdString());
    });
    auto* add_rect = new QToolButton(tools);
    add_rect->setIcon(compose::rect_tool_icon(this));
    add_rect->setToolTip(tr("Add rectangle"));
    connect(add_rect, &QToolButton::clicked, editor_, &OverlayEditor::add_rect);
    for (QToolButton* b : {add_text, add_rx_button_, add_image, add_rect}) {
        b->setToolButtonStyle(Qt::ToolButtonIconOnly);
        flow->addWidget(b);
    }
    flow->addWidget(vrule(tools, open->sizeHint().height()));

    template_combo_ = new QComboBox(tools);
    template_combo_->setObjectName(QStringLiteral("editor_template"));
    template_combo_->setToolTip(tr("Loads that template's layout onto the picture, replacing "
                                   "the overlay there now. \"None\" clears it."));
    connect(template_combo_, &QComboBox::activated, this, [this](int index) {
        if (index < 0 || index >= static_cast<int>(templates_.size())) return;
        editor_->set_doc(templates_[index]);
        refresh_fields();
    });
    flow->addWidget(style::row(tools, {new QLabel(tr("Template"), tools), template_combo_}));
    outer->addWidget(tools);

    // --- text --------------------------------------------------------
    auto* text_row = new QWidget(this);
    auto* text_flow = new FlowLayout(text_row);
    text_flow->setContentsMargins(0, 0, 0, 0);
    text_edit_ = new QPlainTextEdit(text_row);
    text_edit_->setObjectName(QStringLiteral("editor_text"));
    text_edit_->setTabChangesFocus(true);
    text_edit_->setFixedHeight(text_edit_->fontMetrics().lineSpacing() * 2 +
                               static_cast<int>(text_edit_->document()->documentMargin()) * 2 +
                               text_edit_->frameWidth() * 2 + 4);
    text_edit_->setMinimumWidth(240);
    text_edit_->setEnabled(false);
    text_edit_->setToolTip(tr("The selected text item's words. Enter starts a new line; Tab "
                              "leaves the field. Right-click an item for its font, colours "
                              "and order."));
    connect(text_edit_, &QPlainTextEdit::textChanged, this, [this] {
        if (loading_) return;
        if (auto* item = editor_->selected_item()) {
            if (auto* text = std::get_if<overlay::TextItem>(item)) {
                text->text = text_edit_->toPlainText().toStdString();
                editor_->refresh_item();
            }
        }
    });
    align_combo_ = new QComboBox(text_row);
    align_combo_->addItem(tr("Left"), QStringLiteral("left"));
    align_combo_->addItem(tr("Centre"), QStringLiteral("center"));
    align_combo_->addItem(tr("Right"), QStringLiteral("right"));
    align_combo_->setEnabled(false);
    connect(align_combo_, &QComboBox::currentIndexChanged, this, [this] {
        if (loading_) return;
        if (auto* item = editor_->selected_item()) {
            if (auto* text = std::get_if<overlay::TextItem>(item)) {
                text->align = align_combo_->currentData().toString().toStdString();
                editor_->refresh_item();
            }
        }
    });
    text_flow->addWidget(style::row(text_row, {new QLabel(tr("Text"), text_row), text_edit_}, 1));
    text_flow->addWidget(style::row(text_row, {new QLabel(tr("Align"), text_row), align_combo_}));
    outer->addWidget(text_row);

    picture_label_ = new style::ElidingLabel(this);
    picture_label_->setObjectName(QStringLiteral("editor_picture_label"));
    picture_label_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    outer->addWidget(picture_label_);

    // --- saving -------------------------------------------------------
    name_ = new QLineEdit(this);
    name_->setObjectName(QStringLiteral("editor_name"));
    name_->setPlaceholderText(tr("A name for this Send"));
    name_->setMaxLength(60);
    save_ = new QPushButton(tr("&Save to Sends"), this);
    save_->setObjectName(QStringLiteral("editor_save"));
    save_->setDefault(true);
    connect(save_, &QPushButton::clicked, this, [this] {
        if (!save(false).empty()) accept();
    });
    save_new_ = new QPushButton(tr("Save as &new Send"), this);
    save_new_->setObjectName(QStringLiteral("editor_save_new"));
    save_new_->setToolTip(tr("Keep the Send you opened as it was, and save this as another"));
    connect(save_new_, &QPushButton::clicked, this, [this] {
        if (!save(true).empty()) accept();
    });
    auto* close = new QPushButton(tr("Close"), this);
    connect(close, &QPushButton::clicked, this, &QDialog::reject);
    auto* bottom = new QHBoxLayout;
    bottom->addWidget(new QLabel(tr("Name"), this));
    bottom->addWidget(name_, 1);
    bottom->addWidget(save_);
    bottom->addWidget(save_new_);
    bottom->addWidget(close);
    outer->addLayout(bottom);
    status_ = new QLabel(this);
    status_->setObjectName(QStringLiteral("editor_status"));
    status_->setWordWrap(true);
    outer->addWidget(status_);
}

void QrssEditor::refresh_templates() {
    templates_.clear();
    template_combo_->clear();
    templates_.push_back(overlay::Doc());
    template_combo_->addItem(tr("None"));
    for (overlay::Doc& doc : overlay::load_builtin_templates(compose::builtin_templates_dir())) {
        template_combo_->addItem(QString::fromStdString(doc.name));
        templates_.push_back(std::move(doc));
    }
    if (!context_.template_dir.empty()) {
        for (overlay::LoadedTemplate& t : overlay::load_templates(context_.template_dir)) {
            template_combo_->addItem(QString::fromStdString(
                t.doc.name.empty() ? t.path.stem().string() : t.doc.name));
            templates_.push_back(std::move(t.doc));
        }
    }
}

void QrssEditor::refresh_fields() {
    editor_->set_fields(context_.fields ? context_.fields() : overlay::Fields{});
}

void QrssEditor::set_base(const images::Picture& base) {
    base_ = base;
    editor_->set_base_image(base_);
}

void QrssEditor::start_blank() {
    send_id_.clear();
    source_.reset();
    source_path_.clear();
    framing_button_->setEnabled(false);
    set_base(black_picture());
    editor_->set_doc({});
    template_combo_->setCurrentIndex(0);
    name_->clear();
    picture_label_->setText(tr("A plain black picture: add text, or choose a picture with "
                               "Image..."));
    status_->clear();
    update_buttons();
}

void QrssEditor::start_from(const images::Picture& base, const overlay::Doc& doc,
                            const std::string& label) {
    send_id_.clear();
    source_.reset();
    source_path_.clear();
    framing_button_->setEnabled(false);
    set_base(base.width == images::IMG_W && base.height == images::IMG_H ? base
                                                                          : images::fit(base));
    editor_->set_doc(doc);
    name_->setText(QString::fromStdString(label));
    picture_label_->setText(tr("The composition from the Transmit pane"));
    status_->clear();
    update_buttons();
}

bool QrssEditor::open_send(const std::string& id) {
    const QrssSend* s = sends_ ? sends_->find(id) : nullptr;
    if (s == nullptr) return false;
    const std::optional<images::Picture> base = sends_->load_base(id);
    if (!base) return false;
    send_id_ = id;
    source_.reset();
    source_path_.clear();
    framing_button_->setEnabled(false);
    set_base(*base);
    editor_->set_doc(sends_->load_doc(id));
    name_->setText(QString::fromStdString(s->label));
    picture_label_->setText(tr("Editing the Send \"%1\". Save replaces it; sends already on "
                               "the schedule keep the picture they were scheduled with.")
                                .arg(QString::fromStdString(s->label)));
    status_->clear();
    update_buttons();
    return true;
}

bool QrssEditor::load_picture(const QString& path) {
    images::Picture loaded;
    try {
        loaded = images::load(path.toStdString());
    } catch (const std::exception& e) {
        status_->setText(tr("Could not open %1: %2").arg(path, QString::fromUtf8(e.what())));
        return false;
    }
    source_ = std::move(loaded);
    source_path_ = path;
    framing_ = images::Framing{};
    // As in the schedule: any size but the one sent asks first, so the
    // picture is never rescaled or cropped unseen. Cancel keeps the
    // default framing.
    if (source_->width != images::IMG_W || source_->height != images::IMG_H) {
        CropDialog dialog(*source_, framing_, this);
        if (dialog.exec() == QDialog::Accepted) framing_ = dialog.framing();
    }
    apply_framing();
    if (name_->text().isEmpty()) name_->setText(QFileInfo(path).completeBaseName());
    return true;
}

void QrssEditor::choose_framing() {
    if (!source_) return;
    CropDialog dialog(*source_, framing_, this);
    if (dialog.exec() != QDialog::Accepted) return;
    framing_ = dialog.framing();
    apply_framing();
}

void QrssEditor::apply_framing() {
    if (!source_) return;
    try {
        set_base(images::fit(*source_, framing_));
    } catch (const std::exception& e) {
        status_->setText(tr("Could not frame %1: %2")
                             .arg(QFileInfo(source_path_).fileName(), QString::fromUtf8(e.what())));
        return;
    }
    const images::Picture& src = *source_;
    QString caption = tr("%1, %2x%3").arg(QFileInfo(source_path_).fileName())
                          .arg(src.width)
                          .arg(src.height);
    const bool four_by_three = src.width * images::IMG_H == src.height * images::IMG_W;
    if (framing_.zoom < 1.0) {
        caption += tr(", padded to 4:3");
    } else if (!four_by_three || framing_.zoom > 1.0) {
        caption += tr(", cropped to 4:3");
    }
    picture_label_->setText(caption);
    framing_button_->setEnabled(true);
    update_buttons();
}

void QrssEditor::on_selection(overlay::Item* item) {
    loading_ = true;
    auto* text = item ? std::get_if<overlay::TextItem>(item) : nullptr;
    text_edit_->setEnabled(text != nullptr);
    align_combo_->setEnabled(text != nullptr);
    text_edit_->setPlainText(text ? QString::fromStdString(text->text) : QString());
    if (text) {
        align_combo_->setCurrentIndex(
            std::max(0, align_combo_->findData(QString::fromStdString(text->align))));
    }
    loading_ = false;
}

void QrssEditor::update_buttons() {
    const bool editing = !send_id_.empty();
    save_->setText(editing ? tr("&Save changes") : tr("&Save to Sends"));
    save_new_->setVisible(editing);
}

std::string QrssEditor::save(bool as_new) {
    if (sends_ == nullptr) return {};
    refresh_fields();
    const std::optional<images::Picture> picture = editor_->composed_image();
    if (!picture) {
        status_->setText(tr("Nothing to save yet: choose a picture first."));
        return {};
    }
    std::string label = name_->text().trimmed().toStdString();
    if (label.empty()) {
        label = QDateTime::currentDateTimeUtc()
                    .toString(QStringLiteral("'picture' d MMM HH:mm'Z'"))
                    .toStdString();
    }
    std::string why;
    std::string id;
    if (!send_id_.empty() && !as_new) {
        if (sends_->update(send_id_, label, base_, editor_->doc(), *picture, &why)) id = send_id_;
    } else {
        id = sends_->add(label, base_, editor_->doc(), *picture, &why);
    }
    if (id.empty()) {
        status_->setText(tr("Could not save: %1").arg(QString::fromStdString(why)));
        return {};
    }
    send_id_ = id;
    update_buttons();
    status_->setText(tr("Saved \"%1\".").arg(QString::fromStdString(label)));
    emit saved(QString::fromStdString(id));
    return id;
}

}  // namespace sstvae::gui
