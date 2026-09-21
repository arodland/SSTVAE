#include "item_menu.hpp"

#include <QAbstractButton>
#include <QAction>
#include <QApplication>
#include <QColor>
#include <QColorDialog>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVariant>
#include <QWheelEvent>
#include <QWidgetAction>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <variant>

#include "overlay_editor.hpp"

namespace sstvae::gui {
namespace {

// --- sizes in pixels -------------------------------------------------------
//
// The menu shows sizes in pixels of the 640x480 transmitted frame, which
// is the unit an operator composing a picture thinks in; the document
// stores fractions of it, which is what keeps a saved overlay meaningful
// at any resolution. The ranges are the editor's own clamps
// (`OverlayEditor::scale_item`), rounded inward so neither end is a value
// the document would refuse.
constexpr double TEXT_SIZE_MIN = 0.01;
constexpr double TEXT_SIZE_MAX = 1.5;
constexpr double RECT_STROKE_MAX = 0.2;  // the palette this menu replaced offered no more

int px(double fraction, int extent) {
    return static_cast<int>(std::lround(fraction * extent));
}

// A text stroke is a fraction of the glyph size, which is a percentage
// everywhere an operator meets it.
constexpr int STROKE_MAX_PCT = 50;

// --- fill and stroke modes -------------------------------------------------
//
// One button cycles a fill or stroke through its kind and its gradient's
// shape together, because to the operator "radial" is a kind of fill,
// not a separate setting. For text, "none" is outlined text; for a rect,
// it is no fill (or no stroke) at all.
enum class Mode { None, Solid, Linear, Radial };

Mode mode_of(const std::string& kind, const std::string& shape) {
    if (kind == "none") return Mode::None;
    if (kind == "gradient") return shape == "radial" ? Mode::Radial : Mode::Linear;
    // An unknown kind reads as solid, which is what the renderer draws
    // for it on text. (A rect draws nothing for one; it shows as solid
    // here all the same, and cycling from it writes a known kind.)
    return Mode::Solid;
}

void apply_mode(Mode mode, std::string& kind, std::string& shape) {
    switch (mode) {
        case Mode::None: kind = "none"; break;
        case Mode::Solid: kind = "solid"; break;
        case Mode::Linear:
            kind = "gradient";
            shape = "linear";
            break;
        case Mode::Radial:
            kind = "gradient";
            shape = "radial";
            break;
    }
}

// Solid -> linear -> radial -> outline for text, which is always filled
// unless asked not to be; none -> solid -> linear -> radial for a rect,
// which starts empty.
Mode next_text_mode(Mode mode) {
    switch (mode) {
        case Mode::Solid: return Mode::Linear;
        case Mode::Linear: return Mode::Radial;
        case Mode::Radial: return Mode::None;
        case Mode::None: return Mode::Solid;
    }
    return Mode::Solid;
}

Mode next_rect_mode(Mode mode) {
    switch (mode) {
        case Mode::None: return Mode::Solid;
        case Mode::Solid: return Mode::Linear;
        case Mode::Linear: return Mode::Radial;
        case Mode::Radial: return Mode::None;
    }
    return Mode::None;
}

QString mode_label(Mode mode, bool text) {
    switch (mode) {
        case Mode::None: return text ? ItemMenu::tr("OUTLINE") : ItemMenu::tr("NONE");
        case Mode::Solid: return ItemMenu::tr("SOLID");
        case Mode::Linear: return ItemMenu::tr("LINEAR");
        case Mode::Radial: return ItemMenu::tr("RADIAL");
    }
    return QString();
}

bool is_gradient(Mode mode) { return mode == Mode::Linear || mode == Mode::Radial; }

// --- the font family -------------------------------------------------------

// The generic keywords, plus "Default" for no request. Real family names
// are not offered: the set that means the same thing on three platforms
// is exactly these four.
struct Family {
    const char* label;
    const char* value;
};
constexpr Family FAMILIES[] = {
    {"Default", ""},
    {"Sans-Serif", "sans-serif"},
    {"Serif", "serif"},
    {"Monospace", "monospace"},
    {"Cursive", "cursive"},
};

QString family_label(const std::string& value) {
    for (const Family& family : FAMILIES) {
        if (value == family.value) return ItemMenu::tr(family.label);
    }
    // A document may name a real face; show it rather than claim it is
    // the default.
    return QString::fromStdString(value);
}

// --- small widgets ---------------------------------------------------------

// Paint `color` onto `button` as its icon. The last colour is remembered
// on the button itself, so reloading the same colour is free; an unset
// property and an invalid colour are different, which lets a button be
// given its (empty) swatch at construction. It should be: a button grows
// when first handed an icon, and a control that changes size when the
// operator selects something is a layout that moves under them.
void paint_swatch(QAbstractButton* button, const QColor& color) {
    static const char* const REMEMBERED = "sstvae_swatch_color";
    const QVariant previous = button->property(REMEMBERED);
    if (previous.isValid() && previous.value<QColor>() == color) return;
    button->setProperty(REMEMBERED, color);

    const int size = button->style()->pixelMetric(QStyle::PM_SmallIconSize);
    // Device pixels, so a HiDPI screen gets a crisp square rather than an
    // upscaled one.
    const qreal dpr = button->devicePixelRatioF();
    QPixmap swatch(static_cast<int>(std::lround(size * dpr)),
                   static_cast<int>(std::lround(size * dpr)));
    swatch.setDevicePixelRatio(dpr);
    swatch.fill(color.isValid() ? color : Qt::transparent);
    if (color.isValid()) {
        QPainter painter(&swatch);
        painter.setPen(button->palette().color(QPalette::WindowText));
        painter.drawRect(0, 0, size - 1, size - 1);
    }
    button->setIcon(QIcon(swatch));
}

// One wheel event in notches, sign preserved. A high-resolution wheel
// reports less than a notch per event, and rounding that to zero makes a
// field simply not respond on a trackpad.
int notches_of(const QWheelEvent* wheel) {
    const int dy = wheel->angleDelta().y();
    if (dy == 0) return 0;
    const int whole = dy / 120;
    return whole != 0 ? whole : (dy > 0 ? 1 : -1);
}

QToolButton* toggle(QWidget* parent, const QString& label, const QString& name,
                    const QString& tip) {
    auto* button = new QToolButton(parent);
    button->setObjectName(name);
    button->setText(label);
    button->setCheckable(true);
    button->setToolTip(tip);
    return button;
}

QSpinBox* spin(QWidget* parent, const QString& name, int lo, int hi, const QString& suffix,
               const QString& tip) {
    auto* box = new QSpinBox(parent);
    box->setObjectName(name);
    box->setRange(lo, hi);
    box->setSuffix(suffix);
    box->setToolTip(tip);
    return box;
}

QPushButton* mode_button(QWidget* parent, const QString& name, const QString& tip) {
    auto* button = new QPushButton(parent);
    button->setObjectName(name);
    button->setToolTip(tip);
    return button;
}

// A row of live controls, parented to its menu so `findChild` reaches it
// before the menu has ever been shown. `QWidgetActionPrivate::
// defaultWidget` is a `QPointer`, so the double ownership is safe in
// either destruction order.
QWidgetAction* row_action(QMenu* menu, QWidget* row) {
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(row);
    menu->addAction(action);
    return action;
}

QHBoxLayout* row_layout(QWidget* row) {
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(8, 4, 8, 4);
    layout->setSpacing(4);
    return layout;
}

const char* const ROTATION_TIP = QT_TRANSLATE_NOOP(
    "sstvae::gui::ItemMenu",
    "Degrees, counter-clockwise. Scroll to change it; hold Shift to snap to "
    "the next multiple of 15.");
const char* const ANGLE_TIP = QT_TRANSLATE_NOOP(
    "sstvae::gui::ItemMenu",
    "The gradient's direction: 0 runs left to right, 90 bottom to top -- the "
    "same counter-clockwise sense as the item's own rotation. A radial "
    "gradient has none.");

}  // namespace

ItemMenu::ItemMenu(OverlayEditor* editor, QWidget* parent)
    : QMenu(parent), editor_(editor) {
    setObjectName(QStringLiteral("item_menu"));
    build_format();
    style_menu_ = new QMenu(tr("&Style"), this);
    style_menu_->setObjectName(QStringLiteral("menu_style"));
    addMenu(style_menu_);
    style_menu_->installEventFilter(this);
    build_text_style();
    build_rect_style();
    build_layers();
    addSeparator();
    remove_action_ = addAction(tr("&Remove"));
    remove_action_->setObjectName(QStringLiteral("menu_remove"));
    connect(remove_action_, &QAction::triggered, editor_, &OverlayEditor::remove_selected);
    update_layer_row();
}

// --- building ------------------------------------------------------------------

void ItemMenu::build_format() {
    format_menu_ = new QMenu(tr("&Format"), this);
    format_menu_->setObjectName(QStringLiteral("menu_format"));
    addMenu(format_menu_);
    format_menu_->installEventFilter(this);
    auto* row = new QWidget(format_menu_);
    row_action(format_menu_, row);
    auto* layout = row_layout(row);

    bold_ = toggle(row, tr("B"), QStringLiteral("menu_bold"), tr("Bold"));
    italic_ = toggle(row, tr("I"), QStringLiteral("menu_italic"), tr("Italic"));
    underline_ = toggle(row, tr("U"), QStringLiteral("menu_underline"), tr("Underline"));
    for (const auto& [button, field] :
         {std::pair{bold_, &overlay::TextItem::bold},
          std::pair{italic_, &overlay::TextItem::italic},
          std::pair{underline_, &overlay::TextItem::underline}}) {
        connect(button, &QToolButton::toggled, this, [this, field = field](bool on) {
            if (auto* text = current<overlay::TextItem>()) {
                text->*field = on;
                edited();
            }
        });
        layout->addWidget(button);
    }

    // **A button with its own menu, not a `QComboBox`** -- see the header.
    family_ = new QPushButton(row);
    family_->setObjectName(QStringLiteral("menu_family"));
    family_->setToolTip(tr("Font family. A document that names its own font "
                           "file keeps it -- the file wins over a family."));
    auto* families = new QMenu(family_);
    for (const Family& family : FAMILIES) {
        QAction* action = families->addAction(tr(family.label));
        const std::string value = family.value;
        connect(action, &QAction::triggered, this, [this, value] {
            if (auto* text = current<overlay::TextItem>()) {
                text->font_family = value;
                family_->setText(family_label(value));
                edited();
            }
        });
    }
    family_->setMenu(families);
    layout->addWidget(family_);

    layout->addWidget(new QLabel(tr("Size"), row));
    size_ = spin(row, QStringLiteral("menu_size"),
                 static_cast<int>(std::ceil(TEXT_SIZE_MIN * overlay::CANVAS_H)),
                 static_cast<int>(std::floor(TEXT_SIZE_MAX * overlay::CANVAS_H)), tr(" px"),
                 tr("Cap height, in pixels of the 640x480 transmitted frame. Scroll "
                    "to change it; hold Shift for 10 px."));
    size_->installEventFilter(this);
    connect(size_, &QSpinBox::valueChanged, this, [this](int value) {
        if (auto* text = current<overlay::TextItem>()) {
            text->size = value / static_cast<double>(overlay::CANVAS_H);
            edited();
        }
    });
    layout->addWidget(size_);
}

void ItemMenu::build_text_style() {
    auto* row = new QWidget(style_menu_);
    text_style_action_ = row_action(style_menu_, row);
    auto* layout = row_layout(row);

    auto* clear = new QPushButton(tr("Clear"), row);
    clear->setObjectName(QStringLiteral("menu_clear"));
    clear->setToolTip(tr("Back to the default colours, weight and fill. The "
                         "text, its place, size and rotation are left alone."));
    connect(clear, &QPushButton::clicked, this, [this] {
        overlay::TextItem* text = current<overlay::TextItem>();
        if (text == nullptr) return;
        // **Appearance only.** Everything positional -- and the text
        // itself -- survives, so this is never the button that loses
        // somebody's callsign. `font` survives too: a document naming
        // its own face names a file it needs, not a style choice.
        const overlay::TextItem fresh;
        text->color = fresh.color;
        text->fill_kind = fresh.fill_kind;
        text->fill_color2 = fresh.fill_color2;
        text->fill_angle = fresh.fill_angle;
        text->fill_gradient = fresh.fill_gradient;
        text->stroke_color = fresh.stroke_color;
        text->stroke_width = fresh.stroke_width;
        text->bold = fresh.bold;
        text->italic = fresh.italic;
        text->underline = fresh.underline;
        text->font_family = fresh.font_family;
        load_text(*text);
        edited();
    });
    layout->addWidget(clear);

    // The split fill control: first stop, mode, second stop and angle.
    // `color` is the first stop in every mode -- it is what "solid"
    // always drew, which is what keeps an existing document unchanged.
    fill_from_ = color_well(row, QStringLiteral("menu_fill_from"),
                            tr("Fill colour, and a gradient's first stop."));
    connect(fill_from_, &QPushButton::clicked, this,
            [this] { pick_color(fill_from_, &overlay::TextItem::color); });
    layout->addWidget(fill_from_);

    fill_mode_ = mode_button(row, QStringLiteral("menu_fill_mode"),
                             tr("Solid, a linear or radial gradient, or outlined text "
                                "with no fill at all."));
    connect(fill_mode_, &QPushButton::clicked, this, [this] {
        overlay::TextItem* text = current<overlay::TextItem>();
        if (text == nullptr) return;
        apply_mode(next_text_mode(mode_of(text->fill_kind, text->fill_gradient)),
                   text->fill_kind, text->fill_gradient);
        fill_mode_->setText(mode_label(mode_of(text->fill_kind, text->fill_gradient), true));
        update_enabled();
        edited();
    });
    layout->addWidget(fill_mode_);

    fill_to_ = color_well(row, QStringLiteral("menu_fill_to"),
                          tr("A gradient's second stop. Kept while the fill is "
                             "solid, so turning a gradient off and on again does "
                             "not lose it."));
    connect(fill_to_, &QPushButton::clicked, this,
            [this] { pick_color(fill_to_, &overlay::TextItem::fill_color2); });
    layout->addWidget(fill_to_);

    fill_angle_ = spin(row, QStringLiteral("menu_fill_angle"), -180, 180,
                       QStringLiteral("°"), tr(ANGLE_TIP));
    connect(fill_angle_, &QSpinBox::valueChanged, this, [this](int value) {
        if (auto* text = current<overlay::TextItem>()) {
            text->fill_angle = value;
            edited();
        }
    });
    layout->addWidget(fill_angle_);

    // **The stroke toggle has no field of its own.** `stroke_width` is a
    // fraction of the glyph size and zero already means no stroke.
    stroke_on_ = toggle(row, tr("Stroke"), QStringLiteral("menu_stroke_on"),
                        tr("Outline the glyphs."));
    connect(stroke_on_, &QToolButton::toggled, this, [this](bool on) {
        overlay::TextItem* text = current<overlay::TextItem>();
        if (text == nullptr) return;
        text->stroke_width = on ? stroke_restore_ : 0.0;
        loading_ = true;
        stroke_width_->setValue(static_cast<int>(std::lround(text->stroke_width * 100.0)));
        loading_ = false;
        update_enabled();
        edited();
    });
    layout->addWidget(stroke_on_);

    stroke_color_ = color_well(row, QStringLiteral("menu_stroke_color"), tr("Outline colour."));
    connect(stroke_color_, &QPushButton::clicked, this,
            [this] { pick_color(stroke_color_, &overlay::TextItem::stroke_color); });
    layout->addWidget(stroke_color_);

    stroke_width_ = spin(row, QStringLiteral("menu_stroke_width"), 0, STROKE_MAX_PCT, tr("%"),
                         tr("Outline width, as a percentage of the glyph size, so it "
                            "scales with the text."));
    stroke_width_->installEventFilter(this);
    connect(stroke_width_, &QSpinBox::valueChanged, this, [this](int value) {
        overlay::TextItem* text = current<overlay::TextItem>();
        if (text == nullptr) return;
        text->stroke_width = value / 100.0;
        // What the toggle restores: the last width actually chosen.
        if (value > 0) stroke_restore_ = text->stroke_width;
        edited();
    });
    layout->addWidget(stroke_width_);

    layout->addWidget(new QLabel(tr("Rotation"), row));
    rotation_ = spin(row, QStringLiteral("menu_rotation"), -180, 180, QStringLiteral("°"),
                     tr(ROTATION_TIP));
    rotation_->installEventFilter(this);
    connect(rotation_, &QSpinBox::valueChanged, this, [this](int value) {
        if (auto* text = current<overlay::TextItem>()) {
            text->rotation = value;
            edited();
        }
    });
    layout->addWidget(rotation_);
}

void ItemMenu::build_rect_style() {
    auto* rows = new QWidget(style_menu_);
    rect_style_action_ = row_action(style_menu_, rows);
    auto* column = new QVBoxLayout(rows);
    column->setContentsMargins(0, 0, 0, 0);
    column->setSpacing(0);

    // Fill and stroke are the same shape of row, so one builder serves
    // both: the mode, the colour, the second stop and the angle, each
    // writing its own four fields.
    // One caption width for all three rows, so their controls start in
    // the same column rather than wherever each caption happens to end.
    const int caption = rows->fontMetrics().horizontalAdvance(tr("Rotation")) + 4;
    const auto captioned = [caption](const QString& text, QWidget* parent) {
        auto* label = new QLabel(text, parent);
        label->setMinimumWidth(caption);
        return label;
    };
    const auto paint_row = [this, rows, column, captioned](
                               const QString& title, const char* prefix,
                               std::string overlay::RectItem::* kind,
                               std::string overlay::RectItem::* shape,
                               std::string overlay::RectItem::* color,
                               std::string overlay::RectItem::* color2,
                               double overlay::RectItem::* angle, QPushButton*& mode,
                               QPushButton*& from, QPushButton*& to, QSpinBox*& degrees) {
        auto* row = new QWidget(rows);
        auto* layout = row_layout(row);
        layout->addWidget(captioned(title, row));
        const QString p = QLatin1String(prefix);
        mode = mode_button(row, p + QStringLiteral("_mode"),
                           tr("None, solid, or a linear or radial gradient."));
        QPushButton* mode_button_ptr = mode;
        connect(mode, &QPushButton::clicked, this, [this, kind, shape, mode_button_ptr] {
            overlay::RectItem* rect = current<overlay::RectItem>();
            if (rect == nullptr) return;
            apply_mode(next_rect_mode(mode_of(rect->*kind, rect->*shape)), rect->*kind,
                       rect->*shape);
            mode_button_ptr->setText(mode_label(mode_of(rect->*kind, rect->*shape), false));
            update_enabled();
            edited();
        });
        layout->addWidget(mode);
        from = color_well(row, p + QStringLiteral("_color"), tr("The colour, and a "
                                                                "gradient's first stop."));
        QPushButton* from_ptr = from;
        connect(from, &QPushButton::clicked, this,
                [this, from_ptr, color] { pick_color(from_ptr, color); });
        layout->addWidget(from);
        to = color_well(row, p + QStringLiteral("_to"), tr("A gradient's second stop."));
        QPushButton* to_ptr = to;
        connect(to, &QPushButton::clicked, this,
                [this, to_ptr, color2] { pick_color(to_ptr, color2); });
        layout->addWidget(to);
        degrees = spin(row, p + QStringLiteral("_angle"), -180, 180, QStringLiteral("°"),
                       tr(ANGLE_TIP));
        connect(degrees, &QSpinBox::valueChanged, this, [this, angle](int value) {
            if (auto* rect = current<overlay::RectItem>()) {
                rect->*angle = value;
                edited();
            }
        });
        layout->addWidget(degrees);
        column->addWidget(row);
        return layout;
    };

    QHBoxLayout* fill = paint_row(tr("Fill"), "menu_rect_fill", &overlay::RectItem::fill_kind,
              &overlay::RectItem::fill_gradient, &overlay::RectItem::fill_color,
              &overlay::RectItem::fill_color2, &overlay::RectItem::fill_angle,
              rect_fill_mode_, rect_fill_color_, rect_fill_to_, rect_fill_angle_);
    QHBoxLayout* stroke =
        paint_row(tr("Stroke"), "menu_rect_stroke", &overlay::RectItem::stroke_kind,
                  &overlay::RectItem::stroke_gradient, &overlay::RectItem::stroke_color,
                  &overlay::RectItem::stroke_color2, &overlay::RectItem::stroke_angle,
                  rect_stroke_mode_, rect_stroke_color_, rect_stroke_to_, rect_stroke_angle_);
    rect_stroke_width_ = spin(rows, QStringLiteral("menu_rect_stroke_width"), 0,
                              px(RECT_STROKE_MAX, overlay::CANVAS_W), tr(" px"),
                              tr("Outline width, in pixels of the 640x480 transmitted "
                                 "frame."));
    rect_stroke_width_->installEventFilter(this);
    connect(rect_stroke_width_, &QSpinBox::valueChanged, this, [this](int value) {
        if (auto* rect = current<overlay::RectItem>()) {
            rect->stroke_width = value / static_cast<double>(overlay::CANVAS_W);
            edited();
        }
    });
    stroke->addWidget(rect_stroke_width_);
    // The Fill row has no width field, so without this it spreads its
    // controls across the width the Stroke row's one extra field made.
    fill->addStretch(1);

    auto* turn = new QWidget(rows);
    auto* turn_layout = row_layout(turn);
    turn_layout->addWidget(captioned(tr("Rotation"), turn));
    rect_rotation_ = spin(turn, QStringLiteral("menu_rect_rotation"), -180, 180,
                          QStringLiteral("°"), tr(ROTATION_TIP));
    rect_rotation_->installEventFilter(this);
    connect(rect_rotation_, &QSpinBox::valueChanged, this, [this](int value) {
        if (auto* rect = current<overlay::RectItem>()) {
            rect->rotation = value;
            edited();
        }
    });
    turn_layout->addWidget(rect_rotation_);
    turn_layout->addStretch(1);
    column->addWidget(turn);
}

void ItemMenu::build_layers() {
    layers_menu_ = new QMenu(tr("&Layers"), this);
    layers_menu_->setObjectName(QStringLiteral("menu_layers"));
    addMenu(layers_menu_);
    layers_menu_->installEventFilter(this);
    auto* row = new QWidget(layers_menu_);
    row_action(layers_menu_, row);
    auto* layout = row_layout(row);

    layer_up_ = new QPushButton(row);
    layer_up_->setObjectName(QStringLiteral("menu_layer_up"));
    layer_down_ = new QPushButton(row);
    layer_down_->setObjectName(QStringLiteral("menu_layer_down"));
    // A reorder is the editor's own mutation: it re-renders and announces
    // the item's new address itself, so these write nothing through
    // `edited()`. The menu stays open, so the enabled state is refreshed
    // for the item's new position.
    connect(layer_up_, &QPushButton::clicked, this, [this] {
        if (shifted_) editor_->bring_selected_to_front();
        else editor_->raise_selected();
        update_layer_row();
    });
    connect(layer_down_, &QPushButton::clicked, this, [this] {
        if (shifted_) editor_->send_selected_to_back();
        else editor_->lower_selected();
        update_layer_row();
    });
    layout->addWidget(layer_up_);
    layout->addWidget(layer_down_);

    // Shift changes what these buttons *do*, so it has to change what
    // they say before the click. Nothing delivers a modifier change to a
    // widget without focus, and a menu's widgets take none, so this is a
    // poll -- cheap, and only while the submenu is open.
    shift_watch_ = new QTimer(this);
    shift_watch_->setObjectName(QStringLiteral("menu_shift_watch"));
    shift_watch_->setInterval(50);
    connect(shift_watch_, &QTimer::timeout, this,
            [this] { set_layer_modifiers(QApplication::keyboardModifiers()); });
    connect(layers_menu_, &QMenu::aboutToShow, this, [this] {
        set_layer_modifiers(QApplication::keyboardModifiers());
        shift_watch_->start();
    });
    connect(layers_menu_, &QMenu::aboutToHide, shift_watch_, &QTimer::stop);
}

QPushButton* ItemMenu::color_well(QWidget* parent, const QString& name, const QString& tip) {
    auto* well = new QPushButton(parent);
    well->setObjectName(name);
    well->setToolTip(tip);
    paint_swatch(well, QColor());  // its size fixed now -- see `paint_swatch`
    return well;
}

// --- editing -------------------------------------------------------------------

overlay::Item* ItemMenu::current_item() {
    // Null while the controls are being filled from an item, so their
    // change signals do not write the value straight back.
    if (loading_) return nullptr;
    return editor_->selected_item();
}

template <typename T>
T* ItemMenu::current() {
    overlay::Item* item = current_item();
    return item != nullptr ? std::get_if<T>(item) : nullptr;
}

void ItemMenu::edited() { editor_->refresh_item(); }

void ItemMenu::load_text(const overlay::TextItem& text) {
    loading_ = true;
    bold_->setChecked(text.bold);
    italic_->setChecked(text.italic);
    underline_->setChecked(text.underline);
    family_->setText(family_label(text.font_family));
    size_->setValue(px(text.size, overlay::CANVAS_H));

    paint_swatch(fill_from_, QColor(QString::fromStdString(text.color)));
    paint_swatch(fill_to_, QColor(QString::fromStdString(text.fill_color2)));
    fill_mode_->setText(mode_label(mode_of(text.fill_kind, text.fill_gradient), true));
    fill_angle_->setValue(static_cast<int>(std::lround(text.fill_angle)));

    const bool stroked = text.stroke_width > 0.0;
    stroke_on_->setChecked(stroked);
    stroke_width_->setValue(static_cast<int>(std::lround(text.stroke_width * 100.0)));
    // So the toggle restores what this item had rather than the default.
    if (stroked) stroke_restore_ = text.stroke_width;
    paint_swatch(stroke_color_, QColor(QString::fromStdString(text.stroke_color)));

    rotation_->setValue(static_cast<int>(std::lround(text.rotation)));
    loading_ = false;
}

void ItemMenu::load_rect(const overlay::RectItem& rect) {
    loading_ = true;
    rect_fill_mode_->setText(mode_label(mode_of(rect.fill_kind, rect.fill_gradient), false));
    paint_swatch(rect_fill_color_, QColor(QString::fromStdString(rect.fill_color)));
    paint_swatch(rect_fill_to_, QColor(QString::fromStdString(rect.fill_color2)));
    rect_fill_angle_->setValue(static_cast<int>(std::lround(rect.fill_angle)));
    rect_stroke_mode_->setText(
        mode_label(mode_of(rect.stroke_kind, rect.stroke_gradient), false));
    paint_swatch(rect_stroke_color_, QColor(QString::fromStdString(rect.stroke_color)));
    paint_swatch(rect_stroke_to_, QColor(QString::fromStdString(rect.stroke_color2)));
    rect_stroke_angle_->setValue(static_cast<int>(std::lround(rect.stroke_angle)));
    rect_stroke_width_->setValue(px(rect.stroke_width, overlay::CANVAS_W));
    rect_rotation_->setValue(static_cast<int>(std::lround(rect.rotation)));
    loading_ = false;
}

void ItemMenu::update_enabled() {
    const overlay::Item* item = editor_->selected_item();
    if (item == nullptr) return;
    if (const auto* text = std::get_if<overlay::TextItem>(item)) {
        const Mode mode = mode_of(text->fill_kind, text->fill_gradient);
        fill_from_->setEnabled(mode != Mode::None);
        fill_to_->setEnabled(is_gradient(mode));
        fill_angle_->setEnabled(mode == Mode::Linear);
        const bool stroked = text->stroke_width > 0.0;
        stroke_color_->setEnabled(stroked);
        stroke_width_->setEnabled(stroked);
    } else if (const auto* rect = std::get_if<overlay::RectItem>(item)) {
        const Mode fill = mode_of(rect->fill_kind, rect->fill_gradient);
        rect_fill_color_->setEnabled(fill != Mode::None);
        rect_fill_to_->setEnabled(is_gradient(fill));
        rect_fill_angle_->setEnabled(fill == Mode::Linear);
        const Mode stroke = mode_of(rect->stroke_kind, rect->stroke_gradient);
        rect_stroke_color_->setEnabled(stroke != Mode::None);
        rect_stroke_to_->setEnabled(is_gradient(stroke));
        rect_stroke_angle_->setEnabled(stroke == Mode::Linear);
        rect_stroke_width_->setEnabled(stroke != Mode::None);
    }
}

void ItemMenu::update_layer_row() {
    layer_up_->setText(shifted_ ? tr("Bring to front") : tr("Bring forward"));
    layer_down_->setText(shifted_ ? tr("Send to back") : tr("Send backward"));
    layer_up_->setEnabled(editor_->can_raise_selected());
    layer_down_->setEnabled(editor_->can_lower_selected());
}

void ItemMenu::set_layer_modifiers(Qt::KeyboardModifiers modifiers) {
    const bool shifted = (modifiers & Qt::ShiftModifier) != Qt::NoModifier;
    if (shifted == shifted_) return;
    shifted_ = shifted;
    update_layer_row();
}

template <typename T>
void ItemMenu::pick_color(QAbstractButton* well, std::string T::* field) {
    const T* before = current<T>();
    if (before == nullptr) return;
    const QColor start(QString::fromStdString(before->*field));

    // **Close the menu first.** The dialog is modal and this menu holds a
    // mouse grab; releasing it here rather than relying on the style
    // keeps the behaviour the same on all three platforms. The dialog
    // runs its own event loop, which is why the item is resolved again
    // below rather than held across it.
    hide();
    const QColor chosen = QColorDialog::getColor(start, parentWidget());
    if (!chosen.isValid()) return;
    T* item = current<T>();
    if (item == nullptr) return;
    item->*field = chosen.name().toStdString();
    paint_swatch(well, chosen);
    edited();
}

void ItemMenu::popup_for(overlay::Item* item, const QPoint& global_pos) {
    // The editor selects what was right-clicked before it emits, so this
    // holds by construction. Refused rather than guessed at when it does
    // not: editing an item other than the selected one is silently wrong.
    if (item == nullptr || item != editor_->selected_item()) return;

    const auto* text = std::get_if<overlay::TextItem>(item);
    const auto* rect = std::get_if<overlay::RectItem>(item);
    format_menu_->menuAction()->setVisible(text != nullptr);
    style_menu_->menuAction()->setVisible(text != nullptr || rect != nullptr);
    // The widgets as well as the actions: `QMenu` skips a hidden action
    // when it lays itself out and never hides its widget, so a row shown
    // once would stay painted over the other kind's rows.
    for (auto [action, offered] :
         {std::pair{text_style_action_, text != nullptr},
          std::pair{rect_style_action_, rect != nullptr}}) {
        action->setVisible(offered);
        action->defaultWidget()->setVisible(offered);
    }
    if (text != nullptr) load_text(*text);
    if (rect != nullptr) load_rect(*rect);
    update_enabled();
    update_layer_row();
    popup(global_pos);
}

// --- the wheel -----------------------------------------------------------------

void ItemMenu::step_rotation(QSpinBox* field, int notches, bool coarse) {
    if (!coarse) {
        field->setValue(field->value() + notches);
        return;
    }
    // Shift snaps to the next multiple of 15 in the direction of travel
    // -- a destination rather than a step, so a rotation nudged to 7
    // degrees comes back to 15 or to 0 rather than to 22 or -8.
    const int direction = notches > 0 ? 1 : -1;
    for (int i = 0; i < std::abs(notches); ++i) {
        const double turns = field->value() / 15.0;
        const int next = direction > 0 ? static_cast<int>(std::floor(turns)) * 15 + 15
                                       : static_cast<int>(std::ceil(turns)) * 15 - 15;
        field->setValue(next);
    }
}

bool ItemMenu::eventFilter(QObject* watched, QEvent* event) {
    if (event->type() != QEvent::Wheel) return QMenu::eventFilter(watched, event);
    auto* wheel = static_cast<QWheelEvent*>(event);

    // A wheel over a submenu arrives at the menu, which would scroll
    // itself; a wheel over a field may arrive at the field or at the line
    // edit inside it. Resolve to a widget, then walk up.
    QWidget* target = qobject_cast<QWidget*>(watched);
    if (auto* menu = qobject_cast<QMenu*>(watched)) {
        target = menu->childAt(wheel->position().toPoint());
    }
    const int notches = notches_of(wheel);
    const bool coarse = (wheel->modifiers() & Qt::ShiftModifier) != Qt::NoModifier;
    for (QWidget* widget = target; widget != nullptr; widget = widget->parentWidget()) {
        if (widget == size_) {
            if (notches != 0) size_->setValue(size_->value() + notches * (coarse ? 10 : 1));
        } else if (widget == rotation_ || widget == rect_rotation_) {
            if (notches != 0) step_rotation(static_cast<QSpinBox*>(widget), notches, coarse);
        } else if (widget == stroke_width_ || widget == rect_stroke_width_) {
            auto* field = static_cast<QSpinBox*>(widget);
            if (notches != 0) field->setValue(field->value() + notches);
        } else {
            continue;
        }
        event->accept();
        return true;
    }

    // Anywhere else on a submenu: swallowed. These menus are a row or
    // three and never scrollable, so the only thing a wheel could do
    // here is move the menu out from under the pointer.
    if (qobject_cast<QMenu*>(watched) != nullptr) {
        event->accept();
        return true;
    }
    return QMenu::eventFilter(watched, event);
}

}  // namespace sstvae::gui
