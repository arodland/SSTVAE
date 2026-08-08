#include "settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPalette>
#include <QFrame>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QStringList>
#include <QTabWidget>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "audio/qt/qtaudio.hpp"
#include "checkpoint/checkpoint.hpp"
#include "rig/backend.hpp"
#include "rig/hamlib.hpp"
#include "rig_config.hpp"
#include "style.hpp"

namespace sstvae::gui {

namespace {

// A line edit with a Browse button, for a directory.
QWidget* folder_row(QLineEdit** out, const QString& value, const QString& caption,
                    QWidget* parent) {
    auto* holder = new QWidget(parent);
    auto* layout = new QHBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    auto* edit = new QLineEdit(value, holder);
    auto* browse = new QPushButton(QObject::tr("Browse..."), holder);
    QObject::connect(browse, &QPushButton::clicked, holder, [edit, caption, holder] {
        const QString path =
            QFileDialog::getExistingDirectory(holder, caption, edit->text());
        if (!path.isEmpty()) edit->setText(path);
    });
    layout->addWidget(edit, 1);
    layout->addWidget(browse);
    *out = edit;
    return holder;
}

// A combo over fixed (value, label) pairs, selecting `current`.
QComboBox* choice(QWidget* parent,
                  std::initializer_list<std::pair<const char*, const char*>> items,
                  const std::string& current) {
    auto* combo = new QComboBox(parent);
    for (const auto& [value, label] : items) {
        combo->addItem(QObject::tr(label), QString::fromLatin1(value));
    }
    const int index = combo->findData(QString::fromStdString(current));
    combo->setCurrentIndex(index >= 0 ? index : 0);
    return combo;
}

std::string chosen(const QComboBox* combo) {
    return combo->currentData().toString().toStdString();
}

// A combo that is only as wide as it needs to be. In a QFormLayout the
// field column stretches, which left a three-item combo as wide as the
// dialog with its options huddled at one end.
QComboBox* compact(QComboBox* combo) {
    combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    combo->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    return combo;
}

// A tab page that scrolls rather than clips.
//
// The rig tab is taller than the dialog opens at, and a QFormLayout
// given too little height does not compress -- it truncates, and what
// goes first is the wrapped grey help text at the bottom of each
// section. Clipping is worse than scrolling in both directions: the
// operator cannot tell whether the text is cut off or simply ends, and
// there is no size the dialog can default to that is right on a laptop
// panel and on a 4K monitor. A scroll area makes the question moot.
QWidget* scrolling(QWidget* page) {
    auto* area = new QScrollArea(page->parentWidget());
    area->setWidget(page);
    area->setWidgetResizable(true);
    area->setFrameShape(QFrame::NoFrame);
    // Never sideways: the width is the dialog's, and a horizontal bar
    // would mean a label is refusing to wrap.
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    return area;
}

// One artifact "Download All" fetches, with the label its report uses.
struct ModelPart {
    std::string_view part;
    const char* label;
};

const std::vector<ModelPart>& model_parts() {
    static const std::vector<ModelPart> parts = {
        {"encoder", "Encoder"},
        {"decoder", "Decoder"},
        {checkpoint::GRAD_PART, "Refinement gradient graph"},
    };
    return parts;
}

}  // namespace

SettingsDialog::SettingsDialog(const settings::Config& config, QWidget* parent)
    : QDialog(parent), config_(config) {
    setWindowTitle(tr("SSTVAE settings"));
    // Opens tall enough for the rig tab, the tallest, so the common case
    // needs no scrolling at all -- but sized rather than *pinned* there,
    // because a minimum that does not fit a small laptop panel is its
    // own bug. The scroll areas make anything smaller merely awkward
    // instead of broken.
    resize(640, 700);

    auto* tabs = new QTabWidget(this);
    // Ordered by how often an operator has to touch them, not by how the
    // config file is structured. Folders and the model are the two that
    // are set once and then never again -- the model's correct setting
    // is "blank" for the life of the station -- so they go at the end;
    // the callsign, which is the one field nobody can leave alone, moved
    // to the top of Transmit, next to the CW ID that sends it.
    tabs->addTab(scrolling(audio_tab()), tr("Audio"));
    tabs->addTab(scrolling(rig_tab()), tr("Rig control"));
    tabs->addTab(scrolling(receive_tab()), tr("Receive"));
    tabs->addTab(scrolling(transmit_tab()), tr("Transmit"));
    tabs->addTab(scrolling(folders_tab()), tr("Folders"));
    tabs->addTab(scrolling(model_tab()), tr("Model"));

    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(tabs);
    layout->addWidget(buttons);

    connect(this, &SettingsDialog::rigTestFinished, this,
            &SettingsDialog::on_rig_test_finished, Qt::QueuedConnection);
    connect(this, &SettingsDialog::modelDownloadFinished, this,
            &SettingsDialog::on_model_download_finished, Qt::QueuedConnection);
}

SettingsDialog::~SettingsDialog() = default;

// --- model ------------------------------------------------------------------

QWidget* SettingsDialog::model_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    model_path_ = new QLineEdit(QString::fromStdString(config_.model_path), page);
    model_path_->setPlaceholderText(tr("(published model)"));
    connect(model_path_, &QLineEdit::textChanged, this,
            &SettingsDialog::sync_precision_enabled);
    auto* browse_dir = new QPushButton(tr("Browse folder..."), page);
    connect(browse_dir, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getExistingDirectory(
            this, tr("Folder of exported .onnx files"), QDir::homePath());
        if (!path.isEmpty()) model_path_->setText(path);
    });
    auto* browse_file = new QPushButton(tr("Browse file..."), page);
    connect(browse_file, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Model file"), QDir::homePath(),
            tr("Model files (*.onnx *.pt *.ckpt);;ONNX (*.onnx);;"
               "Checkpoints (*.pt *.ckpt)"));
        if (!path.isEmpty()) model_path_->setText(path);
    });
    auto* model_row = new QWidget(page);
    auto* model_layout = new QHBoxLayout(model_row);
    model_layout->setContentsMargins(0, 0, 0, 0);
    model_layout->addWidget(model_path_, 1);
    model_layout->addWidget(browse_dir);
    model_layout->addWidget(browse_file);
    form->addRow(tr("Model"), model_row);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Leave blank for the published model, downloaded once "
                        "and cached."),
                     tr("Otherwise a folder of exported .onnx files, or a "
                        "single .onnx. Both stations must run the same model to "
                        "exchange images — but not the same precision, which is "
                        "a local choice."),
                     page));

    precision_ = new QComboBox(page);
    for (const std::string_view p : checkpoint::PRECISIONS) {
        const QString name = QString::fromUtf8(p.data(), static_cast<int>(p.size()));
        precision_->addItem(name, name);
    }
    const int index = precision_->findData(QString::fromStdString(config_.precision));
    precision_->setCurrentIndex(index >= 0 ? index : 0);
    form->addRow(tr("Precision"), compact(precision_));
    precision_note_ = style::note(QString(), page);
    form->addRow(QString(), precision_note_);
    sync_precision_enabled();

    // Centred and set apart, like the rig tab's test buttons: everything
    // else on this tab takes effect on OK, but this one does something --
    // real network I/O -- the moment it is pressed.
    download_all_ = new QPushButton(tr("Download all models"), page);
    connect(download_all_, &QPushButton::clicked, this,
            &SettingsDialog::download_all_models);
    auto* download_row = new QWidget(page);
    auto* download_layout = new QHBoxLayout(download_row);
    download_layout->setContentsMargins(0, 14, 0, 0);
    download_layout->addStretch(1);
    download_layout->addWidget(download_all_);
    download_layout->addStretch(1);
    form->addRow(download_row);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Caches a complete set now, before going somewhere with "
                        "no connection."),
                     tr("Otherwise each artifact is only fetched the first time "
                        "it is actually needed — the encoder on the first Send, "
                        "the gradient graph on the first refined transmission."),
                     page));
    return page;
}

void SettingsDialog::sync_precision_enabled() {
    // Precision only means something for a folder or the published
    // model. A .onnx filename already names its precision -- picking
    // v1-encoder-int8.onnx selects int8 whatever this combo says, and
    // the decoder is resolved beside it at the same precision. Greying
    // the control out shows the user the same rule
    // `settings::codec_precision` applies.
    const QString suffix =
        QFileInfo(model_path_->text().trimmed()).suffix().toLower();
    QString text;
    if (suffix == QLatin1String("onnx")) {
        text = tr("Set by the file name — that artifact is already one precision.");
    } else if (suffix == QLatin1String("pt") || suffix == QLatin1String("ckpt")) {
        text = tr("Not applicable to a .pt checkpoint (that runs on torch, which "
                  "this app does not embed).");
    } else {
        text = tr("fp16 is the default and measures identical to fp32. Purely "
                  "local: it never has to match the far end.");
    }
    precision_->setEnabled(suffix != QLatin1String("onnx") &&
                           suffix != QLatin1String("pt") &&
                           suffix != QLatin1String("ckpt"));
    precision_note_->setText(text);
}

void SettingsDialog::set_download_busy(bool busy) {
    download_all_->setEnabled(!busy);
    download_all_->setText(busy ? tr("Downloading...") : tr("Download all models"));
}

void SettingsDialog::download_all_models() {
    // On a worker thread, for the same reason as the rig tests: a fetch
    // of the published model is a real HTTP download, and the GUI
    // thread must not block on the network any more than it blocks on
    // the rig. The path/precision are read from the *pending* values in
    // the dialog, not `config_`, so pointing the picker at a folder and
    // pressing this button downloads nothing -- resolving those parts is
    // pure local path arithmetic.
    set_download_busy(true);
    const std::string path = model_path_->text().trimmed().toStdString();
    const std::string precision = precision_->currentData().toString().toStdString();

    std::thread([this, path, precision] {
        QStringList lines;
        bool ok = true;
        for (const ModelPart& part : model_parts()) {
            const bool was_cached =
                path.empty() &&
                checkpoint::find_cached(checkpoint::onnx_filename(part.part, precision))
                    .has_value();
            try {
                checkpoint::resolve_onnx(part.part, path, precision);
                if (path.empty()) {
                    lines << tr("%1: %2").arg(
                        QString::fromUtf8(part.label),
                        was_cached ? tr("already cached") : tr("downloaded"));
                } else {
                    lines << tr("%1: ready").arg(QString::fromUtf8(part.label));
                }
            } catch (const std::exception& e) {
                ok = false;
                lines << tr("%1: FAILED\n%2").arg(QString::fromUtf8(part.label),
                                                   QString::fromUtf8(e.what()));
            }
        }
        emit modelDownloadFinished(ok, lines.join(QStringLiteral("\n\n")));
    }).detach();
}

void SettingsDialog::on_model_download_finished(bool ok, const QString& message) {
    set_download_busy(false);
    if (ok) {
        QMessageBox::information(
            this, tr("Models"), tr("All model artifacts are ready.\n\n%1").arg(message));
    } else {
        QMessageBox::warning(
            this, tr("Models"),
            tr("Some artifacts could not be fetched.\n\n%1").arg(message));
    }
}

// --- audio ------------------------------------------------------------------

QWidget* SettingsDialog::audio_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    // No backend picker. The reference has one because it carries a
    // PortAudio path as well; here there is only QtMultimedia, and a
    // control with one option is worse than no control. `audio.backend`
    // stays in the config rather than being deleted, so a config shared
    // with the Python app round-trips.
    input_device_ = new QComboBox(page);
    output_device_ = new QComboBox(page);
    fill_device_combo(input_device_, true, config_.audio.input_device);
    fill_device_combo(output_device_, false, config_.audio.output_device);
    form->addRow(tr("Input (from radio)"), input_device_);
    form->addRow(tr("Output (to radio)"), output_device_);

    // Enumeration happens once, when the dialog opens. Plugging in a USB
    // interface -- or loading the loopback modules below -- is exactly
    // the sort of thing done *while* this dialog is open, and closing
    // and reopening it to see the result is a poor answer.
    auto* refresh = new QPushButton(tr("Refresh"), page);
    connect(refresh, &QPushButton::clicked, this, &SettingsDialog::refresh_devices);
    form->addRow(QString(), style::row(page, {refresh}));
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Qt does not list PulseAudio/PipeWire monitor sources, "
                        "so a loopback has to be published as a real source."),
                     tr("  pactl load-module module-null-sink "
                        "sink_name=null-sink\n"
                        "  pactl load-module module-remap-source "
                        "source_name=sstvae_loop master=null-sink.monitor "
                        "channels=1"),
                     page));

    form->addRow(QString(),
                 style::note(tr("Transmit level is on the Transmit panel, not here — "
                                "setting it means watching the radio's ALC while "
                                "sending, so no modal dialog may be in the way."),
                             page));
    return page;
}

void SettingsDialog::refresh_devices() {
    // Re-enumerate, keeping whatever is selected now rather than what
    // was in the config when the dialog opened -- otherwise Refresh
    // quietly discards a choice the operator just made.
    fill_device_combo(input_device_, true,
                      input_device_->currentData().toString().toStdString());
    fill_device_combo(output_device_, false,
                      output_device_->currentData().toString().toStdString());
}

void SettingsDialog::fill_device_combo(QComboBox* combo, bool input,
                                       const std::string& current) {
    combo->clear();
    combo->addItem(tr("System default"), QString());
    try {
        const std::vector<std::string> names =
            input ? audio::qt::input_device_names() : audio::qt::output_device_names();
        for (const std::string& name : names) {
            combo->addItem(QString::fromStdString(name), QString::fromStdString(name));
        }
    } catch (const std::exception& e) {
        combo->addItem(tr("(audio unavailable: %1)").arg(QString::fromUtf8(e.what())),
                       QString());
        combo->setEnabled(false);
    }
    if (current.empty()) return;

    int index = combo->findData(QString::fromStdString(current));
    if (index < 0) {
        // A device that has gone away. Keep it rather than silently
        // switching the user to the default -- they may just have it
        // unplugged.
        combo->addItem(tr("%1 (not found)").arg(QString::fromStdString(current)),
                       QString::fromStdString(current));
        index = combo->count() - 1;
    }
    combo->setCurrentIndex(index);
}

// --- rig --------------------------------------------------------------------

QWidget* SettingsDialog::rig_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);
    const settings::RigConfig& rig = config_.rig;

    rig_enabled_ = new QCheckBox(tr("Use rig control (PTT and frequency)"), page);
    rig_enabled_->setChecked(rig.enabled);
    form->addRow(rig_enabled_);

    // Editable on purpose: the list can fail to load, and a config
    // written by a different Hamlib may name a model this one does not
    // list. The number must still be typeable, and a saved value must
    // survive a round trip rather than being silently reset.
    rig_model_ = new QComboBox(page);
    rig_model_->setEditable(true);
    rig_model_->setInsertPolicy(QComboBox::NoInsert);
    QString model_error;
    try {
        for (const rig::RigModel& model : rig::list_models()) {
            rig_model_->addItem(QString::fromStdString(model.label()), model.model);
        }
    } catch (const std::exception& e) {
        model_error = QString::fromUtf8(e.what()).section(QLatin1Char('\n'), 0, 0);
    }
    // 300-odd entries labelled "<mfg> <model>". Qt's default completer
    // anchors at the start, so a user typing the only part they know
    // ("FT-847", "IC-7300") would match nothing; match anywhere instead.
    if (QCompleter* completer = rig_model_->completer()) {
        completer->setCaseSensitivity(Qt::CaseInsensitive);
        completer->setFilterMode(Qt::MatchContains);
    }
    const int model_index = rig_model_->findData(rig.model);
    if (model_index >= 0) {
        rig_model_->setCurrentIndex(model_index);
    } else {
        rig_model_->setEditText(QString::number(rig.model));
    }
    form->addRow(tr("Rig"), rig_model_);
    if (!model_error.isEmpty()) {
        rig_model_note_ = style::note(tr("%1 — enter a model number by hand.").arg(model_error),
                               page);
        form->addRow(QString(), rig_model_note_);
    }

    // "Device", not "serial port": NET rigctl takes host:port, and so do
    // a growing number of radios with an Ethernet or Wi-Fi interface.
    rig_device_ = new QLineEdit(QString::fromStdString(rig.device), page);
    rig_device_->setPlaceholderText(tr("/dev/ttyUSB0, COM5, or host:port"));
    form->addRow(tr("Device"), rig_device_);
    form->addRow(QString(),
                 style::note(tr("A serial port, or host:port for a networked radio or "
                         "for model 2 (\"NET rigctl\"), which shares one radio "
                         "with WSJT-X or fldigi."),
                      page));

    // Five one-per-row combos became two rows: these are the settings
    // almost nobody changes, and giving each its own row pushed the
    // controls that matter off the bottom of the dialog.
    rig_baud_ = compact(new QComboBox(page));
    rig_baud_->addItem(tr("Default"), 0);
    for (const int rate : {1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200}) {
        rig_baud_->addItem(QString::number(rate), rate);
    }
    const int baud_index = rig_baud_->findData(rig.baud);
    rig_baud_->setCurrentIndex(baud_index >= 0 ? baud_index : 0);

    data_bits_ = compact(choice(
        page, {{"default", "Default"}, {"seven", "7"}, {"eight", "8"}}, rig.data_bits));
    stop_bits_ = compact(choice(
        page, {{"default", "Default"}, {"one", "1"}, {"two", "2"}}, rig.stop_bits));
    parity_ = compact(choice(page,
                             {{"default", "Default"},
                              {"none", "None"},
                              {"odd", "Odd"},
                              {"even", "Even"}},
                             rig.parity));
    handshake_ = compact(choice(page,
                                {{"default", "Default"},
                                 {"none", "None"},
                                 {"xonxoff", "XON/XOFF"},
                                 {"hardware", "Hardware"}},
                                rig.handshake));

    // **Every control in a compound row is labelled, including the
    // first.** These read "Baud / bits [combo] Data [combo] Stop
    // [combo]", so two of the three were named inline and the first was
    // named only by the row -- which you have to work out from position.
    form->addRow(tr("Serial"),
                 style::row(page, {new QLabel(tr("Baud"), page), rig_baud_,
                                   new QLabel(tr("Data"), page), data_bits_,
                                   new QLabel(tr("Stop"), page), stop_bits_}));
    form->addRow(QString(),
                 style::row(page, {new QLabel(tr("Parity"), page), parity_,
                                   new QLabel(tr("Handshake"), page), handshake_}));

    // A checkbox plus a value, rather than a three-item combo whose
    // first entry was "Default". "Default" reads as a *level* next to
    // High and Low, when what it actually means is "do not touch this
    // line at all" -- a different kind of answer, so a different control.
    dtr_forced_ = new QCheckBox(tr("DTR"), page);
    dtr_ = compact(choice(page, {{"high", "High"}, {"low", "Low"}},
                          rig.dtr == "default" ? "high" : rig.dtr));
    dtr_forced_->setChecked(rig.dtr != "default");
    connect(dtr_forced_, &QCheckBox::toggled, dtr_, &QWidget::setEnabled);
    dtr_->setEnabled(dtr_forced_->isChecked());

    rts_forced_ = new QCheckBox(tr("RTS"), page);
    rts_ = compact(choice(page, {{"high", "High"}, {"low", "Low"}},
                          rig.rts == "default" ? "high" : rig.rts));
    rts_forced_->setChecked(rig.rts != "default");
    connect(rts_forced_, &QCheckBox::toggled, rts_, &QWidget::setEnabled);
    rts_->setEnabled(rts_forced_->isChecked());

    form->addRow(tr("Force control lines"),
                 style::row(page, {dtr_forced_, dtr_, rts_forced_, rts_}));
    form->addRow(QString(),
                 style::note(tr("Held for the whole session, which is how an interface "
                         "powered from the control lines stays fed."),
                      page));

    ptt_method_ = compact(choice(
        page, {{"vox", "VOX"}, {"cat", "CAT"}, {"dtr", "DTR"}, {"rts", "RTS"}},
        rig.ptt_method));
    connect(ptt_method_, &QComboBox::currentIndexChanged, this,
            &SettingsDialog::sync_ptt_enabled);
    ptt_device_ = new QLineEdit(QString::fromStdString(rig.ptt_device), page);
    ptt_device_->setPlaceholderText(tr("(the device above)"));
    form->addRow(tr("PTT"),
                 style::row(page, {ptt_method_, new QLabel(tr("Port"), page), ptt_device_}, 2));
    form->addRow(QString(),
                 style::note(tr("VOX means do not key at all — the radio is keyed "
                                "by the audio. DTR and RTS may use a different "
                                "port from CAT."),
                             page));
    sync_ptt_enabled();

    rig_mode_ = compact(choice(
        page, {{"none", "None"}, {"usb", "USB"}, {"pkt_usb", "Data/Pkt"}}, rig.mode));
    form->addRow(tr("Mode on connect"), style::row(page, {rig_mode_}));

    poll_interval_s_ = new QDoubleSpinBox(page);
    poll_interval_s_->setRange(0.5, 60.0);
    poll_interval_s_->setSingleStep(0.5);
    poll_interval_s_->setSuffix(tr(" s"));
    poll_interval_s_->setValue(rig.poll_interval_s);
    ptt_lead_ = new QDoubleSpinBox(page);
    ptt_lead_->setRange(0.0, 3.0);
    ptt_lead_->setSingleStep(0.05);
    ptt_lead_->setSuffix(tr(" s"));
    ptt_lead_->setValue(rig.ptt_lead_s);
    ptt_tail_ = new QDoubleSpinBox(page);
    ptt_tail_->setRange(0.0, 3.0);
    ptt_tail_->setSingleStep(0.05);
    ptt_tail_->setSuffix(tr(" s"));
    ptt_tail_->setValue(rig.ptt_tail_s);
    form->addRow(tr("Timing"),
                 style::row(page, {new QLabel(tr("Poll"), page), poll_interval_s_,
                                   new QLabel(tr("Lead"), page), ptt_lead_,
                                   new QLabel(tr("Tail"), page), ptt_tail_}));

    // Centred, and set apart from the rows above. Everything else on
    // this tab is a setting that takes effect on OK; these two *do*
    // something the moment they are pressed, to a radio that may be
    // connected to an antenna. Left-aligned in the field column they
    // read as one more row of controls.
    test_cat_ = new QPushButton(tr("Test CAT"), page);
    connect(test_cat_, &QPushButton::clicked, this, &SettingsDialog::test_cat);
    test_ptt_ = new QPushButton(tr("Test PTT (0.5 s)"), page);
    connect(test_ptt_, &QPushButton::clicked, this, &SettingsDialog::test_ptt);

    auto* tests = new QWidget(page);
    auto* tests_layout = new QHBoxLayout(tests);
    tests_layout->setContentsMargins(0, 14, 0, 0);
    tests_layout->addStretch(1);
    tests_layout->addWidget(test_cat_);
    tests_layout->addWidget(test_ptt_);
    tests_layout->addStretch(1);
    form->addRow(tests);
    return page;
}

void SettingsDialog::sync_filename_preview() {
    // The same function that actually names the file, with sample
    // fields -- so the preview cannot describe a rule the saver does
    // not follow.
    settings::FilenameFields sample;
    sample.callsign = "KD8XYZ";
    sample.mode = "B";
    sample.freq_hz = 14'230'000.0;
    const std::string name = settings::format_filename(
        filename_template_->text().toStdString(), sample);
    filename_preview_->setText(
        name.empty() ? tr("(empty — the reception would be saved unnamed)")
                     : tr("e.g. %1.png").arg(QString::fromStdString(name)));
}

void SettingsDialog::sync_ptt_enabled() {
    const std::string method = chosen(ptt_method_);
    ptt_device_->setEnabled(method == "dtr" || method == "rts");
}

int SettingsDialog::rig_model_number() const {
    // A chosen item carries its number as item data. Free text is either
    // a bare number or a label that was typed or completed to match an
    // item, so fall back to matching the text, then to the number in
    // brackets of a label like "Yaesu FT-847 (1001)".
    const QString text = rig_model_->currentText().trimmed();
    const int index = rig_model_->findText(text);
    if (index >= 0) {
        const QVariant data = rig_model_->itemData(index);
        if (data.isValid()) return data.toInt();
    }
    bool ok = false;
    const int direct = text.toInt(&ok);
    if (ok) return direct;

    const int open = text.lastIndexOf(QLatin1Char('('));
    const int close = text.lastIndexOf(QLatin1Char(')'));
    if (open >= 0 && close > open) {
        const int parsed = text.mid(open + 1, close - open - 1).toInt(&ok);
        if (ok) return parsed;
    }
    return 1;  // the dummy rig: harmless, and obviously not a real radio
}

settings::RigConfig SettingsDialog::pending_rig() const {
    settings::RigConfig rig = config_.rig;
    rig.enabled = rig_enabled_->isChecked();
    rig.model = rig_model_number();
    rig.device = rig_device_->text().trimmed().toStdString();
    rig.baud = rig_baud_->currentData().toInt();
    rig.data_bits = chosen(data_bits_);
    rig.stop_bits = chosen(stop_bits_);
    rig.parity = chosen(parity_);
    rig.handshake = chosen(handshake_);
    rig.dtr = dtr_forced_->isChecked() ? chosen(dtr_) : std::string("default");
    rig.rts = rts_forced_->isChecked() ? chosen(rts_) : std::string("default");
    rig.ptt_method = chosen(ptt_method_);
    rig.ptt_device = ptt_device_->text().trimmed().toStdString();
    rig.mode = chosen(rig_mode_);
    rig.poll_interval_s = poll_interval_s_->value();
    rig.ptt_lead_s = ptt_lead_->value();
    rig.ptt_tail_s = ptt_tail_->value();
    return rig;
}

void SettingsDialog::test_cat() { run_rig_test(false); }
void SettingsDialog::test_ptt() { run_rig_test(true); }

void SettingsDialog::set_rig_test_busy(bool busy) {
    test_cat_->setEnabled(!busy);
    test_ptt_->setEnabled(!busy);
}

void SettingsDialog::run_rig_test(bool key_ptt) {
    // On a worker thread, not here. A radio that is powered off costs
    // the timeout, and "nothing on the GUI thread ever blocks on the
    // rig" is not a rule with an exception for test buttons -- it is the
    // rule that stops the window freezing. The buttons are disabled
    // until the answer arrives, so there is no second request in flight.
    set_rig_test_busy(true);
    const settings::RigConfig config = pending_rig();

    std::thread([this, config, key_ptt] {
        try {
            std::unique_ptr<rig::RigBackend> backend = make_backend(config);
            backend->open();
            QString message;
            if (key_ptt) {
                backend->set_ptt(true);
                QThread::msleep(500);
                backend->set_ptt(false);
                message = tr("PTT keyed and released.");
            } else {
                const double hz = backend->frequency_hz();
                message = tr("Connected to %1.\nDial frequency: %2 MHz")
                              .arg(QString::fromStdString(backend->description()))
                              .arg(hz / 1e6, 0, 'f', 4);
            }
            backend->close();
            emit rigTestFinished(true, message);
        } catch (const std::exception& e) {
            emit rigTestFinished(false, QString::fromUtf8(e.what()));
        }
    }).detach();
}

void SettingsDialog::on_rig_test_finished(bool ok, const QString& message) {
    set_rig_test_busy(false);
    if (ok) {
        QMessageBox::information(this, tr("Rig control"), message);
    } else {
        QMessageBox::warning(this, tr("Rig control"), message);
    }
}

// --- folders ----------------------------------------------------------------

QWidget* SettingsDialog::folders_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);
    const settings::FolderConfig& folders = config_.folders;
    form->addRow(tr("Received images"),
                 folder_row(&receive_dir_,
                            QString::fromStdString(folders.receive_dir),
                            tr("Received images"), page));
    form->addRow(tr("Images to send"),
                 folder_row(&transmit_dir_,
                            QString::fromStdString(folders.transmit_dir),
                            tr("Images to send"), page));
    form->addRow(tr("Overlay templates"),
                 folder_row(&template_dir_,
                            QString::fromStdString(folders.template_dir),
                            tr("Overlay templates"), page));
    form->addRow(QString(),
                 style::note(tr("Saving and reusing overlay templates is not "
                                "implemented yet; this is where they will go."),
                             page));
    return page;
}

// --- receive ----------------------------------------------------------------

QWidget* SettingsDialog::receive_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);
    const settings::ReceiveConfig& receive = config_.receive;

    autosave_ = new QCheckBox(tr("Save every completed reception automatically"),
                              page);
    autosave_->setChecked(receive.autosave);
    form->addRow(autosave_);

    save_audio_ = new QCheckBox(tr("Also save the captured audio (diagnostic)"),
                                page);
    save_audio_->setChecked(receive.save_audio);
    form->addRow(save_audio_);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Writes a .wav beside each received image, exactly as "
                        "captured."),
                     tr("Use it when an image decodes badly: run sstvae-decode "
                        "on the dump to see whether the audio or the decoder "
                        "was at fault."),
                     page));

    low_cpu_ = new QCheckBox(tr("Low-CPU mode"), page);
    low_cpu_->setChecked(receive.low_cpu);
    form->addRow(low_cpu_);
    form->addRow(QString(),
                 style::note(tr("Low-CPU mode only looks for the start of a "
                                "transmission, so it cannot pick up one already in "
                                "progress or decode retrospectively."),
                             page));

    filename_template_ =
        new QLineEdit(QString::fromStdString(receive.filename_template), page);
    form->addRow(tr("Filename"), filename_template_);
    // **A live preview, not a list of fields to imagine.** The template
    // has five substitutions and the operator was expected to hold the
    // result in their head; `settings::format_filename` is right here
    // and is the same function that names the file.
    filename_preview_ = style::note(QString(), page);
    connect(filename_template_, &QLineEdit::textChanged, this,
            &SettingsDialog::sync_filename_preview);
    form->addRow(QString(), filename_preview_);
    sync_filename_preview();
    form->addRow(QString(),
                 style::note(tr("Fields: {date} {time} {freq} {callsign} {mode}. "
                                "Fields with no value are dropped from the name."),
                             page));

    save_size_ = new QLineEdit(QString::fromStdString(receive.save_size), page);
    save_size_->setPlaceholderText(tr("640x480 (as received)"));
    // **Validated, because the failure was silent and permanent.**
    // `rx::parse_size` returns nothing for anything it does not
    // understand and the caller then just does not resize -- so
    // "640 x 480" or "640x480 " was accepted by the dialog, saved to
    // the config, and quietly ignored on every reception thereafter.
    // Empty stays legal: it means "as received".
    save_size_->setValidator(new QRegularExpressionValidator(
        QRegularExpression(QStringLiteral("^(\\d{1,5}x\\d{1,5})?$")), save_size_));
    form->addRow(tr("Saved size"), save_size_);

    // `setSuffix(" s")` and a bare label, which is what the Rig tab
    // does -- the two tabs used to state the same unit two ways, one in
    // the field and one in parentheses after the label. `compact`,
    // because a three-digit number does not want the whole field column.
    buffer_seconds_ = new QDoubleSpinBox(page);
    buffer_seconds_->setRange(100.0, 600.0);
    buffer_seconds_->setSuffix(tr(" s"));
    buffer_seconds_->setValue(receive.buffer_seconds);
    form->addRow(tr("Buffer"), style::row(page, {buffer_seconds_}));
    form->addRow(QString(),
                 style::note(tr("Must exceed the longest mode (C, ~95 s) with margin "
                                "for retrospective decoding."),
                             page));

    poll_interval_ = new QDoubleSpinBox(page);
    poll_interval_->setRange(1.0, 30.0);
    poll_interval_->setSuffix(tr(" s"));
    poll_interval_->setValue(receive.poll_interval);
    form->addRow(tr("Decode every"), style::row(page, {poll_interval_}));
    return page;
}

QWidget* SettingsDialog::transmit_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    callsign_ = new QLineEdit(QString::fromStdString(config_.callsign), page);
    callsign_->setMaxLength(8);  // the beacon's callsign field
    callsign_->setPlaceholderText(QStringLiteral("N0CALL"));
    // Only what a callsign can contain. `apply_to` upper-cases it, so
    // lower case is accepted here and normalised on the way out.
    callsign_->setValidator(new QRegularExpressionValidator(
        QRegularExpression(QStringLiteral("^[A-Za-z0-9/]{0,8}$")), callsign_));
    form->addRow(tr("Callsign"), callsign_);
    form->addRow(QString(),
                 style::note(tr("Up to 8 characters. Sent continuously on the beacon "
                                "carrier, so a receiver can identify you even from "
                                "a partial reception."),
                             page));

    optimize_ = new QCheckBox(tr("Refine each image before sending"), page);
    optimize_->setChecked(config_.transmit.optimize);
    form->addRow(optimize_);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Worth around 1.5 dB of recovered quality, for no extra "
                        "airtime — most visibly on text and line art."),
                     tr("The encoder is trained to do well on average, not on "
                        "the image in front of it, so for any one image there "
                        "are better inputs to the same decoder. The search runs "
                        "in the time you spend composing.\n\n"
                        "It needs nothing of the receiving station: every "
                        "station decodes it as an ordinary transmission and "
                        "simply gets a better image. Downloads an extra 18 MB "
                        "the first time. If Send arrives before it has settled, "
                        "it finishes quickly and sends what it has."),
                     page));

    cw_id_ = new QCheckBox(tr("Send CW ID after each transmission"), page);
    cw_id_->setChecked(config_.transmit.cw_id);
    form->addRow(cw_id_);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("18 wpm at 1000 Hz, 500 ms after the image ends, under "
                        "the same PTT key-up."),
                     tr("A no-op with no callsign set above, whatever the "
                        "message says."),
                     page));

    cw_message_ = new QLineEdit(
        QString::fromStdString(config_.transmit.cw_message), page);
    cw_message_->setPlaceholderText(QStringLiteral("SSTVAE DE {callsign}"));
    form->addRow(tr("CW message"), cw_message_);
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("{callsign} is replaced with the callsign above; "
                        "everything else is sent as typed."),
                     tr("The default both identifies you and advertises the mode "
                        "and the software: someone who tunes across the signal "
                        "with no idea what it is hears the CW ID, and can go "
                        "find SSTVAE and a successful decode."),
                     page));

    // The transmit level itself stays on the send bar, where it is
    // adjusted, and keeps a short tooltip there. What lives here is the
    // *procedure*, because a tooltip only ever reaches an operator who
    // already suspects they have it wrong -- and drive is the one
    // adjustment that ruins a transmission silently. The two texts name
    // the same target on purpose; if this one changes, change the
    // tooltip in `tx_panel.cpp` with it.
    // One disclosure, not two paragraphs always on screen. The tooltip
    // on the slider in `tx_panel.cpp` is the short form and must name
    // the same target as this: "barely moving ALC" and "no ALC action"
    // are different drive levels, and an operator following either
    // should land in the same place.
    form->addRow(QString(),
                 style::note_with_detail(
                     tr("Transmit level is on the Transmit panel, beside the "
                        "mode. Set it so the radio shows no ALC action at all."),
                     tr("ALC is a compressor: it flattens the peaks this "
                        "waveform carries information in, and the far end sees "
                        "that as a lower SNR and a mangled image — while your "
                        "own meters look healthy. Start low and raise it until "
                        "power output stops rising, then back off."),
                     page));
    return page;
}

// --- result -----------------------------------------------------------------

void SettingsDialog::apply_to(settings::Config& config) const {
    config.callsign = callsign_->text().trimmed().toUpper().toStdString();
    config.model_path = model_path_->text().trimmed().toStdString();
    config.precision = precision_->currentData().toString().toStdString();

    config.audio.input_device =
        input_device_->currentData().toString().toStdString();
    config.audio.output_device =
        output_device_->currentData().toString().toStdString();

    config.rig = pending_rig();

    config.folders.receive_dir = receive_dir_->text().toStdString();
    config.folders.transmit_dir = transmit_dir_->text().toStdString();
    config.folders.template_dir = template_dir_->text().toStdString();

    config.transmit.optimize = optimize_->isChecked();
    config.transmit.cw_id = cw_id_->isChecked();
    config.transmit.cw_message = cw_message_->text().toStdString();
    config.receive.autosave = autosave_->isChecked();
    config.receive.save_audio = save_audio_->isChecked();
    config.receive.low_cpu = low_cpu_->isChecked();
    config.receive.filename_template = filename_template_->text().toStdString();
    config.receive.save_size = save_size_->text().trimmed().toStdString();
    config.receive.buffer_seconds = buffer_seconds_->value();
    config.receive.poll_interval = poll_interval_->value();
}

}  // namespace sstvae::gui
