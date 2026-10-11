#include "qrss_page.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>

#include "qrss_spectrogram.hpp"
#include "qrss_window.hpp"

namespace sstvae::gui {

QrssPage::QrssPage(QrssWindow* signals_view, QWidget* parent)
    : QWidget(parent), signals_(signals_view) {
    setObjectName(QStringLiteral("qrss_page"));
    auto* box = new QVBoxLayout(this);

    auto* row = new QHBoxLayout;
    listen_ = new QPushButton(this);
    listen_->setObjectName(QStringLiteral("qrss_listen"));
    listen_->setToolTip(tr("Starts and stops the radio's audio and the QRSS listener "
                           "together. The audio also feeds the SSTVAE decoder, the same "
                           "as Start receiving in SSTVAE mode."));
    connect(listen_, &QPushButton::clicked, this, [this] { emit listenRequested(!listening_); });
    row->addWidget(listen_);
    listen_state_ = new QLabel(this);
    listen_state_->setObjectName(QStringLiteral("qrss_listen_state"));
    row->addWidget(listen_state_, 1);
    schedule_summary_ = new QLabel(this);
    schedule_summary_->setObjectName(QStringLiteral("qrss_schedule_summary"));
    schedule_summary_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    row->addWidget(schedule_summary_, 1);
    auto* schedule = new QPushButton(tr("Sched&ule..."), this);
    schedule->setObjectName(QStringLiteral("qrss_schedule_button"));
    schedule->setToolTip(tr("Prepare pictures and schedule QRSS sends: once, back to back, "
                            "every few hours."));
    connect(schedule, &QPushButton::clicked, this, &QrssPage::scheduleRequested);
    row->addWidget(schedule);
    // The listener's command line and output: for when it misbehaves.
    details_ = new QPushButton(tr("Listener &details"), this);
    details_->setObjectName(QStringLiteral("qrss_details"));
    details_->setCheckable(true);
    details_->setToolTip(tr("Show the QRSS listener's command line and its output."));
    row->addWidget(details_);
    box->addLayout(row);

    // The spectrogram and the received pane, side by side, the divider
    // the operator's to move.
    splitter_ = new QSplitter(Qt::Horizontal, this);
    splitter_->setObjectName(QStringLiteral("qrss_splitter"));
    splitter_->setChildrenCollapsible(false);
    spectrogram_ = new QrssSpectrogram(splitter_);
    splitter_->addWidget(spectrogram_);
    if (signals_ != nullptr) {
        signals_->setParent(splitter_, Qt::Widget);
        splitter_->addWidget(signals_);
        signals_->set_spectrogram(spectrogram_);
        signals_->set_details_visible(false);
        connect(details_, &QPushButton::toggled, signals_, &QrssWindow::set_details_visible);
        signals_->show();
    }
    splitter_->setStretchFactor(0, 3);
    splitter_->setStretchFactor(1, 2);
    box->addWidget(splitter_, 1);
    set_listening(false);
}

void QrssPage::set_listening(bool on) {
    listening_ = on;
    show_state();
}

void QrssPage::set_transmitting(bool on) {
    transmitting_ = on;
    show_state();
}

void QrssPage::show_state() {
    listen_->setEnabled(!transmitting_);
    if (transmitting_) {
        listen_->setText(tr("Transmission active"));
        listen_state_->setText(tr("Sending: receiving starts again when the send ends."));
        return;
    }
    listen_->setText(listening_ ? tr("&Stop receiving") : tr("&Start receiving"));
    listen_state_->setText(listening_ ? tr("Receiving: the QRSS listener hears the radio.")
                                      : tr("Not receiving: start it to hear QRSS signals."));
}

void QrssPage::set_schedule_summary(const QString& text) { schedule_summary_->setText(text); }

}  // namespace sstvae::gui
