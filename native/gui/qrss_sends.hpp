// The QRSS "Sends": pictures prepared for the schedule and kept on disk,
// so the same picture can be scheduled again tomorrow without composing
// it again.
//
// A Send is three files in one directory (`config_dir()/qrss_sends`):
//
//   <id>.png       the picture as it goes on the air, 640 x 480
//   <id>-base.png  the picture under the overlay, already framed
//   <id>.json      its name, when it was made, and the overlay document
//
// The base and the overlay are kept apart so the Editor can open a Send
// again with its text still editable; the flat PNG is what the schedule
// copies when a Send is scheduled. A scheduled entry keeps its own copy
// (gui/qrss_schedule.hpp), so editing a Send later does not change a
// picture already on the schedule: that would give its later passes a
// different picture ID, and receivers could no longer add them up.

#ifndef SSTVAE_GUI_QRSS_SENDS_HPP
#define SSTVAE_GUI_QRSS_SENDS_HPP

#include <QObject>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "images/types.hpp"
#include "overlay/model.hpp"

namespace sstvae::gui {

struct QrssSend {
    std::string id;
    std::string label;
    double created = 0.0;   // unix seconds
    double updated = 0.0;
    std::filesystem::path picture;   // <id>.png
    std::filesystem::path base;      // <id>-base.png
};

class QrssSends : public QObject {
    Q_OBJECT

public:
    explicit QrssSends(std::filesystem::path dir, QObject* parent = nullptr);

    const std::filesystem::path& dir() const { return dir_; }
    // Newest first.
    const std::vector<QrssSend>& sends() const { return sends_; }
    const QrssSend* find(const std::string& id) const;

    // Saves a new Send; returns its id, or empty with `why` set.
    std::string add(const std::string& label, const images::Picture& base,
                    const overlay::Doc& doc, const images::Picture& picture,
                    std::string* why = nullptr);
    // Replaces an existing Send's files. False (with `why`) if it is gone
    // or cannot be written.
    bool update(const std::string& id, const std::string& label, const images::Picture& base,
                const overlay::Doc& doc, const images::Picture& picture,
                std::string* why = nullptr);
    bool rename(const std::string& id, const std::string& label);
    void remove(const std::string& id);

    // What the Editor needs to open a Send again.
    std::optional<images::Picture> load_base(const std::string& id) const;
    overlay::Doc load_doc(const std::string& id) const;

    // Re-read the directory.
    void reload();

signals:
    void changed();

private:
    bool write(const QrssSend& send, const images::Picture& base, const overlay::Doc& doc,
               const images::Picture& picture, std::string* why);

    std::filesystem::path dir_;
    std::vector<QrssSend> sends_;
};

}  // namespace sstvae::gui

#endif
