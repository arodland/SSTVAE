// The transmit side, as QML sees it.
//
// A view, owning nothing -- the counterpart of `Listener`. The picture
// and its framing live in `Composition`, the engine lives in `Session`,
// and both outlive this object, so a rotation in the middle of composing
// or in the middle of an over loses nothing. Every property is derived
// on demand for the same reason it is on the receive side.
//
// Starting an over goes *through the service*, never straight to
// `Session`, exactly as starting capture does. The picture does not fit
// in an Intent, so the sequence is stage-then-ask: this object stages the
// request on the session and asks the service to send it.
//
// The station settings live here rather than in `core/settings/`.
// **QSettings, not the desktop's config.json**, which is a real
// divergence and a deliberate one: that file's schema is a rig section,
// folder paths and a window layout, none of which exist on a phone,
// and its hand-editability -- the property the round-trip discipline
// protects -- buys nothing on a device with no text editor and no shared
// machine. The app already stored `ui/showTechnical` this way. What does
// carry over is the discipline itself: every setting displayed is
// written back through the same key it was read from, with the key
// spelled once as a constant.

#ifndef SSTVAE_ANDROID_TRANSMITTER_HPP
#define SSTVAE_ANDROID_TRANSMITTER_HPP

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

#include <map>
#include <string>
#include <vector>

#include "overlay/model.hpp"
#include "overlay/template.hpp"

class Transmitter : public QObject {
    Q_OBJECT
    QML_ELEMENT

    // --- the picture -----------------------------------------------
    Q_PROPERTY(bool hasPicture READ hasPicture NOTIFY changed)
    // Bumped whenever the framing moves, because QML caches images by
    // URL -- the same reason the live receive preview carries an id.
    Q_PROPERTY(int previewId READ previewId NOTIFY changed)
    Q_PROPERTY(double zoom READ zoom WRITE setZoom NOTIFY changed)
    // The bottom of the zoom travel for the picture currently loaded:
    // the point at which all of it is visible, letterboxed. 1.0 for a
    // 4:3 source, where there is nothing to letterbox. Exposed so the
    // slider and the pinch stop exactly where `images::fit` clamps,
    // rather than at a constant that would be wrong for every aspect
    // but one.
    Q_PROPERTY(double minZoom READ minZoom NOTIFY changed)
    Q_PROPERTY(double centerX READ centerX NOTIFY changed)
    Q_PROPERTY(double centerY READ centerY NOTIFY changed)

    // --- station settings -------------------------------------------
    Q_PROPERTY(QString callsign READ callsign WRITE setCallsign NOTIFY changed)
    Q_PROPERTY(QString mode READ mode WRITE setMode NOTIFY changed)
    Q_PROPERTY(QStringList modes READ modes CONSTANT)
    Q_PROPERTY(double level READ level WRITE setLevel NOTIFY changed)
    Q_PROPERTY(bool cwId READ cwId WRITE setCwId NOTIFY changed)
    Q_PROPERTY(QString cwMessage READ cwMessage WRITE setCwMessage NOTIFY changed)
    // Seconds of leader, 0 for none. A duration rather than a switch
    // because VOX circuits differ and the operator is the one who can
    // hear whether the start of the transmission survived.
    Q_PROPERTY(double voxLead READ voxLead WRITE setVoxLead NOTIFY changed)
    Q_PROPERTY(QStringList outputDevices READ outputDevices NOTIFY devicesChanged)
    Q_PROPERTY(QString outputDevice READ outputDevice WRITE setOutputDevice NOTIFY changed)

    // --- templates (docs/overlay-templates.md) ------------------------
    //
    // No editor here (step 4, not built) -- a template is picked, not
    // composed, and the only free text the operator ever types is a
    // custom field's value.
    Q_PROPERTY(QStringList templateNames READ templateNames NOTIFY changed)
    // Index 0 is always "None": the picture goes out unmodified, and
    // choosing it is how a template is left.
    Q_PROPERTY(int templateIndex READ templateIndex WRITE setTemplateIndex
                   NOTIFY changed)
    // Whether the *current* template has a `{theircall}` hole -- the
    // field below is shown enabled only then, matching the desktop's
    // "and only those" rule.
    Q_PROPERTY(bool wantsTheirCall READ wantsTheirCall NOTIFY changed)
    Q_PROPERTY(QString theirCall READ theirCall WRITE setTheirCall NOTIFY changed)
    // `{snr}` is never typed -- see `refreshFields()` -- so there is no
    // corresponding WRITE property; the reply target it comes from is
    // exposed only as `hasReplyTarget`, for a screen that wants to say
    // whose picture is bound in.
    Q_PROPERTY(bool hasReplyTarget READ hasReplyTarget NOTIFY changed)
    // Button text: "Custom fields..." or "Custom fields (2)...". The
    // fields themselves are a pop-up (`customFieldLabels()` and
    // friends below), not a property, for the reason the desktop's are
    // a pop-up too -- see `docs/overlay-templates.md`.
    Q_PROPERTY(bool hasCustomFields READ hasCustomFields NOTIFY changed)
    Q_PROPERTY(QString customFieldsLabel READ customFieldsLabel NOTIFY changed)
    // Empty unless the current template needs `{theircall}` and it is
    // blank. Blocks Send, same tier as `cwIdProblem` -- "  de KC2G" on
    // the air is the whole point of a reply template gone wrong, and
    // the desktop's screen is large enough that the gap already reads
    // as a gap; this one is not.
    Q_PROPERTY(QString templateFieldProblem READ templateFieldProblem
                   NOTIFY changed)

    // --- the over ----------------------------------------------------
    Q_PROPERTY(bool encoderReady READ encoderReady NOTIFY changed)
    Q_PROPERTY(QString encoderStatus READ encoderStatus NOTIFY changed)
    Q_PROPERTY(bool transmitting READ transmitting NOTIFY changed)
    Q_PROPERTY(bool canSend READ canSend NOTIFY changed)
    Q_PROPERTY(QString txStatus READ txStatus NOTIFY changed)
    Q_PROPERTY(double txProgress READ txProgress NOTIFY changed)
    // How long this mode's transmission will take, as text, so the
    // operator can decide whether to start one now.
    Q_PROPERTY(QString airtime READ airtime NOTIFY changed)
    // How the radio will be put into transmit for this over. A sentence
    // rather than a flag, because the two cases behave differently in a
    // way the operator can otherwise only find out on the air: with rig
    // control keying the radio the VOX leader is skipped, so a leader
    // that was doing the job yesterday is silently not in the waveform
    // today.
    Q_PROPERTY(QString keying READ keying NOTIFY changed)
    // Whether the rig will key the radio, as a fact rather than as a
    // substring of `keying` -- that text is translatable, so testing its
    // prefix works until somebody ships a translation.
    Q_PROPERTY(bool rigKeyed READ rigKeyed NOTIFY changed)
    Q_PROPERTY(QString lastError READ lastError NOTIFY changed)
    // Empty unless the CW ID settings would key a partial
    // identification; see `tx::cw_id_problem`. Non-empty blocks Send,
    // and the text is shown rather than merely disabling the button --
    // a control that is off for an invisible reason is worse than one
    // that refuses out loud.
    Q_PROPERTY(QString cwIdProblem READ cwIdProblem NOTIFY changed)
    // True until the operator has been through the first-transmit
    // prompt once. Send opens that instead of transmitting while it is
    // true.
    Q_PROPERTY(bool needsFirstTransmitPrompt READ needsFirstTransmitPrompt
                   NOTIFY changed)

public:
    explicit Transmitter(QObject* parent = nullptr);
    ~Transmitter() override;

    bool hasPicture() const;
    int previewId() const { return preview_id_; }
    double zoom() const;
    double minZoom() const;
    double centerX() const;
    double centerY() const;
    void setZoom(double z);

    QString callsign() const { return callsign_; }
    void setCallsign(const QString& c);
    QString mode() const { return mode_; }
    void setMode(const QString& m);
    QStringList modes() const;
    double level() const { return level_; }
    void setLevel(double v);
    bool cwId() const { return cw_id_; }
    void setCwId(bool on);
    QString cwMessage() const { return cw_message_; }
    void setCwMessage(const QString& m);
    double voxLead() const { return vox_lead_s_; }
    void setVoxLead(double s);
    QStringList outputDevices() const { return devices_; }
    QString outputDevice() const { return device_; }
    void setOutputDevice(const QString& d);

    bool encoderReady() const;
    QString encoderStatus() const;
    bool transmitting() const;
    bool canSend() const;
    QString txStatus() const;
    double txProgress() const;
    QString airtime() const;
    QString keying() const;
    bool rigKeyed() const;
    QString lastError() const;
    QString cwIdProblem() const;
    bool needsFirstTransmitPrompt() const { return !acknowledged_; }

    QStringList templateNames() const;
    int templateIndex() const { return template_index_; }
    void setTemplateIndex(int index);
    bool wantsTheirCall() const;
    QString theirCall() const { return theircall_; }
    void setTheirCall(const QString& c);
    bool hasReplyTarget() const;
    bool hasCustomFields() const;
    QString customFieldsLabel() const;
    QString templateFieldProblem() const;

    // The current template's custom-field labels, in the order they
    // first appear -- what the "Custom fields..." pop-up builds its
    // rows from. `customFieldValue`/`setCustomFieldValue` are keyed by
    // that same label, which is the only identity a custom field has
    // (docs/overlay-templates.md), so a value survives a template
    // switch that reuses the same label.
    Q_INVOKABLE QStringList customFieldLabels() const;
    Q_INVOKABLE QString customFieldValue(const QString& label) const;
    Q_INVOKABLE void setCustomFieldValue(const QString& label, const QString& value);

    // Reply to a reception: binds a `last_rx` item to *this* picture,
    // seeds {theircall} from `callsign` (still editable after), and
    // selects the last reply template used (or plain "Reply" the first
    // time). Called from Pictures, the viewer, and the Listen tab's
    // completion row -- see docs/overlay-templates.md.
    Q_INVOKABLE void replyTo(const QString& path, const QString& callsign,
                             double snrDb);

    // Record that the operator has read the first-transmit prompt and
    // accepted responsibility for operating legally. Persisted, so it
    // is asked once per install rather than once per launch.
    //
    // **Not a licence check and not a gate**, deliberately: the app
    // cannot tell whether it is connected to a radio at all, the
    // service may not be amateur, and the operator may be identifying
    // by voice or by a means this app never sees. What it is is a
    // roadblock to casual misuse -- someone who has not thought about
    // any of that has now been asked to.
    // Taking a template from another station: the operator pastes what
    // the desktop's Share window showed (or what a QR scanner put on
    // the clipboard), and it lands in the template folder as if it had
    // been saved here. Returns why it could not be imported, or an
    // empty string when it worked.
    //
    // **`overlay::sanitize_imported` first**, always: a document from
    // elsewhere must not name files on this phone. See that function.
    Q_INVOKABLE QString importTemplate(const QString& payload);
    // The same import, read off another screen by the phone's camera
    // (`TemplateScanner.java`, ML Kit's unbundled scanner). Result or
    // failure arrives through `onScanned`; a failure lands in
    // `lastError`, since by then the popup that asked is gone.
    Q_INVOKABLE void scanTemplate();

    Q_INVOKABLE void acknowledgeFirstTransmit();

    // Drag motion, as a fraction of the *preview's* own width and
    // height -- so QML passes `-dx/width, -dy/height` and never has to
    // know the crop window's size in source coordinates, which depends
    // on the aspect ratio and the zoom and is clamped in `Composition`
    // where both are known.
    Q_INVOKABLE void panBy(double dx, double dy);
    Q_INVOKABLE void pickImage();
    Q_INVOKABLE void takePhoto();
    Q_INVOKABLE void clearPicture();
    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void loadEncoder();
    Q_INVOKABLE void send();
    Q_INVOKABLE void cancel();

    // Called from the picker's JNI callback, on the Android UI thread.
    void onPicked(const QString& path, const QString& error);
    void onScanned(const QString& payload, const QString& error);

signals:
    void devicesChanged();
    void changed();

private:
    void bump();
    void refreshTemplates();
    // Recomputes the whole `overlay::Fields` from station settings,
    // `theircall_`, the reply target and the custom-field map, and
    // pushes it to `Composition` -- the same "recompute the lot, do not
    // track it incrementally" choice the desktop's `refresh_fields`
    // makes, since it is a handful of short strings.
    void refreshFields();
    int templateIndexForName(const QString& name) const;

    QStringList devices_;
    QString error_;
    QTimer poll_;
    int preview_id_ = 0;

    QString callsign_;
    QString mode_;
    double level_ = 0.9;
    bool cw_id_ = false;
    QString cw_message_;
    bool acknowledged_ = false;
    double vox_lead_s_ = 0.0;
    QString device_;

    // Parallel to `templateNames()`; index 0 is the built-in "None" (an
    // empty document, not read from any file).
    std::vector<sstvae::overlay::Doc> templates_;
    int template_index_ = 0;
    QString theircall_;
    // Keyed by label, so a "Comment" field's value survives a switch
    // between templates that both declare one -- the label is the only
    // identity a custom field has. In-memory only: it follows the
    // reply target across a rotation and is gone on relaunch, matching
    // docs/overlay-templates.md's persistence rule.
    std::map<std::string, std::string> custom_field_values_;
    // The template index `replyTo()` reopens, remembered from whichever
    // {theircall}-using template was selected most recently -- by Reply
    // or by hand. -1 until the first reply, which falls back to plain
    // "Reply".
    int last_reply_template_index_ = -1;
};

#endif
