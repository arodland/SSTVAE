// The settings dialog, as a round trip.
//
// The failure this exists for is a field the dialog *displays* but
// forgets to write back in `apply_to` -- or writes back from the wrong
// control. Nothing crashes, nothing warns; the operator changes a
// setting, presses OK, and it silently reverts. The two places are
// hundreds of lines apart and grow at different times, which is exactly
// when they drift.
//
// So: build a config in which **no field holds its default**, show it
// to the dialog, apply it to an empty config, and require the result to
// be identical. A dropped field cannot survive that, because a dropped
// field comes back as its default and no default appears here.

#include <QApplication>

#include <string>

#include "check.hpp"
#include "settings/settings.hpp"
#include "settings_dialog.hpp"

using namespace sstvae;

namespace {

// Every field different from `settings::Config{}`.
settings::Config populated() {
    settings::Config c;
    c.callsign = "KC2G";
    c.model_path = "/opt/models/v2";
    c.precision = "int8";

    c.audio.input_device = "USB Audio CODEC";
    c.audio.output_device = "SSTVAE-Loopback";

    c.rig.enabled = true;
    c.rig.model = 2;  // NET rigctl, which is in every Hamlib build
    c.rig.poll_interval_s = 1.5;
    c.rig.ptt_lead_s = 0.45;
    c.rig.ptt_tail_s = 0.25;
    c.rig.device = "localhost:4532";
    c.rig.baud = 38400;
    c.rig.data_bits = "seven";
    c.rig.stop_bits = "two";
    c.rig.parity = "odd";
    c.rig.handshake = "hardware";
    c.rig.dtr = "high";
    c.rig.rts = "low";
    c.rig.ptt_method = "rts";
    c.rig.ptt_device = "/dev/ttyUSB9";
    c.rig.mode = "pkt_usb";

    c.folders.receive_dir = "/srv/sstv/in";
    c.folders.transmit_dir = "/srv/sstv/out";
    c.transmit.optimize = true;  // not the default, per this fixture's rule
    c.transmit.cw_id = true;
    c.transmit.cw_message = "TEST DE {callsign}";
    c.folders.template_dir = "/srv/sstv/tpl";

    c.receive.autosave = false;
    c.receive.auto_start = true;
    c.receive.save_audio = true;
    c.receive.low_cpu = true;
    c.receive.blind_wide = true;
    c.receive.drift_track = "fast";
    c.receive.filename_template = "{callsign}_{date}";
    c.receive.save_size = "320x240";
    c.receive.buffer_seconds = 155.0;
    c.receive.poll_interval = 11.0;
    return c;
}

void test_every_field_survives_the_dialog() {
    const settings::Config before = populated();
    gui::SettingsDialog dialog(before);

    settings::Config after;
    dialog.apply_to(after);

    // Compared as JSON so the failure names the field rather than saying
    // two structs differ.
    const std::string want = settings::to_json(before);
    const std::string got = settings::to_json(after);
    if (want != got) {
        check::fail("settings dialog: every field survives a round trip",
                    "\n  in:  " + want + "\n  out: " + got);
        return;
    }
    check::is_true(true, "settings dialog: every field survives a round trip");
}

void test_defaults_survive_too() {
    // The other direction: a fresh install must not be mangled by
    // opening the dialog and pressing OK without touching anything.
    const settings::Config before;
    gui::SettingsDialog dialog(before);
    settings::Config after;
    dialog.apply_to(after);
    check::equal(settings::to_json(after), settings::to_json(before),
                 "settings dialog: OK on an untouched dialog changes nothing");
}

void test_a_callsign_is_normalized() {
    settings::Config before;
    before.callsign = "  kc2g  ";
    gui::SettingsDialog dialog(before);
    settings::Config after;
    dialog.apply_to(after);
    // Upper-cased and trimmed: it goes out on the beacon, where the
    // field is fixed-width and case-insensitive by convention.
    check::equal(after.callsign, std::string("KC2G"),
                 "settings dialog: the callsign is trimmed and upper-cased");
}

void test_an_unknown_rig_model_is_kept_not_reset() {
    // A config written by a different Hamlib build may name a model this
    // one does not list. Silently resetting it to the dummy rig would
    // lose a setting the operator chose, and they would find out at the
    // first attempt to key.
    settings::Config before;
    before.rig.model = 999999;
    gui::SettingsDialog dialog(before);
    settings::Config after;
    dialog.apply_to(after);
    check::equal(after.rig.model, 999999,
                 "settings dialog: an unlisted rig model survives");
}

}  // namespace

int main(int argc, char** argv) {
    check::report_crashes_instead_of_prompting();
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QApplication app(argc, argv);

    test_every_field_survives_the_dialog();
    test_defaults_survive_too();
    test_a_callsign_is_normalized();
    test_an_unknown_rig_model_is_kept_not_reset();

    return check::report("settings dialog");
}
