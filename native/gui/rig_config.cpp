#include "rig_config.hpp"

#include <exception>
#include <initializer_list>
#include <string>
#include <utility>

#include "rig/hamlib.hpp"

namespace sstvae::gui {

namespace {

// The settings file spells the enumerated rig options in lowercase, so
// a config a human opens does not read like Hamlib's combo boxes. This
// is the one place the two vocabularies meet.
//
// An unrecognized value falls back to Default rather than being an
// error, and that direction is deliberate: Default means "do not set
// the token", so a typo leaves the backend on its own settings instead
// of forcing a wrong one onto a radio.
template <typename E>
E lookup(const std::string& value, std::initializer_list<std::pair<const char*, E>> table,
         E fallback) {
    for (const auto& [name, mapped] : table) {
        if (value == name) return mapped;
    }
    return fallback;
}

}  // namespace

std::unique_ptr<rig::RigBackend> make_backend(const settings::RigConfig& config) {
    rig::HamlibConfig hamlib;
    hamlib.model = config.model;
    hamlib.device = config.device;
    hamlib.baud = config.baud;
    hamlib.data_bits = lookup<rig::DataBits>(
        config.data_bits,
        {{"seven", rig::DataBits::Seven}, {"eight", rig::DataBits::Eight}},
        rig::DataBits::Default);
    hamlib.stop_bits = lookup<rig::StopBits>(
        config.stop_bits,
        {{"one", rig::StopBits::One}, {"two", rig::StopBits::Two}},
        rig::StopBits::Default);
    hamlib.parity = lookup<rig::Parity>(config.parity,
                                        {{"none", rig::Parity::None},
                                         {"odd", rig::Parity::Odd},
                                         {"even", rig::Parity::Even}},
                                        rig::Parity::Default);
    hamlib.handshake = lookup<rig::Handshake>(
        config.handshake,
        {{"none", rig::Handshake::None},
         {"xonxoff", rig::Handshake::XonXoff},
         {"hardware", rig::Handshake::Hardware}},
        rig::Handshake::Default);
    hamlib.dtr = lookup<rig::LineState>(
        config.dtr, {{"high", rig::LineState::High}, {"low", rig::LineState::Low}},
        rig::LineState::Default);
    hamlib.rts = lookup<rig::LineState>(
        config.rts, {{"high", rig::LineState::High}, {"low", rig::LineState::Low}},
        rig::LineState::Default);
    // Cat rather than Vox as the fallback: an unreadable value must not
    // silently turn keying off, which would look like a dead PTT.
    hamlib.ptt_method = lookup<rig::PttMethod>(config.ptt_method,
                                               {{"vox", rig::PttMethod::Vox},
                                                {"cat", rig::PttMethod::Cat},
                                                {"dtr", rig::PttMethod::Dtr},
                                                {"rts", rig::PttMethod::Rts}},
                                               rig::PttMethod::Cat);
    hamlib.ptt_device = config.ptt_device;
    hamlib.mode = lookup<rig::RigMode>(
        config.mode, {{"usb", rig::RigMode::Usb}, {"pkt_usb", rig::RigMode::PktUsb}},
        rig::RigMode::None);
    return rig::make_hamlib_backend(hamlib);
}

rig::RigConfig controller_config(const settings::RigConfig& config) {
    rig::RigConfig out;
    out.poll_interval_s = config.poll_interval_s;
    return out;
}


}  // namespace sstvae::gui
