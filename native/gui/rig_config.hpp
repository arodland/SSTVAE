// Turning stored settings into a Hamlib session.
//
// Shared by `AppState`, which opens the radio for real, and the
// settings dialog's Test CAT / Test PTT buttons. **One mapping, not
// two**: a second copy would let the test button pass while the app
// fails, which is worse than having no test button -- it would send the
// operator looking at their radio instead of at the setting that
// differs.
//
// The settings file spells the enumerated options in lowercase so a
// config a human opens does not read like Hamlib's combo boxes; this is
// the one place the two vocabularies meet.

#ifndef SSTVAE_GUI_RIG_CONFIG_HPP
#define SSTVAE_GUI_RIG_CONFIG_HPP

#include <memory>

#include "rig/backend.hpp"
#include "rig/controller.hpp"
#include "settings/settings.hpp"

namespace sstvae::gui {

// A backend for this configuration. Does not open it.
std::unique_ptr<rig::RigBackend> make_backend(const settings::RigConfig& config);

// The controller's own settings, which are about polling rather than
// about the radio.
rig::RigConfig controller_config(const settings::RigConfig& config);

}  // namespace sstvae::gui

#endif
