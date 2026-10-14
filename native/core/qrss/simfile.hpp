// qrss_simulate.py's .npz, as the receiver reads it: the front end's
// samples (`fe`), their rate, where t0 falls, the slot and the frame's name.
// Port of what qrss_receive.py takes from sstvae.qrss.channel.load_npz (the
// reference); the channel's truth arrays are a simulator's, for scoring a
// receiver, and the receiver never looks at them.
//
// load_npz reads every member it knows, in this order, and a missing one
// is a KeyError ("'truth_u is not a file in the archive'"), so a file that
// is not a simulator's (a PassResult, a .qrsp) is refused the same way
// here: truth_u, truth_f_hz, truth_t_pos_s, truth_impulse_mask,
// truth_noise_var_fe, truth_agc_gain (only if present), truth_signal_power,
// truth_time_scale, truth_u_data (only if present), fe, fs, t0_index, q,
// frame. Each is read as np.load reads it (the .npy header checked), and
// the scalars converted as int() and float() convert a 0-d array.
//
// The frame's name is a 0-d str array (np.savez of a str: dtype '<U5' for
// "short"), which npz.hpp does not read (its dtypes are numbers and bools),
// so this file walks the archive itself -- np.savez's form only: stored
// members, the central directory's sizes or their zip64 extra, no zip64 end
// record (an archive under 2 GiB, which a FULL slot's 29 MB is) -- and hands
// each numeric member to npz::parse_npy. Once npz.hpp reads str members the
// walk can go and this reads through npz::parse_npz.
//
// Not reproduced: zipfile's handling of damaged archives (refused here with
// the class np.load would raise, zipfile.BadZipFile, in zipfile's words for
// "File is not a zip file", the bad magic numbers and offsets and a bad
// CRC-32, else in this file's own, "simfile: ..."); compressed and zip64
// archives, which zipfile reads and np.savez does not write for a slot;
// and members of other dtypes or shapes than the simulator writes:
// `frame` must be a str.
//
// What save_npz never writes, load_npz refuses as it reads it (ValueError),
// and so does this: `fe` other than complex64 (n,), "simfile: fe is <descr>
// of <ndim> dimensions, not complex64 (n,) as qrss_simulate.py writes it";
// fs <= 0, "simfile: fs <fs> is not a sample rate"; q < 0, "simfile: q <q>
// is a quarter hour before 1970-01-01T00:00Z".

#ifndef SSTVAE_QRSS_SIMFILE_HPP
#define SSTVAE_QRSS_SIMFILE_HPP

#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::qrss::simfile {

// The receiver's part of channel.Sim.
struct Sim {
    std::vector<std::complex<float>> fe;   // complex64: the front end's stream
    std::int64_t fs = 0;                   // int(d["fs"])
    double t0_index = 0.0;                 // float(d["t0_index"])
    std::int64_t q = 0;                    // int(d["q"])
    std::string frame;                     // str(d["frame"]), UTF-8 (lframe.get_any reads it)
};

// A refusal as the reference raises it: the class its traceback names
// ("KeyError", "ValueError", "TypeError", "EOFError", "OverflowError",
// "zipfile.BadZipFile") and the message as Python prints it after the
// class (a KeyError's repr'd: "'fs is not a file in the archive'").
class Error : public std::invalid_argument {
public:
    Error(std::string cls, const std::string& message) : std::invalid_argument(message), cls_(std::move(cls)) {}
    const std::string& cls() const noexcept { return cls_; }

private:
    std::string cls_;
};

// channel.load_npz of a file's bytes (np.load's first look included: an
// empty file is EOFError, an .npy a TypeError, anything else that is not a
// zip the pickle refusal). Error for every refusal.
Sim parse(std::span<const std::uint8_t> file);

// The same from a path: std::runtime_error when the file cannot be read.
Sim load(const std::string& path);

}  // namespace sstvae::qrss::simfile

#endif  // SSTVAE_QRSS_SIMFILE_HPP
