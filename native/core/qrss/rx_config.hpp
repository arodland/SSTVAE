// GENERATED FILE -- DO NOT EDIT.
// Written by tools/gen_qrss_rx_config.py from sstvae/qrss.
// Regenerate with: python tools/gen_qrss_rx_config.py
//
// The QRSSTVAE receiver's module constants (calibrations, settings and
// thresholds), one namespace per Python module. Not a format: nothing
// here goes on the air (config.hpp has those). Private Python names
// (_FP_NUM) appear with the leading underscores moved to the end (FP_NUM_).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "qrss/config.hpp"

namespace sstvae::qrss::rxconfig {

// One entry of a Python dict of str to number, in insertion order.
struct NamedValue {
    std::string_view name;
    double value;
};

// The value for `name`, or `fallback` (dict.get).
template <std::size_t N>
constexpr double lookup(const std::array<NamedValue, N>& table, std::string_view name, double fallback) {
    for (const NamedValue& e : table)
        if (e.name == name) return e.value;
    return fallback;
}

// --- sstvae/qrss/track.py ---
namespace track {
inline constexpr double KAPPA_SELF = 0x1.e147ae147ae14p-1;  // 0.94
inline constexpr double KAPPA_SELF_KNOWN = 0x1.1eb851eb851ecp-3;  // 0.14
inline constexpr std::array<double, 13> Q_SPREADS_HZ{0x1.47ae147ae147bp-7, 0x1.079e59f2ba9d2p-6, 0x1.a7daa4fca42afp-6, 0x1.54f3775b81301p-5, 0x1.122fad6cb5350p-4, 0x1.b90ea9e6eeb70p-4, 0x1.62bbecaab8a5dp-3, 0x1.1d495182a9931p-2, 0x1.cae536501e258p-2, 0x1.7111f0c34c1a9p-1, 0x1.28d4562e09fe8p+0, 0x1.dd753a3ec02f3p+0, 0x1.8000000000000p+1};  // (0.01, 0.01609, 0.02587, 0.04162, 0.06694, 0.10768, 0.17321, 0.2786, 0.44814, 0.72084, 1.15949, 1.86507, 3.0)
inline constexpr double LEAD_KEEP_S = 0x1.0000000000000p+3;  // 8.0
inline constexpr double RECENTER_WIN_S = 0x1.e000000000000p+4;  // 30.0
inline constexpr double RECENTER_MIN_HZ = 0x1.47ae147ae147bp-6;  // 0.02
inline constexpr double KNOTS_S = 0x1.4000000000000p+3;  // 10.0
inline constexpr std::array<double, 4> CARRIER_LP_HZ{0x1.eb851eb851eb8p-6, 0x1.3333333333333p-2, 0x1.8000000000000p+0, 0x1.8000000000000p+1};  // (0.03, 0.3, 1.5, 3.0)
inline constexpr double PU_SMOOTH_S = 0x1.0000000000000p+0;  // 1.0
inline constexpr double C_MIN = 0x1.999999999999ap-4;  // 0.1
inline constexpr double PSI_MIN = 0x1.0000000000000p-1;  // 0.5
inline constexpr double NOISE_WIN_S = 0x1.4000000000000p+3;  // 10.0
inline constexpr int MF_RES = 512;
inline constexpr std::array<double, 13> Q_GRID{0x1.dac586775972ep-31, 0x1.ee6a986cb10a6p-29, 0x1.00e054b6c2940p-26, 0x1.0b697c2ad13f9p-24, 0x1.162538d9832e0p-22, 0x1.2170df0ce8fa7p-20, 0x1.2d2bddb62599bp-18, 0x1.395012bafeb44p-16, 0x1.45ffad59ba676p-14, 0x1.532e51c37fbd4p-12, 0x1.60e74490214d2p-10, 0x1.6f2e576eca2b4p-8, 0x1.7e07eeb30f785p-6};  // (8.636044786572047e-10, 3.5973526851336617e-09, 1.4952168138210566e-08, 6.226171992997598e-08, 2.5904308884119435e-07, 1.078251389710088e-06, 4.487802860128621e-06, 1.867489730143368e-05, 7.772415610998195e-05, 0.000323468146433412, 0.0013462195966281077, 0.005602737756625831, 0.023317320923744517)
inline constexpr double R_FLOOR = 0x1.0624dd2f1a9fcp-10;  // 0.001
inline constexpr bool DEBUG_SEG_ = false;
inline constexpr double PATH_SEG_S = 0x1.4000000000000p+4;  // 20.0
inline constexpr double PATH_FINE_S = 0x1.4000000000000p+2;  // 5.0
inline constexpr double PATH_FINE_SHARE = 0x1.6666666666666p-1;  // 0.7
inline constexpr double PATH_FINE_RATIO = 0x1.4000000000000p+5;  // 40.0
inline constexpr double PATH_SPAN_HZ = 0x1.0000000000000p+1;  // 2.0
inline constexpr double SEG_TIMING_S = 0x1.4000000000000p+4;  // 20.0
inline constexpr int V_TIMING_TRIES = 8;
inline constexpr double LINE_SEG_S = 0x1.e000000000000p+5;  // 60.0
inline constexpr double LINE_SPAN_HZ = 0x1.0000000000000p+0;  // 1.0
inline constexpr double LINE_DRIFT_HZ_MIN = 0x1.999999999999ap-2;  // 0.4
inline constexpr double LINE_DRIFT_STEP = 0x1.47ae147ae147bp-7;  // 0.01
inline constexpr double LEAD_LINE_S = 0x1.4000000000000p+4;  // 20.0
// not emitted: _Q_GRIDS (dict), _MF_TABLES (dict)
}  // namespace track

// --- sstvae/qrss/ltrack.py ---
namespace ltrack {
inline constexpr std::array<double, 4> CARRIER_LP_HZ_L{0x1.eb851eb851eb8p-6, 0x1.3333333333333p-2, 0x1.0000000000000p+0, 0x1.0000000000000p+1};  // (0.03, 0.3, 1.0, 2.0)
inline constexpr int PILOT_K = 2;
inline constexpr int PILOT_REACH = 16;
inline constexpr double FP_T_ = 0x1.869fbe76c8b43p-1;  // 0.7629374999999999
inline constexpr int FP_NUM_ = 12207;
inline constexpr int FP_DEN_ = 16000;
inline constexpr double DELTA_PRIOR_SD = 0x1.0000000000000p-2;  // 0.25
inline constexpr double SEG_TIMING_S = 0x1.4000000000000p+4;  // 20.0
inline constexpr double PATH_MED_LO_HZ = 0x1.0000000000000p+0;  // 1.0
inline constexpr double K_PL_DATA = 0x1.0624dd2f1a9fcp-6;  // 0.016
inline constexpr double K_PL_ID = 0.0;  // 0.0
inline constexpr std::array<NamedValue, 4> L_KEYED_POWER{{NamedValue{"l-tiny", 0x1.0cccccccccccdp+0}, NamedValue{"l-short", 0x1.0e978d4fdf3b6p+0}, NamedValue{"l-medium", 0x1.0666666666666p+0}, NamedValue{"l-full", 0x1.024dd2f1a9fbep+0}}};  // {'l-tiny': 1.05, 'l-short': 1.057, 'l-medium': 1.025, 'l-full': 1.009}
// not emitted: _PILOT_TABLE (list)
}  // namespace ltrack

// --- sstvae/qrss/frontend.py ---
namespace frontend {
inline constexpr int FE_DECIM_ = 2;
inline constexpr std::array<double, 255> FE_TAPS_{-0x1.471881376ed95p-18, -0x1.d03509843a83ap-18, -0x1.e873aa50c7786p-20, 0x1.0e75556b8ae7ap-17, 0x1.c38b67162bf51p-17, 0x1.a2e48bb3c21c5p-18, -0x1.65bb6c6433d1cp-17, -0x1.776cf1ec8130ep-16, -0x1.e2b8f2315372bp-17, 0x1.7e32331122e34p-17, 0x1.17cfe3ff36c24p-15, 0x1.ca920bfc44d52p-16, -0x1.1e3981a6f9c48p-17, -0x1.7da584d6a576dp-15, -0x1.814e63b7610a8p-15, 0x1.1db6fd800303cp-61, 0x1.dfc07d20f23c2p-15, 0x1.27fe66a4acaf9p-14, 0x1.14d7e880cd436p-16, -0x1.1506fe26b2b3dp-14, -0x1.a743fef68b9c3p-14, -0x1.6afa1b449a5f8p-15, 0x1.20e66b6bc3ba5p-14, 0x1.1c7c70366411bp-13, 0x1.592ddf68300b4p-14, -0x1.03243adcb8afap-14, -0x1.694935bb0fa42p-13, -0x1.1ae587260ba5dp-13, 0x1.52871f8ecd672p-15, 0x1.b1e715ce7cc1fp-13, 0x1.a62325eb03034p-13, -0x1.9fb5bb4e860e4p-59, -0x1.eb391c51c5b68p-13, -0x1.25ce9137b0629p-12, -0x1.0ad507aa94583p-14, 0x1.03a789a835716p-12, 0x1.82517d687e82ep-12, 0x1.4301e6f6f0154p-13, -0x1.f5e4cad6f43b9p-13, -0x1.e2ef1bd59754cp-12, -0x1.1e923b9601f74p-12, 0x1.a5413893b328dp-13, 0x1.1fbbd16426c69p-11, 0x1.b9e1ee5a56bb6p-12, -0x1.037719eca8bc0p-13, -0x1.469a3533d569ap-11, -0x1.3841b373c76f9p-11, -0x1.b8d5a3ac267b7p-62, 0x1.5f8fde873bfc6p-11, 0x1.9e0520faf6036p-11, 0x1.725ecf00a9f08p-13, -0x1.632f33e9d9179p-11, -0x1.0485ac20bc98bp-10, -0x1.adbfc950eee29p-12, 0x1.497f1ecd8c236p-11, 0x1.39066d9a42c5ep-10, 0x1.6eee691c71ed1p-11, -0x1.0a7bb45b93ddcp-11, -0x1.67d82804f52c9p-10, -0x1.113b3cb149e0cp-10, 0x1.3d68900b2718bp-12, 0x1.8b5cd1b8de044p-10, 0x1.762c3aedb7881p-10, -0x1.46190f061af8fp-58, -0x1.9d32dfffb2f16p-10, -0x1.e2271a7755c0cp-10, -0x1.ab81349b7a389p-12, 0x1.96796649bb971p-10, 0x1.27af63fbabe00p-9, 0x1.e3e096ab12a81p-11, -0x1.7028a1068d501p-10, -0x1.5b2e287a6e57dp-9, -0x1.941764bb5a049p-10, 0x1.237b34bb31fbep-10, 0x1.870bf24036deep-9, 0x1.27156bc5a5c45p-9, -0x1.54c548a50e3dcp-11, -0x1.a6154217f2a54p-9, -0x1.8d57804c6b778p-9, 0x1.7b19eac8c5775p-56, 0x1.b28abdf896993p-9, 0x1.f8d6ebeb54e6fp-9, 0x1.bdce902b8ff3ep-11, -0x1.a64cc039e5d31p-9, -0x1.3228f3bb3037ep-8, -0x1.f3810ba43ec9cp-10, 0x1.7b09c33eeaafep-9, 0x1.64a014963c965p-8, 0x1.9e4d88271ebf5p-9, -0x1.2a68571039e0cp-9, -0x1.8fec5d1b9d09bp-8, -0x1.2d9b1a31867e5p-8, 0x1.5c442a30c3640p-10, 0x1.af8a6c367fab2p-8, 0x1.96a0750f96566p-8, -0x1.61248f4092797p-57, -0x1.be50367ffb038p-8, -0x1.03fa2b453afeep-7, -0x1.ccc0617e0fb18p-10, 0x1.b64e2bf3bda20p-8, 0x1.3f5edb8e5bd81p-7, 0x1.061408b9ca2f6p-8, -0x1.907f5d27588c7p-8, -0x1.7bd311dbadaa3p-7, -0x1.bd4b3d917bcf2p-8, 0x1.4414a0103bb39p-8, 0x1.b77dec5909b6ep-7, 0x1.4feac38d98947p-7, -0x1.89d06dc559729p-9, -0x1.f06d97965a21cp-7, -0x1.dcf1d76f0b0e5p-7, 0x1.a76e1784bf58ap-57, 0x1.125746de35d2dp-6, 0x1.48d29d0e58727p-6, 0x1.2cff044c43311p-8, -0x1.2932023a66a95p-6, -0x1.c41c710a6540fp-6, -0x1.85fdf11030babp-7, 0x1.3bf006fe95d5ep-6, 0x1.4125107a9071fp-5, 0x1.993cbe70373f5p-6, -0x1.49dcf574bb1b1p-6, -0x1.fc8b31079c167p-5, -0x1.cb41fa6093821p-5, 0x1.52708f9206b14p-6, 0x1.2cdc6bdec25ddp-3, 0x1.0ef4ccbed5787p-2, 0x1.4000a5fc3fb67p-2, 0x1.0ef4ccbed5787p-2, 0x1.2cdc6bdec25ddp-3, 0x1.52708f9206b14p-6, -0x1.cb41fa6093821p-5, -0x1.fc8b31079c167p-5, -0x1.49dcf574bb1b1p-6, 0x1.993cbe70373f5p-6, 0x1.4125107a9071fp-5, 0x1.3bf006fe95d5ep-6, -0x1.85fdf11030babp-7, -0x1.c41c710a6540fp-6, -0x1.2932023a66a95p-6, 0x1.2cff044c43311p-8, 0x1.48d29d0e58727p-6, 0x1.125746de35d2dp-6, 0x1.a76e1784bf58ap-57, -0x1.dcf1d76f0b0e5p-7, -0x1.f06d97965a21cp-7, -0x1.89d06dc559729p-9, 0x1.4feac38d98947p-7, 0x1.b77dec5909b6ep-7, 0x1.4414a0103bb39p-8, -0x1.bd4b3d917bcf2p-8, -0x1.7bd311dbadaa3p-7, -0x1.907f5d27588c7p-8, 0x1.061408b9ca2f6p-8, 0x1.3f5edb8e5bd81p-7, 0x1.b64e2bf3bda20p-8, -0x1.ccc0617e0fb18p-10, -0x1.03fa2b453afeep-7, -0x1.be50367ffb038p-8, -0x1.61248f4092797p-57, 0x1.96a0750f96566p-8, 0x1.af8a6c367fab2p-8, 0x1.5c442a30c3640p-10, -0x1.2d9b1a31867e5p-8, -0x1.8fec5d1b9d09bp-8, -0x1.2a68571039e0cp-9, 0x1.9e4d88271ebf5p-9, 0x1.64a014963c965p-8, 0x1.7b09c33eeaafep-9, -0x1.f3810ba43ec9cp-10, -0x1.3228f3bb3037ep-8, -0x1.a64cc039e5d31p-9, 0x1.bdce902b8ff3ep-11, 0x1.f8d6ebeb54e6fp-9, 0x1.b28abdf896993p-9, 0x1.7b19eac8c5775p-56, -0x1.8d57804c6b778p-9, -0x1.a6154217f2a54p-9, -0x1.54c548a50e3dcp-11, 0x1.27156bc5a5c45p-9, 0x1.870bf24036deep-9, 0x1.237b34bb31fbep-10, -0x1.941764bb5a049p-10, -0x1.5b2e287a6e57dp-9, -0x1.7028a1068d501p-10, 0x1.e3e096ab12a81p-11, 0x1.27af63fbabe00p-9, 0x1.96796649bb971p-10, -0x1.ab81349b7a389p-12, -0x1.e2271a7755c0cp-10, -0x1.9d32dfffb2f16p-10, -0x1.46190f061af8fp-58, 0x1.762c3aedb7881p-10, 0x1.8b5cd1b8de044p-10, 0x1.3d68900b2718bp-12, -0x1.113b3cb149e0cp-10, -0x1.67d82804f52c9p-10, -0x1.0a7bb45b93ddcp-11, 0x1.6eee691c71ed1p-11, 0x1.39066d9a42c5ep-10, 0x1.497f1ecd8c236p-11, -0x1.adbfc950eee29p-12, -0x1.0485ac20bc98bp-10, -0x1.632f33e9d9179p-11, 0x1.725ecf00a9f08p-13, 0x1.9e0520faf6036p-11, 0x1.5f8fde873bfc6p-11, -0x1.b8d5a3ac267b7p-62, -0x1.3841b373c76f9p-11, -0x1.469a3533d569ap-11, -0x1.037719eca8bc0p-13, 0x1.b9e1ee5a56bb6p-12, 0x1.1fbbd16426c69p-11, 0x1.a5413893b328dp-13, -0x1.1e923b9601f74p-12, -0x1.e2ef1bd59754cp-12, -0x1.f5e4cad6f43b9p-13, 0x1.4301e6f6f0154p-13, 0x1.82517d687e82ep-12, 0x1.03a789a835716p-12, -0x1.0ad507aa94583p-14, -0x1.25ce9137b0629p-12, -0x1.eb391c51c5b68p-13, -0x1.9fb5bb4e860e4p-59, 0x1.a62325eb03034p-13, 0x1.b1e715ce7cc1fp-13, 0x1.52871f8ecd672p-15, -0x1.1ae587260ba5dp-13, -0x1.694935bb0fa42p-13, -0x1.03243adcb8afap-14, 0x1.592ddf68300b4p-14, 0x1.1c7c70366411bp-13, 0x1.20e66b6bc3ba5p-14, -0x1.6afa1b449a5f8p-15, -0x1.a743fef68b9c3p-14, -0x1.1506fe26b2b3dp-14, 0x1.14d7e880cd436p-16, 0x1.27fe66a4acaf9p-14, 0x1.dfc07d20f23c2p-15, 0x1.1db6fd800303cp-61, -0x1.814e63b7610a8p-15, -0x1.7da584d6a576dp-15, -0x1.1e3981a6f9c48p-17, 0x1.ca920bfc44d52p-16, 0x1.17cfe3ff36c24p-15, 0x1.7e32331122e34p-17, -0x1.e2b8f2315372bp-17, -0x1.776cf1ec8130ep-16, -0x1.65bb6c6433d1cp-17, 0x1.a2e48bb3c21c5p-18, 0x1.c38b67162bf51p-17, 0x1.0e75556b8ae7ap-17, -0x1.e873aa50c7786p-20, -0x1.d03509843a83ap-18, -0x1.471881376ed95p-18};  // (-4.874106074688636e-06, -6.9172259637793684e-06, -1.8196248172184638e-06, 8.060286481890338e-06, 1.3457076004946537e-05, 6.241988490314871e-06, -1.0661248048298012e-05, -2.2377107528742542e-05, -1.4386249946564733e-05, 1.1390331140746e-05, 3.3356198395139114e-05, 2.733293148178318e-05, -8.530158835241605e-06, -4.549581506300871e-05, -4.593207944039837e-05, 4.840197077294325e-19, 5.719088426975979e-05, 7.057041018291775e-05, 1.6501152006760156e-05, -6.604845894312996e-05, -0.00010091438512269735, -4.3270227739377416e-05, 6.887914576857045e-05, 0.00013565353738132714, 8.22971320557666e-05, -6.178415373940183e-05, -0.0001722745781495717, -0.00013489560768805117, 4.0355661545293876e-05, 0.0002069009191952584, 0.00020129074923180975, -2.8169597820544543e-18, -0.00023423342150216274, -0.0002801960977017759, -6.361774163934045e-05, 0.00024762577460300925, 0.00036842185957619816, 0.00015402194523340496, -0.00023932157520658944, -0.0004605617699321065, -0.00027329561396571074, 0.000200869927189724, 0.0005488084102752813, 0.0004214120323433006, -0.00012372266737256492, -0.0006229445921046184, -0.0005955822839245054, -3.7340137009078004e-19, 0.0006705512795672712, 0.0007896805467408955, 0.00017660634332127717, -0.0006774604528172876, -0.0009938131626624238, -0.000409840741720765, 0.0006284648204140711, 0.00119409603081768, 0.0006998658969274656, -0.0005082764068421969, -0.0013726972986109357, -0.0010422950547964742, 0.00030270428460157293, 0.0015081885341867865, 0.001427355858531182, -4.419450448717684e-18, -0.0015762280672104215, -0.001839266767850625, -0.00040770026109143064, 0.0015505760842945105, 0.002255898432700354, 0.0009229227047254513, -0.001404414013120914, -0.0026487755214616424, -0.001541486285384879, 0.0011119128164853577, 0.002983449288974197, 0.0022513097818080163, -0.0006499684137742895, -0.0032202380352590963, -0.0030314773668668714, 2.055111933351121e-17, 0.0033152920625926867, 0.00385162000535804, 0.0008503091571085764, -0.003221891846759605, -0.004671630387009238, -0.0019054568629702012, 0.002891831477844219, 0.00544167044266639, 0.003160879965034886, -0.0022766691473525887, -0.0061023452104686194, -0.0046021403263924775, 0.0013285303832079182, 0.006584788722192302, 0.006204632357900896, -9.571955068182978e-18, -0.006810200982709395, -0.007933875206127507, -0.001757627439942849, 0.006688009005066081, 0.00974641533386378, 0.0039989968588904825, -0.0061111071039212156, -0.011591323585304741, -0.006794645816156507, 0.00494507701681251, 0.013412228016091734, 0.010251374707142559, -0.0030045637128520886, -0.015149783150904254, -0.01455519694544489, 1.1477094404101066e-17, 0.016744441229858232, 0.020069745421444154, 0.004592836903850609, -0.018139364410311904, -0.027594671622484464, -0.011901610094701661, 0.01928330119184573, 0.03920224398144811, 0.02497786138052401, -0.0201332470923476, -0.06207809043997053, -0.05606173421675487, 0.020656719403727555, 0.1469047953944899, 0.2646057120776892, 0.3125024733744141, 0.2646057120776892, 0.1469047953944899, 0.020656719403727555, -0.05606173421675487, -0.06207809043997053, -0.0201332470923476, 0.02497786138052401, 0.03920224398144811, 0.01928330119184573, -0.011901610094701661, -0.027594671622484464, -0.018139364410311904, 0.004592836903850609, 0.020069745421444154, 0.016744441229858232, 1.1477094404101066e-17, -0.01455519694544489, -0.015149783150904254, -0.0030045637128520886, 0.010251374707142559, 0.013412228016091734, 0.00494507701681251, -0.006794645816156507, -0.011591323585304741, -0.0061111071039212156, 0.0039989968588904825, 0.00974641533386378, 0.006688009005066081, -0.001757627439942849, -0.007933875206127507, -0.006810200982709395, -9.571955068182978e-18, 0.006204632357900896, 0.006584788722192302, 0.0013285303832079182, -0.0046021403263924775, -0.0061023452104686194, -0.0022766691473525887, 0.003160879965034886, 0.00544167044266639, 0.002891831477844219, -0.0019054568629702012, -0.004671630387009238, -0.003221891846759605, 0.0008503091571085764, 0.00385162000535804, 0.0033152920625926867, 2.055111933351121e-17, -0.0030314773668668714, -0.0032202380352590963, -0.0006499684137742895, 0.0022513097818080163, 0.002983449288974197, 0.0011119128164853577, -0.001541486285384879, -0.0026487755214616424, -0.001404414013120914, 0.0009229227047254513, 0.002255898432700354, 0.0015505760842945105, -0.00040770026109143064, -0.001839266767850625, -0.0015762280672104215, -4.419450448717684e-18, 0.001427355858531182, 0.0015081885341867865, 0.00030270428460157293, -0.0010422950547964742, -0.0013726972986109357, -0.0005082764068421969, 0.0006998658969274656, 0.00119409603081768, 0.0006284648204140711, -0.000409840741720765, -0.0009938131626624238, -0.0006774604528172876, 0.00017660634332127717, 0.0007896805467408955, 0.0006705512795672712, -3.7340137009078004e-19, -0.0005955822839245054, -0.0006229445921046184, -0.00012372266737256492, 0.0004214120323433006, 0.0005488084102752813, 0.000200869927189724, -0.00027329561396571074, -0.0004605617699321065, -0.00023932157520658944, 0.00015402194523340496, 0.00036842185957619816, 0.00024762577460300925, -6.361774163934045e-05, -0.0002801960977017759, -0.00023423342150216274, -2.8169597820544543e-18, 0.00020129074923180975, 0.0002069009191952584, 4.0355661545293876e-05, -0.00013489560768805117, -0.0001722745781495717, -6.178415373940183e-05, 8.22971320557666e-05, 0.00013565353738132714, 6.887914576857045e-05, -4.3270227739377416e-05, -0.00010091438512269735, -6.604845894312996e-05, 1.6501152006760156e-05, 7.057041018291775e-05, 5.719088426975979e-05, 4.840197077294325e-19, -4.593207944039837e-05, -4.549581506300871e-05, -8.530158835241605e-06, 2.733293148178318e-05, 3.3356198395139114e-05, 1.1390331140746e-05, -1.4386249946564733e-05, -2.2377107528742542e-05, -1.0661248048298012e-05, 6.241988490314871e-06, 1.3457076004946537e-05, 8.060286481890338e-06, -1.8196248172184638e-06, -6.9172259637793684e-06, -4.874106074688636e-06)
inline constexpr double FE_FLAT_DB = 0x1.999999999999ap-4;  // 0.1
inline constexpr double FE_MIN_HZ = 0x1.8000000000000p+1;  // 3.0
inline constexpr double FE_SKIRT_DB = 0x1.4000000000000p+4;  // 20.0
inline constexpr int F_MIX_DEN = 64;
inline constexpr double PB_PACK_AFTER_S = 0x1.e000000000000p+6;  // 120.0
inline constexpr double PB_LEAD_S = 0x1.e000000000000p+3;  // 15.0
inline constexpr double PB_TAIL_S = 0x1.4000000000000p+2;  // 5.0
// not emitted: _HOUR_RE (Pattern)
}  // namespace frontend

// --- sstvae/qrss/acquire.py ---
namespace acquire {
inline constexpr double STFT_FRAME_S = 0x1.0000000000000p+2;  // 4.0
inline constexpr double STFT_HOP_S = 0x1.0000000000000p+1;  // 2.0
inline constexpr std::array<double, 2> BAND_HZ{0x1.2c00000000000p+8, 0x1.5180000000000p+11};  // (300.0, 2700.0)
inline constexpr double LN2 = 0x1.62e42fefa39efp-1;  // 0.6931471805599453
inline constexpr double CFAR_LOCAL_HZ = 0x1.4000000000000p+5;  // 40.0
inline constexpr int CFAR_LOCAL_BINS = 321;
inline constexpr std::array<double, 2> A1_SPAN_S{-0x1.4000000000000p+1, 0x1.6800000000000p+4};  // (-2.5, 22.5)
inline constexpr std::array<double, 13> A1_DRIFTS{-0x1.8000000000000p+1, -0x1.4000000000000p+1, -0x1.0000000000000p+1, -0x1.8000000000000p+0, -0x1.0000000000000p+0, -0x1.0000000000000p-1, 0.0, 0x1.0000000000000p-1, 0x1.0000000000000p+0, 0x1.8000000000000p+0, 0x1.0000000000000p+1, 0x1.4000000000000p+1, 0x1.8000000000000p+1};  // (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
inline constexpr double A1_PFA = 0x1.a36e2eb1c432dp-14;  // 0.0001
inline constexpr int A1_MAX_CANDS = 16;
inline constexpr double A1_MIN_SEP_HZ = 0x1.4000000000000p+3;  // 10.0
inline constexpr double A1_DRIFT_MARGIN = 0x1.999999999999ap-4;  // 0.1
inline constexpr double A2_TAU_S = 0x1.4000000000000p+1;  // 2.5
inline constexpr std::array<double, 21> A2_DF_HZ{-0x1.0000000000000p-1, -0x1.ccccccccccccdp-2, -0x1.999999999999ap-2, -0x1.6666666666666p-2, -0x1.3333333333333p-2, -0x1.0000000000000p-2, -0x1.999999999999ap-3, -0x1.3333333333333p-3, -0x1.999999999999ap-4, -0x1.999999999999ap-5, -0.0, 0x1.999999999999ap-5, 0x1.999999999999ap-4, 0x1.3333333333333p-3, 0x1.999999999999ap-3, 0x1.0000000000000p-2, 0x1.3333333333333p-2, 0x1.6666666666666p-2, 0x1.999999999999ap-2, 0x1.ccccccccccccdp-2, 0x1.0000000000000p-1};  // (-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
inline constexpr int A2_CHUNK = 66;
inline constexpr double A2_PFA = 0x1.ad7f29abcaf48p-24;  // 1e-07
inline constexpr double NOTCH_HZ = 0x1.3333333333333p-1;  // 0.6
inline constexpr int NOTCH_TAPS = 2001;
inline constexpr double A2_MARGIN_S_ = 0x1.8000000000000p+2;  // 6.0
inline constexpr double A2_DRIFT_SIGMAS = 0x1.8000000000000p+1;  // 3.0
inline constexpr double A2_LOCAL_S = 0x1.8000000000000p+2;  // 6.0
inline constexpr std::array<int, 4> A2_SCALES{1, 2, 5, 10};
inline constexpr std::array<int, 2> A2_ALIAS_K{1, 2};
inline constexpr double A2_ALIAS_HZ = 0x1.0000000000000p-1;  // 0.5
inline constexpr double A2_ALIAS_RATIO = 0x1.8000000000000p+1;  // 3.0
inline constexpr std::array<double, 2> L_BAND_HZ{0x1.8cccccccccccdp+1, 0x1.611aeb6a1847dp+4};  // (3.1, 22.069072164948455)
inline constexpr std::array<double, 2> L_OWN_HZ{0x1.8cccccccccccdp+1, 0x1.619999999999ap+4};  // (3.1, 22.1)
inline constexpr double L_OWN_T = 0x1.8000000000000p+1;  // 3.0
inline constexpr std::array<double, 3> L_HINT_OFFSETS_HZ{0.0, -0x1.0000000000000p+0, 0x1.0000000000000p+0};  // (0.0, -1.0, 1.0)
inline constexpr double WAVEFORM_MERGE_HZ = 0x1.4000000000000p+2;  // 5.0
inline constexpr double TBD_START_S = -0x1.4000000000000p+3;  // -10.0
inline constexpr double HINT_SPAN_HZ = 0x1.0000000000000p+1;  // 2.0
inline constexpr double TBD_END_PAD_S = 0x1.0000000000000p+1;  // 2.0
inline constexpr double TBD_HALF_HZ = 0x1.4000000000000p+5;  // 40.0
inline constexpr double TBD_GRID_HZ = 0x1.9000000000000p+5;  // 50.0
inline constexpr int TBD_MAX_STEP = 2;
inline constexpr double TBD_COST = 0x1.0000000000000p-1;  // 0.5
inline constexpr double TBD_PFA = 0x1.0624dd2f1a9fcp-10;  // 0.001
inline constexpr double TBD_DEDUP_HZ = 0x1.4000000000000p+3;  // 10.0
inline constexpr double SHAPE_L_Z = 0x1.0000000000000p+2;  // 4.0
inline constexpr double SHAPE_CE_ABOVE_Z = 0x1.8000000000000p+1;  // 3.0
inline constexpr double SHAPE_CE_MAX_Z1 = 0x1.8000000000000p+1;  // 3.0
inline constexpr std::array<double, 2> L_ANCHOR_HZ{-0x1.8000000000000p+2, 0x1.a000000000000p+4};  // (-6.0, 26.0)
inline constexpr std::array<int, 4> TBD_GUMBEL_N{31, 132, 331, 896};
inline constexpr std::array<double, 4> TBD_GUMBEL_MU{0x1.3b0a3d70a3d71p+3, 0x1.2a3126e978d50p+4, 0x1.cec8b43958106p+4, 0x1.7a74bc6a7ef9ep+5};  // (9.845, 18.637, 28.924, 47.307)
inline constexpr std::array<double, 4> TBD_GUMBEL_BETA{0x1.676c8b4395810p-1, 0x1.410624dd2f1aap-1, 0x1.3f7ced916872bp-1, 0x1.33b645a1cac08p-1};  // (0.702, 0.627, 0.624, 0.601)
inline constexpr double LEAD_WIN_S = 0x1.0000000000000p+0;  // 1.0
inline constexpr double LEAD_Z = 0x1.0000000000000p+2;  // 4.0
// not emitted: CE_PRE (PreambleKind), L_PRE (PreambleKind), PREAMBLE_KINDS (dict)
}  // namespace acquire

// --- sstvae/qrss/demod.py ---
namespace demod {
inline constexpr double D_PASS = 0x1.7dbf487fcb924p-5;  // 0.0466
inline constexpr double D_REF = 0x1.26e978d4fdf3bp-7;  // 0.009
inline constexpr std::array<double, 2> KAPPA_RANGE{0x1.0000000000000p-1, 0x1.0000000000000p+1};  // (0.5, 2.0)
inline constexpr double KAPPA_PRIOR_SD = 0x1.0000000000000p-1;  // 0.5
inline constexpr double KAPPA_S2_MAX = 0x1.0000000000000p+2;  // 4.0
inline constexpr double KAPPA_S2_REL = 0x1.0000000000000p+2;  // 4.0
inline constexpr double U_MIN_REL = 0x1.a36e2eb1c432dp-14;  // 0.0001
}  // namespace demod

// --- sstvae/qrss/ldemod.py ---
namespace ldemod {
inline constexpr double G_REF_L = 0x1.0000000000000p+0;  // 1.0
inline constexpr double G_HDR_L = 0x1.fdf3b645a1cacp-1;  // 0.996
inline constexpr double G_DATA_L = 0x1.c3126e978d4fep-1;  // 0.881
inline constexpr double G_PRE_L = 0x1.fd70a3d70a3d7p-1;  // 0.995
inline constexpr double PILOT_GAIN_PRE = 0x1.4fdf3b645a1cbp-1;  // 0.656
inline constexpr double PILOT_GAIN_HDR = 0x1.53f7ced916873p-1;  // 0.664
inline constexpr double PILOT_GAIN_DATA = 0x1.67ef9db22d0e5p-1;  // 0.703
inline constexpr double PILOT_GAIN_ID = 0x1.0000000000000p+0;  // 1.0
inline constexpr double G_REF_ABS = 0x1.0189374bc6a7fp+0;  // 1.006
inline constexpr double D_PASS_L = 0x1.db22d0e560419p-5;  // 0.058
inline constexpr double D_REF_L = 0x1.82a9930be0dedp-6;  // 0.0236
inline constexpr double D_HDR_L = 0x1.495182a9930bep-6;  // 0.0201
inline constexpr double D_PRE_L = 0x1.47ae147ae147bp-6;  // 0.02
inline constexpr double D_PASS_LX = 0x1.f4cb1897a67a6p-5;  // 0.061132000000000006
}  // namespace ldemod

// --- sstvae/qrss/cwid.py ---
namespace cwid {
inline constexpr std::array<int, 13> PLAIN_UNITS{0, 1, 2, 3, 4, 184, 185, 186, 187, 188, 189, 190, 191};
inline constexpr int AMP_SMOOTH_UNITS = 15;
inline constexpr double READ_LLR_MIN = 0x1.8000000000000p+2;  // 6.0
inline constexpr std::array<std::string_view, 37> CHARS_{"/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
// not emitted: _INV (dict), CALL_UNITS (slice), _PATS (list)
}  // namespace cwid

// --- sstvae/qrss/receiver.py ---
namespace receiver {
inline constexpr double MERGE_HZ = 0x1.4000000000000p+2;  // 5.0
inline constexpr double GHOST_HZ = 0x1.4000000000000p+4;  // 20.0
inline constexpr double GHOST_T = 0x1.8000000000000p+1;  // 3.0
inline constexpr double GHOST_Z_RATIO = 0x1.999999999999ap-4;  // 0.1
inline constexpr double ALIAS_HZ = 0x1.0000000000000p-1;  // 0.5
inline constexpr double ALIAS_RATIO = 0x1.8000000000000p+1;  // 3.0
inline constexpr double MARGIN_S = 0x1.8000000000000p+1;  // 3.0
inline constexpr double L_EXCLUDE_OFFSET_HZ = -0x1.3000000000000p+3;  // -9.5
inline constexpr double L_OWN_BELOW_HZ = 0x1.619999999999ap+4;  // 22.1
inline constexpr double L_OWN_ABOVE_HZ = 0x1.8cccccccccccdp+1;  // 3.1
inline constexpr double CE_RESID_MIN = 0x1.0000000000000p+1;  // 2.0
inline constexpr std::array<int, 1> L_HEADER_VERSIONS{2};
inline constexpr int CONF_BINS = 120;
inline constexpr double CONF_FLOOR_DB = -0x1.e000000000000p+4;  // -30.0
}  // namespace receiver

// --- sstvae/qrss/render.py ---
namespace render {
inline constexpr double G_MAX = 0x1.91eb851eb851fp-1;  // 0.785
inline constexpr double G_KNEE = 0x1.8f5c28f5c28f6p-2;  // 0.39
inline constexpr int N_GROUPS = 3;
inline constexpr int GROUP_LATENTS = 52800;
}  // namespace render

// --- sstvae/qrss/rules.py ---
namespace rules {
inline constexpr std::string_view EDGE_NOISE = "QRSS_OLD_EDGE_NOISE";
inline constexpr std::string_view GATING = "QRSS_OLD_GATING";
inline constexpr std::array<std::string_view, 2> RULES{"QRSS_OLD_EDGE_NOISE", "QRSS_OLD_GATING"};
}  // namespace rules

// --- sstvae/qrss/ce.py ---
namespace ce {
inline constexpr double ALPHA_ = 0x1.3333333333333p-3;  // 0.15
inline constexpr double BETA_ = 0x1.999999999999ap-1;  // 0.8
inline constexpr int GRID_FS = 16000;
inline constexpr int HALF = 3880;
inline constexpr int OOK_FIRST_KEYED_UNIT = 5;
inline constexpr int OOK_LAST_KEYED_UNIT = 183;
inline constexpr double FSK_SHIFT_HZ = -0x1.07eae2f8151d1p+4;  // -16.494845360824744
inline constexpr double PULSE_NORM = 0x1.fff4b07fc17c3p-1;  // 0.999913707347098
inline constexpr double G0 = 0x1.006d1ed38b121p+0;  // 1.0016650454009055
// not emitted: _M (ndarray), PULSE_TABLE (ndarray), _PULSE_TABLE_SQ (ndarray)
}  // namespace ce

// --- sstvae/qrss/lwave.py ---
namespace lwave {
inline constexpr int GRID_FS = 16000;
inline constexpr int HALF = 7760;
inline constexpr int CLIP_FS = 250;
inline constexpr double PULSE_NORM = 0x1.fff4afe37430dp-1;  // 0.9999136891511583
inline constexpr double G0 = 0x1.006cb85d5f6d6p+0;  // 1.001658938213429
inline constexpr double PF_ = 0x1.3333333333333p-3;  // 0.15
inline constexpr double A_PILOT = 0x1.ae2a81b75e576p-2;  // 0.42008402520840293
inline constexpr double A_CLIP = 0x1.15abc5bd2feafp+0;  // 1.0846522890932808
inline constexpr double L_AVG_POWER = 0x1.428f5c28f5c29p-1;  // 0.63
inline constexpr double A_ID = 0x1.96632cbc9f77fp-1;  // 0.7937253933193772
inline constexpr int ID_RAMP_UNITS = 8;
inline constexpr double KEY_RAMP_S = 0x1.999999999999ap-5;  // 0.05
inline constexpr int INTERP_HALF = 240;
// not emitted: _M (ndarray), PULSE_TABLE (ndarray)
}  // namespace lwave

}  // namespace sstvae::qrss::rxconfig
