#include "qrss/codec/decode.hpp"

#include <span>

namespace sstvae::qrss::codec {

images::Picture render_planes(const OpenDecoder& decoder, const decode::Planes& planes) {
    return render(decoder, std::span<const double>(planes.S), std::span<const double>(planes.W), planes.mode);
}

images::Picture render_accumulator(const OpenDecoder& decoder, const store::Accumulator& acc) {
    return render_planes(decoder, decode::planes_from_accumulator(acc));
}

}  // namespace sstvae::qrss::codec
