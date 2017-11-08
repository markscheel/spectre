// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperInitializerSphere.hpp"

namespace StrahlkorperInitializers {

template <typename Fr>
Sphere<Fr>::Sphere(const size_t l_max, const double radius,
                   std::array<double, 3> center,
                   const OptionContext& /*context*/) noexcept
    // clang-tidy: no std::move on trivially-copyable type.
    : l_max_(l_max),
      radius_(radius),
      center_(std::move(center)) {}  // NOLINT

template <typename Fr>
Strahlkorper<Fr> Sphere<Fr>::create_strahlkorper() const noexcept {
  return Strahlkorper<Fr>(l_max_, l_max_, radius_, center_);
}

}  // namespace StrahlkorperInitializers

// ================================================================

namespace StrahlkorperInitializers {
template class Sphere<Frame::Inertial>;
}
