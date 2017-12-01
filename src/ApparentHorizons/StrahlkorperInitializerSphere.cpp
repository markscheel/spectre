// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperInitializerSphere.hpp"

namespace StrahlkorperInitializers {

template <typename Frame>
Sphere<Frame>::Sphere(const size_t l_max, const double radius,
                      std::array<double, 3> center,
                      const OptionContext& /*context*/) noexcept
    // clang-tidy: no std::move on trivially-copyable type.
    : l_max_(l_max),
      radius_(radius),
      center_(std::move(center)) {}  // NOLINT

template <typename Frame>
Strahlkorper<Frame> Sphere<Frame>::create_strahlkorper() const noexcept {
  if (l_max_ < 2) {
    ERROR("Cannot create a spherical Strahlkorper with l_max<2. You used "
          << l_max_);
  }
  return Strahlkorper<Frame>(l_max_, l_max_, radius_, center_);
}

}  // namespace StrahlkorperInitializers

namespace StrahlkorperInitializers {
template class Sphere<Frame::Inertial>;
}  // namespace StrahlkorperInitializers
