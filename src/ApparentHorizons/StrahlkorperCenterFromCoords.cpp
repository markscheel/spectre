// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperCenterFromCoords.hpp"

#include <array>
#include <numeric>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename Frame>
std::array<double, 3> approx_strahlkorper_center(
    const tnsr::I<DataVector, 3, Frame>& cartesian_coords_on_surface) {
  // Simply averages the coordinates on the surface.
  // This is why it is approximate.
  std::array<double, 3> center{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(center, d) =
        std::accumulate(cartesian_coords_on_surface.get(d).begin(),
                        cartesian_coords_on_surface.get(d).end(), 0.0) /
        cartesian_coords_on_surface.get(d).size();
  }
  return center;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                \
  template void approx_strahlkorper_center( \
      const tnsr::I<DataVector, 3, Frame>& cartesian_coords_on_surface);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Inertial, ::Frame::Distorted))

#undef INSTANTIATE
#undef FRAME
