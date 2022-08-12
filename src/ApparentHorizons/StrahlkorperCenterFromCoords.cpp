// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperCenterFromCoords.hpp"

#include <array>
#include <numeric>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename Frame>
std::array<double, 3> strahlkorper_center_from_coords(
    const tnsr::I<DataVector, 3, Frame>& cartesian_coords) {
  std::array<double, 3> center{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(center, d) = std::accumulate(cartesian_coords.get(d).begin(),
                                         cartesian_coords.get(d).end(), 0.0) /
                         cartesian_coords.get(d).size();
  }
  return center;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                     \
  template void strahlkorper_in_different_frame( \
      const tnsr::I<DataVector, 3, Frame>& cartesian_coords);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Inertial, ::Frame::Distorted))

#undef INSTANTIATE
#undef FRAME
