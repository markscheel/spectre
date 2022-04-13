// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace intrp::OptionHolders {

template <size_t VolumeDim>
LineSegment<VolumeDim>::LineSegment(std::array<double, VolumeDim> begin_in,
                                    std::array<double, VolumeDim> end_in,
                                    size_t number_of_points_in,
                                    ::Verbosity verbosity_in)
    : begin(std::move(begin_in)),  // NOLINT
      end(std::move(end_in)),      // NOLINT
      number_of_points(number_of_points_in),
      verbosity(std::move(verbosity_in)) {}  // NOLINT
// above NOLINT for std::move of trivially copyable type.

template <size_t VolumeDim>
void LineSegment<VolumeDim>::pup(PUP::er& p) {
  p | begin;
  p | end;
  p | number_of_points;
  p | verbosity;
}

template <size_t VolumeDim>
bool operator==(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) {
  return lhs.begin == rhs.begin and lhs.end == rhs.end and
         lhs.number_of_points == rhs.number_of_points and
         lhs.verbosity == rhs.verbosity;
}

template <size_t VolumeDim>
bool operator!=(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                               \
  template struct LineSegment<DIM(data)>;                  \
  template bool operator==(const LineSegment<DIM(data)>&,  \
                           const LineSegment<DIM(data)>&); \
  template bool operator!=(const LineSegment<DIM(data)>&,  \
                           const LineSegment<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace intrp::OptionHolders
