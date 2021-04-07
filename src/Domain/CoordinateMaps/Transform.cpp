// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Transform.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace transform {

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataVector, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept {
  destructive_resize_components(dest, src.begin()->size());
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // symmetry
      dest->get(i, j) = 0.0;
      for (size_t k = 0; k < VolumeDim; ++k) {
        for (size_t p = 0; p < VolumeDim; ++p) {
          dest->get(i, j) +=
              jacobian.get(k, i) * jacobian.get(p, j) * src.get(k, p);
        }
      }
    }
  }
}

}  // namespace transform

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define SRCFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DESTFRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void transform::to_different_frame(                                \
      const gsl::not_null<tnsr::ii<DataVector, DIM(data), DESTFRAME(data)>*>  \
          dest,                                                               \
      const tnsr::ii<DataVector, DIM(data), SRCFRAME(data)>& src,             \
      const Jacobian<DataVector, DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian) noexcept;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid),
                        (Frame::Inertial))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial),
                        (Frame::Grid))

#undef DIM
#undef SRCFRAME
#undef DESTFRAME
#undef INSTANTIATE
