// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// Holds functions and tags related to transforming between frames.
namespace transform {

/// Transforms tensor to different frame.  Note that
/// Jacobian<Dest,Src> is the same type as InverseJacobian<Src,Dest>.
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataVector, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept;

namespace Tags {
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
struct Jacobian : db::SimpleTag {
  using type = ::Jacobian<DataVector, VolumeDim, SrcFrame, DestFrame>;
};
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
struct InverseJacobian : db::SimpleTag {
  using type = ::InverseJacobian<DataVector, VolumeDim, SrcFrame, DestFrame>;
};
}  // namespace Tags

}  // namespace transform
