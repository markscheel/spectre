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

/// Holds functions related to transforming between frames.
namespace transform {

/// Transforms tensor to different frame.  Note that
/// `Jacobian<Dest,Src>` is the same type as `InverseJacobian<Src,Dest>`
/// and represents \f$ dx^{\rm src}/dx^{\rm dest}\f$.
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataVector, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept;

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
auto to_different_frame(const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
                        const Jacobian<DataVector, VolumeDim, DestFrame,
                                       SrcFrame>& jacobian) noexcept
    -> tnsr::ii<DataVector, VolumeDim, DestFrame>;

/// Transforms only the first index to different frame.  Note that
/// `Jacobian<Dest,Src>` is the same type as `InverseJacobian<Src,Dest>`
/// and represents \f$ dx^{\rm src}/dx^{\rm dest}\f$.
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void first_index_to_different_frame(
    const gsl::not_null<tnsr::ijj<DataVector, VolumeDim, DestFrame>*> dest,
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept;

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
auto first_index_to_different_frame(
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept -> tnsr::ijj<DataVector, VolumeDim, DestFrame>;

}  // namespace transform
