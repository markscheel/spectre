// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial Christoffel symbol of the 2nd kind from the
 * the generalized harmonic spatial derivative variable and the
 * inverse spatial metric.
 *
 * \details
 * If \f$ \Phi_{kab} \f$ is the generalized
 * harmonic spatial derivative variable and \f$g^{ij}\f$ is the inverse
 * spatial metric, the Christoffel symbols are
 * \f[
 *      \Gamma^l_{ij} = \frac{1}{2}g^{lk}(\Phi_{ijk}+\Phi_{jik}-\Phi_{kij}).
 * \f]
 *
 * In the not_null version, no memory allocations are performed if the
 * output tensor already has the correct size.
 *
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void christoffel_second_kind(
    gsl::not_null<tnsr::Ijj<DataType, SpatialDim, Frame>*> christoffel,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
auto christoffel_second_kind(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric) noexcept
    -> tnsr::Ijj<DataType, SpatialDim, Frame>;
}  // namespace GeneralizedHarmonic
