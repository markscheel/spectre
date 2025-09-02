// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <cstddef>
#include <optional>

#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Fills a blaze::CompressedMatrix that does a TensorYlm filter
 * operation.
 *
 * Assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * blaze::CompressedMatrix by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:Filter}).
 *
 * Assumes that the components of the Tensor<DataVector> are stored
 * contiguous in memory, in order of the storage_index of the Tensor.
 * Also assumes that the stride is unity.  In this way, we are able to
 * construct a blaze::CustomVector pointing to the beginning of the
 * first element of $T^{\tilde A}_{\ell' m'}$ and multiply that by the
 * blaze::CompressedMatrix we compute here, and get the result.
 *
 * This memory layout here is different than in SpEC.  In SpEC, each
 * tensor component is stored in separately-allocated memory, so this
 * equivalent function in SpEC fills $N^2$ sparse matrices, where $N$
 * is the number of independent components of the Tensor. The
 * advantage of the SpEC method is that each sparse matrix is smaller,
 * so sorting elements into the correct order while constructing each
 * sparse matrix is faster (sorting is > linear in the number of
 * matrix elements).  The disadvantage of the SpEC method is that
 * evaluating the filter for a single Tensor involves $N^2$
 * matrix-vector multiplications, whereas here evaluating the filter
 * involves only one matrix-vector multiplication, which should have
 * more efficient memory access.  It is not clear which method is
 * faster overall without profiling.
 *
 * If half_power is std::nullopt, implements a Heaviside filter:
 * Given src as a tensor of scalar-Ylm coefficients of Cartesian
 * components, transforms to spin-weighted harmonic coefficients,
 * zeroes the top number_of_ell_modes_to_kill ell modes, and transforms
 * back.  This could be implemented in terms of SpinWeightedToCartesian
 * and CartesianToSpinWeighted, but it is often more efficient to implement
 * this as its own function by simplifying the expressions analytically.
 *
 * If half_power is not std::nullopt, then the filter is the smooth
 * version of the Heaviside function described in the TensorYlm
 * namespace documentation, with $\sigma$ equal to half_power and
 * $\ell_{\mathrm{cut}}^+$ equal to $\ell_{\rm max}$ minus
 * number_of_ell_modes_to_kill.
 *
 * \tparam TensorStructure A Tensor_detail::Structure
 *
 * \param matrix The CompressedMatrix to fill
 * \param structure A Tensor_detail::Structure passed in for template deduction.
 * \param ell_max The maximum ylm ell value.
 * \param number_of_ell_modes_to_kill How many top ell modes to set to zero.
 * \param half_power The half power $\sigma$ for more complicated filtering.
 */
template <typename TensorStructure>
void FillFilter(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    TensorStructure /*structure*/, size_t ell_max,
    size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power);

};  // namespace ylm::TensorYlm
