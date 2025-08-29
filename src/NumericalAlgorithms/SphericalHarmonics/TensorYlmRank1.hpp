// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Fills a SparseMatrix that does a TensorYlm filter operation.
 *
 * Assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * SparseMatrixCollection by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:Filter}).
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
 * version of the Heaviside function described above, with $\sigma$
 * equal to half_power and $\ell_{\mathrm{cut}}^+$ equal to
 * $\ell_{\rm max}$ minus number_of_ell_modes_to_kill.
 */
void FillFilter(gsl::not_null<SparseMatrixCollection*> matrices, size_t ell_max,
                size_t number_of_ell_modes_to_kill,
                std::optional<size_t> half_power);

};  // namespace ylm::TensorYlm
