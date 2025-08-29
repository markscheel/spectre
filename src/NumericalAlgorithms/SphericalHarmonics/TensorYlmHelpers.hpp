// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

#include <array>
#include <complex>

#include "Utilities/Array.hpp"

namespace ylm::TensorYlm::helpers {

enum class BasisVector { x, y, z, l, m, mbar };

/// Returns a Cartesian BasisVector for every index.
template <size_t SIZE>
std::array<BasisVector, SIZE> to_cart_basis_vector(
    const cpp20::array<size_t, SIZE>& indices);

/// Returns the m value (m3 in the 2nd Wigner 3j symbol) associated with a
/// Cartesian basis vector.
int bv_to_m(BasisVector basis_vector, int i);

/// Returns the prefactor k associated with a Cartesian basis vector.
std::complex<double> bv_to_k(BasisVector basis_vector, int i);

}  // namespace ylm::TensorYlm::helpers
