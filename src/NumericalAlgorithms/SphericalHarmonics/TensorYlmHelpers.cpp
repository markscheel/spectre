// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"

#include <array>
#include <complex>

#include "Utilities/Array.hpp"

namespace ylm::TensorYlm::helpers {

template <size_t SIZE>
std::array<BasisVector, SIZE> to_cart_basis_vector(
    const cpp20::array<size_t, SIZE>& indices) {
  std::array<BasisVector, SIZE> result;
  for (size_t i = 0; i < SIZE; ++i) {
    switch (indices[i]) {
      case 0:
        result[i] = BasisVector::x;
        break;
      case 1:
        result[i] = BasisVector::y;
        break;
      case 2:
        result[i] = BasisVector::z;
        break;
      default:
        ASSERT(false, "Cannot get here");
    }
  }
  return result;
}

int bv_to_m(const BasisVector basis_vector, const int i) {
  switch (basis_vector) {
    case BasisVector::z:
      return 0.0;
    case BasisVector::y:
    case BasisVector::x:
      return i;
    default:
      ASSERT(false, "Unknown basisvector");
  }
}

std::complex<double> bv_to_k(const BasisVector basis_vector, const int i) {
  switch (basis_vector) {
    case BasisVector::z:
      return {1.0 / sqrt(2.0), 0.0};
    case BasisVector::y:
      return {0, 1};
    case BasisVector::x:
      return {double(-i), 0.0};
    default:
      ASSERT(false, "Unknown basisvector");
  }
}

// Explicit instantiations
template std::array<BasisVector, 1> to_cart_basis_vector(
    const cpp20::array<size_t, 1>& indices);
template std::array<BasisVector, 2> to_cart_basis_vector(
    const cpp20::array<size_t, 2>& indices);
template std::array<BasisVector, 3> to_cart_basis_vector(
    const cpp20::array<size_t, 3>& indices);

}  // namespace ylm::TensorYlm::helpers
