// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <iostream>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename TensorStructure>
void test_tensorylm_filter_vs_spec(const size_t ell_max,
                                   const size_t number_of_ell_modes_to_kill,
                                   const std::optional<size_t> half_power) {
  const std::vector<double> spec_matrix_elements =
      [ell_max, number_of_ell_modes_to_kill, half_power]() {
        // These are numbers that Mark Scheel generated using SpEC's
        // TensorYlmFilter for various special cases.
        if (ell_max == 1 and number_of_ell_modes_to_kill == 0 and
            not half_power.has_value()) {
          return {-0.5,
                  0.7071067811865475,
                  -0.6666666666666667,
                  -0.3333333333333335,
                  -0.2357022603955158,
                  0.5,
                  -0.5000000000000001,
                  0.5000000000000001,
                  0.2357022603955158,
                  -0.5,
                  -0.7071067811865475,
                  0.5000000000000002,
                  -0.5000000000000001,
                  0.2357022603955158,
                  0.5,
                  -0.3333333333333335,
                  -0.6666666666666667,
                  0.2357022603955158,
                  -0.4714045207910316,
                  0.4714045207910316,
                  -0.6666666666666665,
                  0.3535533905932737,
                  -0.3535533905932737,
                  -0.4999999999999999,
                  0.6666666666666665,
                  -0.3535533905932737,
                  -0.3535533905932737,
                  -0.4999999999999999};
        }
      }();

  const std::vector<size_t> spec_src_indices =
      [ell_max, number_of_ell_modes_to_kill, half_power]() {
        // These are numbers that Mark Scheel generated using SpEC's
        // TensorYlmFilter for various special cases.
        if (ell_max == 1 and number_of_ell_modes_to_kill == 0 and
            not half_power.has_value()) {
          return {2,  19, 3,  15, 18, 6,  7,  11, 22, 10, 23, 7, 11, 22,
                  14, 3,  15, 18, 3,  15, 18, 2,  14, 19, 22, 6, 10, 23};
        }
      }();

  const std::vector<size_t> spec_dest_indices =
      [ell_max, number_of_ell_modes_to_kill, half_power]() {
        // These are numbers that Mark Scheel generated using SpEC's
        // TensorYlmFilter for various special cases.
        if (ell_max == 1 and number_of_ell_modes_to_kill == 0 and
            not half_power.has_value()) {
          return {2,  2,  3,  3,  3,  6,  7,  7,  7,  10, 10, 11, 11, 11,
                  14, 15, 15, 15, 18, 18, 18, 19, 19, 19, 22, 23, 23, 23};
        }
      }();

  blaze::CompressedMatrix<double, blaze::rowMajor> matrix;
  ylm::TensorYlm::FillFilter<TensorStructure>(
      make_not_null(&matrix), ell_max, number_of_ell_modes_to_kill, half_power);

  // loop over spec_matrix_elements and make sure all the
  // nonzero ones agree.
  for (size_t i = 0; i < spec_matrix_elements.size(); ++i) {
    CHECK(matrix(spec_dest_indices[i], spec_src_indices[i]) ==
          approx(spec_matrix_elements[i]));
  }

  // loop over matrix elements and make sure all the nonzero ones
  // agree.
  size_t count = 0;
  for (size_t row = 0; row < matrix.rows(); ++row) {
    for (blaze::CompressedMatrix<double, blaze::rowMajor>::Iterator it =
             matrix.begin(row);
         it != matrix.end(row); ++it, ++count) {
      CHECK(it->value() == approx(spec_matrix_elements[count]));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.TensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  test_tensorylm_filter_vs_spec<typename tnsr::i<DataVector, 3>::structure>(
      std::nullopt);
}
