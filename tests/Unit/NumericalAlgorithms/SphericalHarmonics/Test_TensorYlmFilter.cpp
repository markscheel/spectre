// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename TensorStructure>
void test_tensorylm_filter_vs_spec(const std::optional<size_t> half_power) {
  const size_t ell_max = 1;
  const size_t number_of_ell_modes_to_kill = 0;

  const std::vector<double> spec_matrix_elements{
      -0.5,      0.707107,  -0.666667, -0.333333, -0.235702, 0.5,
      -0.5,      0.5,       0.235702,  -0.5,      -0.707107, 0.5,
      -0.5,      0.235702,  0.5,       -0.333333, -0.666667, 0.235702,
      -0.471405, 0.471405,  -0.666667, 0.353553,  -0.353553, -0.5,
      0.666667,  -0.353553, -0.353553, -0.5};

  const std::vector<size_t> spec_src_indices{
      2,  19, 3,  15, 18, 6,  7,  11, 22, 10, 23, 7, 11, 22,
      14, 3,  15, 18, 3,  15, 18, 2,  14, 19, 22, 6, 10, 23};

  const std::vector<size_t> spec_dest_indices{
      2,  2,  3,  3,  3,  6,  7,  7,  7,  10, 10, 11, 11, 11,
      14, 15, 15, 15, 18, 18, 18, 19, 19, 19, 22, 23, 23, 23};

  // const std::vector<double> spec_matrix_elements{-0.5,         -0.666667,
  //                                                -0.9,         -0.244949,
  //                                                -0.7,         -0.122474,
  //                                                -0.7,         0.5,
  //                                                -0.5,         0.9,
  //                                                -1,           0.122474,
  //                                                -0.7,         -0.333333,
  //                                                0.244949,     -0.3,
  //                                                0.122474,     -0.3,
  //                                                0.5,          -1.38778e-17,
  //                                                1.11022e-16,  0.122474,
  //                                                0.3,          0.707107,
  //                                                -0.235702,    0.244949,
  //                                                -0.244949,    -1.11022e-16,
  //                                                -0.3,         0.235702,
  //                                                1.38778e-17,  0.244949,
  //                                                -1.11022e-16, -0.3,
  //                                                0.5,          0.244949,
  //                                                1.11022e-16,  0.122474,
  //                                                0.3,          -0.333333,
  //                                                -1.38778e-17, -0.3,
  //                                                0.122474,     -0.3,
  //                                                -0.5,         -0.5,
  //                                                -0.9,         0.244949,
  //                                                -1,           0.122474,
  //                                                -0.7,         0.5,
  //                                                -0.666667,    0.9,
  //                                                -0.7,         -0.122474,
  //                                                -0.7,         -0.707107,
  //                                                0.235702,     -0.244949,
  //                                                0.244949,     1.11022e-16,
  //                                                -0.3,         0.235702,
  //                                                1.38778e-17,  0.244949,
  //                                                -1.11022e-16, 0.3,
  //                                                -0.471405,    0.353553,
  //                                                -0.489898,    0.122474,
  //                                                -0.3,         -1.11022e-16,
  //                                                -0.353553,    -0.122474,
  //                                                -0.3,         -1.11022e-16,
  //                                                0.471405,     -0.353553,
  //                                                0.489898,     -0.122474,
  //                                                0.3,          -1.11022e-16,
  //                                                -0.353553,    -0.122474,
  //                                                -0.3,         1.11022e-16,
  //                                                -0.666667,    -0.5,
  //                                                -0.6,         -0.7,
  //                                                -1,           0.666667,
  //                                                -0.5,         0.6,
  //                                                -0.7,         -1};
  // const std::vector<size_t> spec_src_indices{
  //     1,  2,  3,  5,  4,  3,  5,  7,  8,  9,  10, 9,  11, 20, 23, 22, 21,
  //     23, 14, 17, 16, 15, 17, 26, 25, 28, 27, 29, 28, 31, 34, 33, 35, 34,
  //     8,  11, 10, 9,  11, 2,  5,  4,  3,  5,  13, 14, 15, 17, 16, 15, 17,
  //     19, 20, 21, 22, 21, 23, 32, 31, 34, 33, 35, 34, 25, 28, 27, 29, 28,
  //     2,  1,  4,  3,  5,  4,  7,  9,  11, 10, 20, 19, 22, 21, 23, 22, 13,
  //     15, 17, 16, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35};
  // const std::vector<size_t> spec_dest_indices{
  //     1,  2,  3,  3,  4,  5,  5,  7,  8,  9,  10, 11, 11, 2,  3,  4,  5,
  //     5,  8,  9,  10, 11, 11, 1,  2,  3,  4,  4,  5,  8,  9,  10, 10, 11,
  //     14, 15, 16, 17, 17, 20, 21, 22, 23, 23, 13, 14, 15, 15, 16, 17, 17,
  //     19, 20, 21, 22, 23, 23, 13, 14, 15, 16, 16, 17, 20, 21, 22, 22, 23,
  //     25, 26, 27, 28, 28, 29, 32, 34, 34, 35, 25, 26, 27, 28, 28, 29, 32,
  //     34, 34, 35, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35};

  blaze::CompressedMatrix<double, blaze::rowMajor> matrix;
  ylm::TensorYlm::FillFilter<TensorStructure>(
      make_not_null(&matrix), ell_max, number_of_ell_modes_to_kill, half_power);

  // // loop over spec_matrix_elements and make sure all the nonzero ones agree.
  // for (size_t i = 0; i < spec_matrix_elements.size(); ++i) {
  //   CHECK(matrix(spec_dest_indices[i], spec_src_indices[i]) ==
  //         approx(spec_matrix_elements[i]));
  // }

  // loop over matrix elements and make sure all the nonzero ones agree.
  size_t count = 0;
  for (size_t row = 0; row < matrix.rows(); ++row) {
    for (blaze::CompressedMatrix<double, blaze::rowMajor>::Iterator it =
             matrix.begin(row);
         it != matrix.end(row); ++it, ++count) {
      const auto column = it->index();
      CHECK(it->value() == approx(spec_matrix_elements[count]));
      std::cout << it->value() << " , indx = " << row << ", " << column
                << std::endl;
    }
  }
  std::cout << "Count = " << count << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.TensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  test_tensorylm_filter_vs_spec<typename tnsr::i<DataVector, 3>::structure>(
      std::nullopt);
}
