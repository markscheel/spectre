// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <fstream>
#include <iostream>
#include <optional>
#include <filesystem>

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
  // There are two "modes" for this test.  If test_all_elements is
  // true, it tests all the matrix elements vs SpEC, reading .txt
  // files that were output by SpEC. This test was done by Mark
  // Scheel, and is not done in CI.
  //
  // Otherwise, it tests a random subset of the matrix elements vs
  // SpEC here (the random numbers being previously determined by
  // SpEC), using inlined values, which is done in CI.
  constexpr bool test_all_elements = true;

  std::vector<double> spec_matrix_elements;
  std::vector<size_t> spec_src_indices;
  std::vector<size_t> spec_dest_indices;

  if constexpr (test_all_elements) {
    // Read from simple file that was output by SpEC.
    // This test was run by Mark Scheel after producing the SpEC
    // files.
    // Because the SpEC files are so large, we don't run this
    // test in CI.
    const std::string spec_symm_string = []() {
      if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                   Symmetry<1>>) {
        return "a";
      } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                          Symmetry<1, 1>>) {
        return "aa";
      } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                          Symmetry<2, 1>>) {
        return "ab";
      } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                          Symmetry<3, 2, 1>>) {
        return "abc";
      } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                          Symmetry<2, 1, 1>>) {
        return "abb";
      }
      return "";
    }();
    // File has a simple binary format:
    //   size (a size_t)
    //   matrix_elements (a vector of doubles of length size)
    //   src_indices (a vector of size_t of length size)
    //   dest_indices (a vector of size_t of length size)
    const std::string filename = "TensorYlmCoefs_" + spec_symm_string + "_" +
                                 std::to_string(half_power.value_or(0)) +
                                 ".txt";
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    size_t num_indices;
    file.read(reinterpret_cast<char*>(&num_indices), sizeof(size_t));
    spec_matrix_elements.resize(num_indices);
    spec_src_indices.resize(num_indices);
    spec_dest_indices.resize(num_indices);
    file.read(reinterpret_cast<char*>(spec_matrix_elements.data()),
              static_cast<std::streamsize>(sizeof(double) * num_indices));
    file.read(reinterpret_cast<char*>(spec_src_indices.data()),
              static_cast<std::streamsize>(sizeof(size_t) * num_indices));
    file.read(reinterpret_cast<char*>(spec_dest_indices.data()),
              static_cast<std::streamsize>(sizeof(size_t) * num_indices));
  } else {
    if constexpr (std::is_same_v<typename TensorStructure::symmetry,
                                 Symmetry<1>>) {
      if (half_power.value_or(0) == 28) {
        spec_src_indices = {
            373, 120, 354, 361, 289, 303, 210, 290, 370, 310, 344, 363, 200,
            192, 293, 119, 66,  73,  49,  228, 445, 463, 434, 393, 47,  370,
            292, 199, 280, 50,  111, 471, 319, 57,  45,  361, 343, 443, 74,
            126, 283, 270, 303, 306, 190, 18,  279, 369, 232, 117, 30,  129,
            396, 371, 110, 283, 129, 435, 100, 190, 217, 158, 443, 149, 50,
            262, 472, 380, 27,  454, 236, 451, 29,  363, 119, 281, 343, 283,
            190, 72,  18,  128, 262, 283, 288, 289, 370, 64,  27,  121, 370,
            49,  19,  270, 30,  50,  129, 470, 468, 182};
        spec_dest_indices = {
            293, 443, 29,  36,  369, 303, 210, 370, 290, 310, 19,  363, 444,
            111, 48,  202, 66,  73,  49,  228, 445, 463, 190, 393, 370, 47,
            49,  120, 37,  48,  190, 471, 319, 57,  45,  36,  20,  199, 74,
            128, 283, 352, 303, 306, 190, 343, 361, 369, 232, 200, 273, 131,
            396, 289, 110, 363, 127, 435, 181, 192, 217, 158, 201, 149, 293,
            344, 472, 380, 27,  129, 236, 128, 27,  363, 202, 283, 18,  40,
            192, 72,  18,  453, 344, 363, 370, 289, 290, 64,  272, 119, 47,
            49,  19,  29,  271, 293, 127, 470, 468, 424};
        spec_matrix_elements = {0.258696961803670078,
                                -0.12474710975541925,
                                4.12380498300465295e-05,
                                0.596254677978356185,
                                0.49786253511962425,
                                -1,
                                -0.736399143678998058,
                                0.288588542569257755,
                                0.288588542569257755,
                                -0.999999999999999556,
                                4.52534306324044533e-12,
                                -0.355616162872347152,
                                -0.332553972411752219,
                                6.49364063228479566e-05,
                                0.060975458654577458,
                                0.0587766567045578323,
                                -0.999999999999999778,
                                -1,
                                -0.672771350773928578,
                                -0.999999999999999778,
                                -0.200107746336269959,
                                -0.999999999999999556,
                                -4.56326525141256172e-05,
                                -1,
                                -0.288588542569257811,
                                -0.288588542569257811,
                                -0.327228649226071311,
                                -0.088164985056836731,
                                -0.0111414147649815665,
                                -0.060975458654577458,
                                -9.31472472856344617e-06,
                                -0.999999999999999889,
                                -0.999999999999999889,
                                -0.999999999999999889,
                                -0.818206305985515914,
                                0.596254677978356185,
                                -1.81013722529617853e-12,
                                -0.32995459399023791,
                                -1,
                                0.131722045502268681,
                                -0.511124579408424373,
                                -4.16567232076563951e-05,
                                -1,
                                0.999999999999999778,
                                -3.60757918496048847e-05,
                                4.43391256639299217e-12,
                                -0.298127338989178092,
                                -0.545515764963789951,
                                -0.999999999999999778,
                                -0.105377312481470586,
                                -6.49364063228479566e-05,
                                -0.060975458654577458,
                                -1,
                                0.144294271284628878,
                                -5.77212560985444827e-05,
                                0.0628825066636218566,
                                -0.1178157791513572,
                                -3.36707402663893738e-05,
                                4.52534306324044613e-12,
                                -9.31472472856344787e-06,
                                -0.999999999999999889,
                                -0.999999999999999889,
                                -0.12474710975541925,
                                -0.999999999999999889,
                                -0.409035811532589,
                                4.52534306324044613e-12,
                                -0.999999999999999445,
                                -0.999999999999999778,
                                -4.81010497656824391e-05,
                                0.038564266144378162,
                                -1,
                                -0.288588542569257755,
                                2.63460200818129502e-05,
                                -0.355616162872347152,
                                0.0587766567045578323,
                                -0.0587766567045578323,
                                8.86782513278598433e-12,
                                -0.488875420591575571,
                                -9.31472472856344787e-06,
                                -1,
                                -5.4304116758885352e-12,
                                0.0890603577582286393,
                                4.52534306324044613e-12,
                                0.0628825066636218566,
                                -0.199145014047849567,
                                -0.672771350773928467,
                                0.288588542569257755,
                                -0.999999999999999778,
                                -1.31730100409064734e-05,
                                0.0587766567045578323,
                                -0.288588542569257811,
                                -0.672771350773928578,
                                -7.24054890118471414e-12,
                                -1.31730100409064734e-05,
                                9.31472472856344617e-06,
                                -0.409035811532589,
                                -0.1178157791513572,
                                -0.999999999999999667,
                                0.999999999999999778,
                                -1.81013722529617853e-12};
      } else {
        spec_src_indices = {
            211, 323, 47,  293, 371, 40,  369, 49,  288, 36,  120, 306, 65,
            210, 382, 46,  128, 118, 210, 49,  200, 119, 289, 280, 443, 154,
            198, 131, 137, 70,  371, 73,  374, 280, 469, 156, 212, 291, 378,
            208, 454, 443, 477, 454, 37,  128, 130, 281, 463, 290, 280, 126,
            360, 38,  121, 48,  207, 398, 290, 451, 363, 129, 290, 72,  362,
            118, 281, 140, 199, 485, 378, 323, 293, 40,  207, 312, 202, 49,
            364, 443, 208, 200, 135, 46,  198, 373, 279, 364, 441, 210, 234,
            45,  130, 370, 121, 64,  200, 119, 288, 363};
        spec_dest_indices = {
            130, 323, 292, 373, 48,  281, 289, 292, 290, 361, 201, 306, 65,
            208, 382, 289, 451, 118, 208, 372, 200, 121, 291, 39,  118, 154,
            119, 454, 137, 70,  289, 73,  374, 280, 469, 156, 454, 50,  378,
            452, 212, 199, 477, 131, 39,  451, 130, 36,  463, 45,  360, 128,
            37,  363, 202, 293, 207, 398, 290, 128, 281, 127, 45,  72,  280,
            201, 36,  140, 443, 485, 378, 323, 48,  363, 207, 312, 121, 372,
            282, 199, 208, 119, 135, 369, 198, 291, 279, 39,  199, 452, 234,
            290, 128, 370, 444, 64,  202, 119, 288, 40};
        spec_matrix_elements = {0.327272727272727215,
                                -1,
                                0.0944754985946660436,
                                0.258731808559231058,
                                -0.311753239990586239,
                                0.0587944735792131287,
                                0.248964798865984605,
                                -0.327272727272727215,
                                -0.131739788601722196,
                                0.29814239699997197,
                                0.366666666666666585,
                                0.999999999999999889,
                                -0.999999999999999667,
                                0.117831649061961044,
                                -0.999999999999999778,
                                -0.218181818181818088,
                                -0.288627415752500727,
                                -0.233333333333333254,
                                0.117831649061961044,
                                -0.308555686335947987,
                                -0.377777777777777768,
                                0.0587944735792131287,
                                0.117831649061961086,
                                0.0881917103688196757,
                                0.329983164553722008,
                                -0.999999999999999778,
                                -0.10540925533894599,
                                -0.258731808559231058,
                                -0.999999999999999889,
                                -0.999999999999999667,
                                0.144313707876250391,
                                -1,
                                -0.999999999999999778,
                                -0.455555555555555436,
                                -1.00000000000000022,
                                -1,
                                -0.258731808559231058,
                                -0.0609836721136306215,
                                -0.999999999999999556,
                                -0.144313707876250419,
                                -0.258731808559231058,
                                -0.329983164553722008,
                                1,
                                -0.258731808559231058,
                                0.0881917103688196757,
                                -0.288627415752500727,
                                -0.672727272727272729,
                                -0.210818510677891952,
                                -0.999999999999999778,
                                0.263479577203444448,
                                0.496903994999953136,
                                0.131739788601722196,
                                -0.248451997499976596,
                                0.332591767713239228,
                                0.488888888888888873,
                                0.0609836721136306215,
                                -0.818181818181818232,
                                -0.999999999999999667,
                                -0.781818181818181746,
                                -0.288627415752500782,
                                0.332591767713239228,
                                -0.117831649061961044,
                                0.263479577203444448,
                                -1,
                                0.329983164553722008,
                                0.0881917103688196757,
                                -0.210818510677891952,
                                -0.999999999999999889,
                                -0.329983164553722008,
                                -0.999999999999999889,
                                -0.999999999999999556,
                                -1,
                                0.0609836721136306215,
                                -0.062853936105470895,
                                -0.818181818181818232,
                                -0.999999999999999889,
                                0.488888888888888873,
                                -0.308555686335947987,
                                0.282842712474618951,
                                -0.329983164553722008,
                                -0.945454545454545103,
                                0.244444444444444464,
                                0.999999999999999778,
                                -0.497929597731969209,
                                -0.33333333333333337,
                                0.0385694607919934707,
                                0.33333333333333337,
                                0.282842712474618951,
                                0.248451997499976596,
                                -0.311753239990586239,
                                -1,
                                0.131739788601722196,
                                -0.0944754985946660436,
                                -0.563636363636363713,
                                -0.062853936105470895,
                                -0.999999999999999667,
                                -0.0587944735792131287,
                                -0.377777777777777768,
                                0.818181818181818232,
                                -0.062853936105470895};
      }
    // } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
    //                                     Symmetry<1, 1>>) {
    // } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
    //                                     Symmetry<2, 1>>) {
    // } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
    //                                     Symmetry<2, 1, 1>>) {
    // } else if constexpr (std::is_same_v<typename TensorStructure::symmetry,
    //                                     Symmetry<3, 2, 1>>) {
    }
  }

  blaze::CompressedMatrix<double, blaze::rowMajor> matrix;
  ylm::TensorYlm::FillFilter<TensorStructure>(
      make_not_null(&matrix), ell_max, number_of_ell_modes_to_kill, half_power);

  // Loop over spec_matrix_elements and make sure all the cases agree.
  // Note that spec_matrix_elements might be a subset of nonzero
  // elements of matrix.
  for (size_t i = 0; i < spec_matrix_elements.size(); ++i) {
    CHECK(matrix(spec_dest_indices[i], spec_src_indices[i]) ==
          approx(spec_matrix_elements[i]));
  }

  // Loop over matrix elements and make sure all the nonzero ones
  // agree with SpEC.  This is done only if test_all_elements.
  if constexpr (test_all_elements) {
    size_t count = 0;
    for (size_t row = 0; row < matrix.rows(); ++row) {
      for (blaze::CompressedMatrix<double, blaze::rowMajor>::Iterator it =
               matrix.begin(row);
           it != matrix.end(row); ++it, ++count) {
        CHECK(it->value() == approx(spec_matrix_elements[count]));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.TensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  const size_t ell_max = 8;
  const size_t num_to_kill = 4;

  for (auto half_power : {std::optional<size_t>(), std::optional<size_t>(28)}) {
    test_tensorylm_filter_vs_spec<typename tnsr::i<DataVector, 3>::structure>(
        ell_max, num_to_kill, half_power);
    // test_tensorylm_filter_vs_spec<typename tnsr::i<DataVector, 3>::structure>(
    //     ell_max, num_to_kill, half_power);
    // test_tensorylm_filter_vs_spec<typename tnsr::ii<DataVector, 3>::structure>(
    //     ell_max, num_to_kill, half_power);
    // test_tensorylm_filter_vs_spec<typename tnsr::ij<DataVector, 3>::structure>(
    //     ell_max, num_to_kill, half_power);
    // test_tensorylm_filter_vs_spec<typename tnsr::ijj<DataVector, 3>::structure>(
    //     ell_max, num_to_kill, half_power);
  }
}
