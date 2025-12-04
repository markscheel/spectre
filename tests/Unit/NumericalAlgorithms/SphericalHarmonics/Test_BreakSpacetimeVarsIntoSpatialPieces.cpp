// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/BreakSpacetimeVarsIntoSpatialPieces.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using gh_spacetime_vars_list =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
               gh::Tags::Pi<DataVector, 3, Frame::Inertial>,
               gh::Tags::Phi<DataVector, 3, Frame::Inertial>>;
using gh_spatial_vars_list =
    tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempi<0, 3>, ::Tags::Tempii<0, 3>,
               ::Tags::TempScalar<1>, ::Tags::Tempi<1, 3>, ::Tags::Tempii<1, 3>,
               ::Tags::Tempi<2, 3>, ::Tags::Tempij<2, 3>,
               ::Tags::Tempijj<2, 3>>;

void test_break_spacetime_vars_into_spatial_pieces() {
  constexpr size_t mesh_size = 10;

  Variables<gh_spacetime_vars_list> gh_spacetime_vars(mesh_size);

  // Fill with random numbers
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < gh_spacetime_vars.size(); ++i) {
    gh_spacetime_vars.data()[i] = dist(generator);
  }

  // Break into spatial pieces, then do the inverse, and make
  // sure we get the original back.
  Variables<gh_spatial_vars_list> gh_spatial_vars(mesh_size);
  break_spacetime_vars_into_spatial_pieces(make_not_null(&gh_spatial_vars),
                                           gh_spacetime_vars);
  Variables<gh_spacetime_vars_list> test_gh_spacetime_vars(mesh_size);
  assemble_spacetime_vars_from_spatial_pieces(
      make_not_null(&test_gh_spacetime_vars), gh_spatial_vars);

  // This should be equal to the last bit, since we aren't doing
  // any operations that should incur roundoff error.
  for (size_t i = 0; i < gh_spacetime_vars.size(); ++i) {
    CHECK(gh_spacetime_vars.data()[i] == test_gh_spacetime_vars.data()[i]);
  }
}

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.BreakSpacetimeVarsIntoSpatialPieces",
                  "[NumericalAlgorithms][Unit]") {
  test_break_spacetime_vars_into_spatial_pieces();
}

}  // namespace
