// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include <iostream>

#include "ApparentHorizons/ChangeCenterOfStrahlkorper.hpp"
#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace {

void test_strahlkorper_in_different_frame() {
  const size_t l_max = 12;
  const std::array<double, 3> grid_center{{0.1, 0.2, 0.3}};

  std::uniform_real_distribution<double> interval_dis(-1.0, 1.0);
  MAKE_GENERATOR(gen);

  const auto strahlkorper_grid =
      [&gen, &grid_center, &interval_dis ]() noexcept {
    // First make a sphere.
    const double avg_radius = 1.0;
    const double delta_radius = 0.1;
    Strahlkorper<Frame::Grid> sk(l_max, l_max, avg_radius, grid_center);
    // Now adjust the coefficients randomly, but in a manner so that
    // coefficients decay like exp(-l).
    auto coefs = sk.coefficients();
    for (SpherepackIterator it(l_max, l_max); it; ++it) {
      coefs[it()] +=
          interval_dis(gen) * delta_radius * exp(-static_cast<double>(it.l()));
    }
    sk = Strahlkorper<Frame::Grid>(coefs, sk);
    change_expansion_center_of_strahlkorper_to_physical(make_not_null(&sk));
    return sk;
  }
  ();

  // Simple map where the exact solution is nontrivial (from the point
  // of view of the algorithm) and straightforward to compute: an
  // overall scaling by a factor of 2 and an overall translation by
  // (1,-20,-5).
  using affine_map = domain::CoordinateMaps::Affine;
  using affine_map_3d =
      domain::CoordinateMaps::ProductOf3Maps<affine_map, affine_map,
                                             affine_map>;
  const auto scaling_and_translation_map =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, affine_map_3d>(
          affine_map_3d{affine_map{-10.0, 10.0, -19.0, 21.0},
                        affine_map{-10.0, 10.0, -40.0, 0.0},
                        affine_map{-10.0, 10.0, -25.0, 15.0}})
          .get_clone();

  // Now compute the Strahlkorper in the inertial frame.
  Strahlkorper<Frame::Inertial> strahlkorper_inertial(l_max, 1.0, grid_center);
  strahlkorper_in_different_frame(make_not_null(&strahlkorper_inertial),
                                  strahlkorper_grid,
                                  scaling_and_translation_map);

  // The center of strahlkorper_inertial is not something we can
  // easily compute analytically, since it is based on averaging over
  // grid points.  However, we can reset strahlkorper_inertial's
  // center to be its physical center, and that we should be able to
  // compute analytically.
  change_expansion_center_of_strahlkorper_to_physical(
      make_not_null(&strahlkorper_inertial));

  const auto strahlkorper_inertial_physical_center =
      strahlkorper_inertial.physical_center();
  const std::array<double, 3> expected_strahlkorper_inertial_physical_center =
      [&strahlkorper_grid]() noexcept {
    const auto strahlkorper_grid_physical_center =
        strahlkorper_grid.physical_center();
    std::array<double, 3> inertial_center = strahlkorper_grid_physical_center;
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(inertial_center, d) *= 2.0;
    }
    inertial_center[0] += 1.0;
    inertial_center[1] -= 20.0;
    inertial_center[2] -= 5.0;

    return inertial_center;
  }
  ();

  for (size_t i = 0; i < 3; ++i) {
    CHECK(approx(gsl::at(strahlkorper_inertial_physical_center, i)) ==
          gsl::at(expected_strahlkorper_inertial_physical_center, i));
  }

  // The inertial coefficients are twice the grid coefficients because
  // of the expansion of the map.
  const DataVector expected_inertial_coefficients =
      strahlkorper_grid.coefficients() * 2.0;
  // Check inertial-frame radius computed with expected coefficients
  // versus actual coefficients.
  // We check radius instead of coefficients because if you check
  // coefficients you would need to obey an overly-strict error tolerance
  // for the higher-order (i.e. very small) coefficients.
  // Radii depend on l_max, so the epsilon below may need to change if
  // l_max is varied.
  Approx custom_approx_ten = Approx::custom().epsilon(1.e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(
      strahlkorper_inertial.ylm_spherepack().spec_to_phys(
          expected_inertial_coefficients),
      strahlkorper_inertial.ylm_spherepack().spec_to_phys(
          strahlkorper_inertial.coefficients()),
      custom_approx_ten);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper.DifferentFrame",
                  "[ApparentHorizons][Unit]") {
  test_strahlkorper_in_different_frame();
}
}  // namespace
