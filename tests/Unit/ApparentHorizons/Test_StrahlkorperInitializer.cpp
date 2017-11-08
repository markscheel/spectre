// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperInitializer.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperInitializer",
                  "[ApparentHorizons][Unit]") {
  constexpr size_t l_max = 5;
  constexpr double radius = 2.;
  constexpr std::array<double, 3> center{{1., 2., 3.}};

  const StrahlkorperInitializers::Sphere<Frame::Inertial> sphere{l_max, radius,
                                                                 center};

  const auto strahlkorper = sphere.create_strahlkorper();
  const Strahlkorper<Frame::Inertial> test_strahlkorper(l_max, l_max, radius,
                                                        center);

  CHECK(strahlkorper == test_strahlkorper);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperInitializer.Factory",
                  "[ApparentHorizons][Unit]") {
  const auto creator =
      test_factory_creation<StrahlkorperInitializer<Frame::Inertial>>(
          "  Sphere:\n"
          "    Lmax: 5\n"
          "    Radius: 2\n"
          "    Center: [3,4,5]");

  const auto strahlkorper = creator->create_strahlkorper();
  const Strahlkorper<Frame::Inertial> test_strahlkorper(5, 5, 2.0,
                                                        {{3.0, 4.0, 5.0}});
  CHECK(strahlkorper == test_strahlkorper);
}
