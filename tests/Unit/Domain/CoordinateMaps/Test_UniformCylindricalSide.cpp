// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

namespace domain {

namespace {

void test_uniform_cylindrical_side(const bool upper_planes_are_equal = false,
                                   const bool lower_planes_are_equal = false) {
  if(upper_planes_are_equal and lower_planes_are_equal) {
    ERROR("Map untested if both upper and lower planes are equal");
  }
  INFO("UniformCylindricalSide");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random center for sphere_two
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Choose a random radius of sphere_two, reasonably large.
  const double radius_two = 6.0 * (unit_dis(gen) + 1.0);
  CAPTURE(radius_two);

  // These angles describe how close the z-planes can be to the
  // centers or edges of the spheres.
  const double min_angle = 0.15;
  const double max_angle = 0.4;

  // Make sure z_plane_plus_two intersects sphere_two on the +z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_plus_two = [&max_angle, &min_angle, &radius_two,
                                   &center_two, &upper_planes_are_equal,
                                   &unit_dis, &gen]() {
    // If the planes are equal, relax the min angle to make sure that sphere_one
    // fits inside sphere_two
    const double new_min_angle =
        upper_planes_are_equal ? min_angle + 0.1 : min_angle;
    return center_two[2] +
           cos((new_min_angle + (max_angle - new_min_angle) * unit_dis(gen)) *
               M_PI) *
               radius_two;
  }();
  CAPTURE(z_plane_plus_two);

  // Make sure z_plane_minus_two intersects sphere_two on the -z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_minus_two = [&max_angle, &min_angle, &radius_two,
                                    &center_two, &lower_planes_are_equal,
                                    &unit_dis, &gen]() {
    // If the planes are equal, relax the min angle to make sure that sphere_one
    // fits inside sphere_two
    const double new_min_angle =
        lower_planes_are_equal ? min_angle + 0.1 : min_angle;
    return center_two[2] -
           cos((new_min_angle + (max_angle - new_min_angle) * unit_dis(gen)) *
               M_PI) *
               radius_two;
  }();
  CAPTURE(z_plane_minus_two);

  // Choose z_plane_frac_plus_one=(z_plane_plus_one-center_one[2])/radius_one
  const double z_plane_frac_plus_one =
      cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI);

  // Choose
  // z_plane_frac_minus_one=(z_plane_minus_one-center_one[2])/radius_one (note
  // that this quantity is < 0).
  const double z_plane_frac_minus_one =
      -cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI);

  // Compute the minimum allowed value of the angle alpha_plus.
  const double theta_max_plus_one = acos(z_plane_frac_plus_one);
  const double theta_max_plus_two =
      acos((z_plane_plus_two - center_two[2]) / radius_two);
  const double min_alpha_plus =
      1.1 * std::max(theta_max_plus_one, theta_max_plus_two);

  // Compute the minimum allowed value of the angle alpha_minus.
  // (these quantities are measured from zero; note the minus signs)
  const double theta_max_minus_one = acos(-z_plane_frac_minus_one);
  const double theta_max_minus_two =
      acos(-(z_plane_minus_two - center_two[2]) / radius_two);
  const double min_alpha_minus =
      1.1 * std::max(theta_max_minus_one, theta_max_minus_two);

  // Choose a random radius of sphere_one, not too small and not larger
  // than sphere_two.
  const double radius_one = [&center_two, &radius_two, &z_plane_frac_plus_one,
                             &z_plane_plus_two, &min_alpha_plus,
                             &theta_max_plus_one, &theta_max_plus_two,
                             &z_plane_frac_minus_one, &z_plane_minus_two,
                             &min_alpha_minus, &theta_max_minus_one,
                             &theta_max_minus_two, &upper_planes_are_equal,
                             &lower_planes_are_equal, &unit_dis, &gen]() {
    const double z_upper_separation = upper_planes_are_equal ? 0.0 : 0.03;
    const double z_lower_separation = lower_planes_are_equal ? 0.0 : 0.03;
    // max_radius_one_to_fit_inside_sphere_two_plus is the largest that
    // radius_one can be and still satisfy both
    // 0.98 radius_two >= radius_one + |C_1-C_2| and
    // z_plane_plus_two >= z_plane_plus_one + z_upper_separation*radius_two
    // when center_one_z is unknown and z_plane_plus_one is unknown
    // (but the quantity z_plane_frac_plus_one is known).
    // This value comes about when center_one and center_two have the
    // same x and y components, and when center_one_z < center_two_z, and
    // when center_one_z takes on its largest possible value consistent with
    // 0.98 radius_two >= radius_one + |C_1-C_2|.
    // The latter condition is C^z_1 >= radius_one + C^z_2 - 0.98 radius_two.
    //
    // Similarly, max_radius_one_to_fit_inside_sphere_two_minus
    // is the largest that radius_one can be and still satisfy both
    // 0.98 radius_two >= radius_one + |C_1-C_2| and
    // z_plane_minus_two <= z_plane_minus_one - z_lower_separation*radius_two
    // when center_one_z is unknown and z_plane_minus_one is unknown
    // (but the quantity z_plane_frac_minus_one is known).
    // This value comes about when center_one and center_two have the
    // same x and y components, and when center_one_z > center_two_z, and
    // when center_one_z takes on its smallest possible value consistent with
    // 0.98 radius_two >= radius_one + |C_1-C_2|.
    // The latter condition is C^z_1 <= -radius_one + C^z_2 + 0.98 radius_two.
    //
    // Here we take the min of both of the above quantities.
    const double max_radius_one_to_fit_inside_sphere_two =
        std::min((z_plane_plus_two - center_two[2] +
                  (0.98 - z_upper_separation) * radius_two) /
                     (z_plane_frac_plus_one + 1.0),
                 (z_plane_minus_two - center_two[2] -
                  (0.98 - z_lower_separation) * radius_two) /
                     (z_plane_frac_minus_one - 1.0));
    // max_radius_one_for_alpha_minus is the largest that radius_one can be
    // and still satisfy alpha_minus > min_alpha_minus.  For
    // tan(min_alpha_minus) > 0, if max_radius_one_to_fit_inside_sphere_two is
    // satisfied, then alpha_minus > min_alpha_minus imposes no additional
    // restriction on radius.
    const double max_radius_one_for_alpha_plus =
        min_alpha_plus > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : std::min(
                  radius_two * sin(theta_max_plus_two) /
                      sin(theta_max_plus_one),
                  (0.98 * radius_two - z_plane_plus_two + center_two[2] -
                   radius_two * sin(theta_max_plus_two) * tan(min_alpha_plus)) /
                      (1.0 - cos(theta_max_plus_one) -
                       sin(theta_max_plus_one) * tan(min_alpha_plus)));
    const double max_radius_one_for_alpha_minus =
        min_alpha_minus > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : std::min(radius_two * sin(theta_max_minus_two) /
                           sin(theta_max_minus_one),
                       (0.98 * radius_two + z_plane_minus_two - center_two[2] -
                        radius_two * sin(theta_max_minus_two) *
                            tan(min_alpha_minus)) /
                           (1.0 - cos(theta_max_minus_one) -
                            sin(theta_max_minus_one) * tan(min_alpha_minus)));
    CHECK(max_radius_one_for_alpha_minus > 0.0);
    CHECK(max_radius_one_for_alpha_plus > 0.0);
    // max_radius_one_to_fit_between_plane_twos is the maximum radius_one
    // that satisfies the two conditions
    // z_plane_plus_two >= z_plane_plus_one + z_upper_separation*radius_two
    // z_plane_minus_two <= z_plane_minus_one - z_lower_separation*radius_two
    //
    // This condition is derived from noting that
    // z_plane_plus_one = center_one[2]+radius_one*z_plane_frac_plus_one
    // and z_plane_minus_one = center_one[2]+radius_one*z_plane_frac_minus_one
    // (recall z_plane_frac_minus_one is negative)
    // and noting that the max value of center_one[2] is >= the min value
    // of center_one[2].
    const double max_radius_one_to_fit_between_plane_twos =
        (z_plane_plus_two - z_plane_minus_two -
         (z_upper_separation + z_lower_separation) * radius_two) /
        (z_plane_frac_plus_one - z_plane_frac_minus_one);
    CHECK(max_radius_one_to_fit_between_plane_twos > 0.0);

    // We add an additional safety factor of 0.99 to
    // max_radius_one_to_fit_inside_sphere_two so that radius_one
    // doesn't get quite that large.
    double max_radius_one = std::min(
        {0.98 * radius_two, 0.99 * max_radius_one_to_fit_inside_sphere_two,
         max_radius_one_to_fit_between_plane_twos,
         max_radius_one_for_alpha_minus, max_radius_one_for_alpha_plus});

    double min_radius_one = 0.08 * radius_two;

    // We also have restrictions that
    // C^z_1 < C^z_2 + r_1 + r_2/5
    // C^z_1 > C^z_2 - r_1 - r_2/5
    // which are guaranteed by the choice of center_one_z in the next
    // lambda.  But the next lambda is bypassed if
    // upper_planes_are_equal or if lower_planes_are_equal.  So we
    // need to enforce these restrictions here by placing a minimum
    // value on r_1.
    // These restrictions are equivalent to
    // r_1 > |C^z_1 - C^z_2| - r_2/5
    //
    // Similarly we have the restriction that
    // r_1 < 0.98 r_2 - |C^z_2 -C^z_1|
    // which is usually enforced elsewhere, but if
    // upper_planes_are_equal or if lower_planes_are_equal, we need to
    // enforce it here because C^z_1 is already determined by r_1.
    if (upper_planes_are_equal) {
      // Here C^z_1 = z_plane_plus_two - z_plane_frac_plus_one r_1
      // so
      // r_1 > z_plane_plus_two - z_plane_frac_plus_one r_1 - C^z_2 - r_2/5
      // and
      // r_1 > -z_plane_plus_two + z_plane_frac_plus_one r_1 + C^z_2 - r_2/5
      min_radius_one =
          std::max({min_radius_one,
                    (z_plane_plus_two - center_two[2] - 0.2 * radius_two) /
                        (1.0 + z_plane_frac_plus_one),
                    (-z_plane_plus_two + center_two[2] - 0.2 * radius_two) /
                        (1.0 - z_plane_frac_plus_one)});
      // So also
      // r_1 < z_plane_plus_two - z_plane_frac_plus_one r_1 - C^z_2 + 0.98 r_2
      // and
      // r_1 < -z_plane_plus_two + z_plane_frac_plus_one r_1 + C^z_2 + 0.98
      // r_2
      max_radius_one =
          std::min({max_radius_one,
                    (z_plane_plus_two - center_two[2] + 0.98 * radius_two) /
                        (1.0 + z_plane_frac_plus_one),
                    (-z_plane_plus_two + center_two[2] + 0.98 * radius_two) /
                        (1.0 - z_plane_frac_plus_one)});
    } else if (lower_planes_are_equal) {
      // Here C^z_1 = z_plane_minus_two - z_plane_frac_minus_one r_1
      // so
      // r_1 > z_plane_minus_two - z_plane_frac_minus_one r_1 - C^z_2 - r_2/5
      // and
      // r_1 > -z_plane_minus_two + z_plane_frac_minus_one r_1 + C^z_2 - r_2/5
      min_radius_one =
          std::max({min_radius_one,
                    (z_plane_minus_two - center_two[2] - 0.2 * radius_two) /
                        (1.0 + z_plane_frac_minus_one),
                    (-z_plane_minus_two + center_two[2] - 0.2 * radius_two) /
                        (1.0 - z_plane_frac_minus_one)});
      // So also
      // r_1 < z_plane_minus_two - z_plane_frac_minus_one r_1 - C^z_2 + 0.98
      // r_2 and r_1 <-z_plane_minus_two + z_plane_frac_minus_one r_1 + C^z_2
      // + 0.98 r_2
      max_radius_one =
          std::min({max_radius_one,
                    (z_plane_minus_two - center_two[2] + 0.98 * radius_two) /
                        (1.0 + z_plane_frac_minus_one),
                    (-z_plane_minus_two + center_two[2] + 0.98 * radius_two) /
                        (1.0 - z_plane_frac_minus_one)});
    }
    CHECK(max_radius_one >= min_radius_one);
    return min_radius_one + unit_dis(gen) * (max_radius_one - min_radius_one);
  }();
  CAPTURE(radius_one);

  // Choose a random z-center of sphere_one.
  const double center_one_z = [&radius_two, &radius_one, &center_two,
                               &z_plane_frac_plus_one, &z_plane_plus_two,
                               &min_alpha_plus, &theta_max_plus_one,
                               &theta_max_plus_two, &z_plane_frac_minus_one,
                               &z_plane_minus_two, &min_alpha_minus,
                               &theta_max_minus_one, &theta_max_minus_two,
                               &upper_planes_are_equal, &lower_planes_are_equal,
                               &unit_dis, &gen]() {
    if (upper_planes_are_equal) {
      return z_plane_plus_two - z_plane_frac_plus_one * radius_one;
    } else if (lower_planes_are_equal) {
      return z_plane_minus_two - z_plane_frac_minus_one * radius_one;
    }
    const double max_center_one_z_from_alpha_plus =
        (tan(min_alpha_plus) <= 0.0 or radius_one * sin(theta_max_plus_one) <=
                                           radius_two * sin(theta_max_plus_two))
            ? std::numeric_limits<double>::max()
            : (radius_two * sin(theta_max_plus_two) -
               radius_one * sin(theta_max_plus_one)) *
                  tan(min_alpha_plus);
    // Note minus sign in min_center_one_z_from_alpha_minus
    const double min_center_one_z_from_alpha_minus =
        (tan(min_alpha_minus) <= 0.0 or
         radius_one * sin(theta_max_minus_one) <=
             radius_two * sin(theta_max_minus_two))
            ? std::numeric_limits<double>::lowest()
            : -(radius_two * sin(theta_max_minus_two) -
                radius_one * sin(theta_max_minus_one)) *
                  tan(min_alpha_minus);
    CHECK(min_center_one_z_from_alpha_minus <=
          max_center_one_z_from_alpha_plus);
    // max_center_one_z comes from the restriction
    // z_plane_plus_two >= z_plane_plus_one + 0.03*radius_two,
    // and the restriction
    // 0.98 r_2 >= r_1 + | C_1 - C_2 |
    // and the restriction
    // C^z_1 < C^z_2 + r_1 + r_2/5
    // which is designed to not allow a tiny sphere 1 at the edge of
    // a large sphere 2.
    const double max_center_one_z =
        std::min({max_center_one_z_from_alpha_plus,
                  z_plane_plus_two - z_plane_frac_plus_one * radius_one -
                      0.03 * radius_two,
                  center_two[2] + radius_one + 0.2 * radius_two,
                  center_two[2] + 0.98 * radius_two - radius_one});
    // min_center_one_z comes from the restriction
    // z_plane_minus_two <= z_plane_minus_one - 0.03*radius_two,
    // and the restriction
    // 0.98 r_2 >= r_1 + |C_1 - C_2 |
    // and the restriction
    // C^z_1 > C^z_2 - r_1 - r_2/5
    // which is designed to not allow a tiny sphere 1 at the edge of
    // a large sphere 2.
    const double min_center_one_z =
        std::max({min_center_one_z_from_alpha_minus,
                  z_plane_minus_two - z_plane_frac_minus_one * radius_one +
                      0.03 * radius_two,
                  center_two[2] - radius_one - 0.2 * radius_two,
                  center_two[2] - 0.98 * radius_two + radius_one});
    CHECK(min_center_one_z <= max_center_one_z);
    return min_center_one_z +
           unit_dis(gen) * (max_center_one_z - min_center_one_z);
  }();
  CAPTURE(center_one_z);

  // Now we can compute z_plane_plus_one and z_plane_minus_one If
  // upper and lower planes are equal, put in exact value so there is
  // no roundoff.
  const double z_plane_plus_one =
      upper_planes_are_equal
          ? z_plane_plus_two
          : center_one_z + radius_one * z_plane_frac_plus_one;
  CAPTURE(z_plane_plus_one);
  const double z_plane_minus_one =
      lower_planes_are_equal
          ? z_plane_minus_two
          : center_one_z + radius_one * z_plane_frac_minus_one;
  CAPTURE(z_plane_minus_one);

  // Only thing remaining are the x and y centers of sphere_one.
  const double horizontal_distance_spheres =
      [&z_plane_plus_one, &z_plane_plus_two, &theta_max_plus_one,
       &theta_max_plus_two, &min_alpha_plus, &z_plane_minus_one,
       &z_plane_minus_two, &theta_max_minus_one, &theta_max_minus_two,
       &min_alpha_minus, &center_one_z, &center_two, &radius_one, &radius_two,
       &unit_dis, &gen]() {
        // Let rho be the horizontal (x-y) distance between the centers of
        // the spheres.

        // maximum rho so that sphere2 is inside of sphere 1, with a
        // safety factor of 0.98
        const double max_rho_sphere =
            sqrt(square(0.98 * radius_two - radius_one) -
                 square(center_one_z - center_two[2]));

        // We don't want a tiny sphere 1 all the way on the edge of sphere 2.
        // So demand that at least some of sphere_one lies along the polar
        // axis of sphere_two.
        const double max_rho_sphere2 = radius_one;

        // Alpha always gets smaller when rho gets larger (for other
        // quantities fixed). So if alpha < min_alpha even when rho=0, then
        // there is no hope.  We always fail.
        const double alpha_plus_if_rho_is_zero =
            atan2(z_plane_plus_two - z_plane_plus_one,
                  radius_one * sin(theta_max_plus_one) -
                      radius_two * sin(theta_max_plus_two));
        CHECK(alpha_plus_if_rho_is_zero >= min_alpha_plus);
        const double alpha_minus_if_rho_is_zero =
            atan2(z_plane_minus_one - z_plane_minus_two,
                  radius_one * sin(theta_max_minus_one) -
                      radius_two * sin(theta_max_minus_two));
        CHECK(alpha_minus_if_rho_is_zero >= min_alpha_minus);

        const double max_rho_alpha_plus_first_term =
            abs(min_alpha_plus - 0.5 * M_PI) < 1.e-4
                ? 0.0
                : (z_plane_plus_two - z_plane_plus_one) / tan(min_alpha_plus);

        const double max_rho_alpha_minus_first_term =
            abs(min_alpha_minus - 0.5 * M_PI) < 1.e-4
                ? 0.0
                : (z_plane_minus_one - z_plane_minus_two) /
                      tan(min_alpha_minus);

        // maximum rho so that the alpha condition is satisfied
        const double max_rho_alpha_plus = max_rho_alpha_plus_first_term -
                                          radius_one * sin(theta_max_plus_one) +
                                          radius_two * sin(theta_max_plus_two);
        const double max_rho_alpha_minus =
            max_rho_alpha_minus_first_term -
            radius_one * sin(theta_max_minus_one) +
            radius_two * sin(theta_max_minus_two);
        const double max_rho =
            std::min({max_rho_sphere, max_rho_sphere2, max_rho_alpha_plus,
                      max_rho_alpha_minus});
        return unit_dis(gen) * max_rho;
      }();

  const double phi = angle_dis(gen);
  const std::array<double, 3> center_one = {
      center_two[0] + horizontal_distance_spheres * cos(phi),
      center_two[1] + horizontal_distance_spheres * sin(phi), center_one_z};
  CAPTURE(center_one);

  const CoordinateMaps::UniformCylindricalSide map(
      center_one, center_two, radius_one, radius_two, z_plane_plus_one,
      z_plane_minus_one, z_plane_plus_two, z_plane_minus_two);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // Point with z > z_plane_plus_two.
  CHECK_FALSE(map.inverse({{0.0, 0.0, z_plane_plus_two + 1.0}}));

  // Point with z < z_plane_minus_two.
  CHECK_FALSE(map.inverse({{0.0, 0.0, z_plane_minus_two - 1.0}}));

  // Point outside sphere_two
  CHECK_FALSE(map.inverse(
      {{center_two[0], center_two[1] + 1.01 * radius_two, center_two[2]}}));

  // Point inside sphere_one (but z_plane_minus_one<z<z_plane_plus_one
  // intersects sphere_one)
  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            0.5 * (z_plane_plus_one + z_plane_minus_one)}}));

  // Point inside the northern cone
  if (z_plane_plus_two != z_plane_plus_one) {
    CHECK_FALSE(map.inverse(
        {{center_two[0],
          center_two[1] + radius_two * sin(theta_max_plus_two) * 0.98,
          z_plane_plus_two - (z_plane_plus_two - center_two[2]) * 1.e-5}}));
    }

    // Point inside the southern cone
    if (z_plane_minus_two != z_plane_minus_one) {
      CHECK_FALSE(map.inverse(
          {{center_two[0],
            center_two[1] + radius_two * sin(theta_max_minus_two) * 0.98,
            z_plane_minus_two + (center_two[2] - z_plane_minus_two) * 1.e-5}}));
    }
  }
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.UniformCylindricalSide",
                  "[Domain][Unit]") {
  test_uniform_cylindrical_side();
  test_uniform_cylindrical_side(true, false);
  test_uniform_cylindrical_side(false, true);
  CHECK(not CoordinateMaps::UniformCylindricalSide{}.is_identity());
}
}  // namespace domain
