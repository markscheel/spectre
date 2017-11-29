// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperDataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "tests/Unit/ApparentHorizons/YlmTestFunctions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_invert_spec_phys_transform() {
  const double avg_radius = 1.0;
  const double delta_radius = 0.1;
  const size_t l_grid = 33;
  const auto l_grid_high_res = static_cast<size_t>(l_grid * 1.5);
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};

  // Create radius as a function of angle
  DataVector radius(YlmSpherepack::physical_size(l_grid, l_grid), avg_radius);
  {
    std::uniform_real_distribution<double> ran(0.0, 1.0);
    std::mt19937 gen;
    for (auto& r : radius) {
      r += delta_radius * ran(gen);
    }
  }
  CAPTURE_PRECISE(radius);

  // Initialize a strahlkorper of l_max=l_grid
  const Strahlkorper<Frame::Inertial> sk(radius, l_grid, l_grid, center);

  // Put that Strahlkorper onto a larger grid
  const Strahlkorper<Frame::Inertial> sk_high_res(l_grid_high_res,
                                                  l_grid_high_res, sk);

  // Compare coefficients
  SpherepackIterator iter(sk.l_max(), sk.m_max());
  SpherepackIterator iter_high_res(sk_high_res.l_max(), sk_high_res.m_max());
  const auto& init_coefs = sk.coefficients();
  const auto& final_coefs = sk_high_res.coefficients();

  for (size_t l = 0; l <= sk.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(init_coefs[iter.set(l, m)()] ==
            approx(final_coefs[iter_high_res.set(l, m)()]));
    }
  }

  for (size_t l = sk.ylm_spherepack().l_max() + 1;
       l <= sk_high_res.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(final_coefs[iter_high_res.set(l, m)()] == approx(0));
    }
  }
}

void test_average_radius() {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  Strahlkorper<Frame::Inertial> s(4, 4, r, center);
  CHECK(s.average_radius() == approx(r));
}

void test_copy_and_move() {
  Strahlkorper<Frame::Inertial> s(4, 4, 3.0, {{0.1, 0.2, 0.3}});

  test_copy_semantics(s);
  auto s_copy = s;
  test_move_semantics(std::move(s), s_copy);
}

void test_physical_center() {
  const std::array<double, 3> physical_center = {{1.5, 0.5, 1.0}};
  const std::array<double, 3> expansion_center = {{0, 0, 0}};
  const double radius = 5.0;
  const int l_max = 9;

  Strahlkorper<Frame::Inertial> sk(l_max, l_max, radius, expansion_center);
  DataVector r(sk.ylm_spherepack().physical_size(), 0.);

  for (size_t s = 0; s < r.size(); ++s) {
    const double theta = sk.ylm_spherepack().theta_phi_points()[0][s];
    const double phi = sk.ylm_spherepack().theta_phi_points()[1][s];
    // Compute the distance (radius as a function of theta,phi) from
    // the expansion_center to a spherical surface of radius `radius`
    // centered at physical_center.
    const double a = 1.0;
    const double b = -2 * cos(phi) * sin(theta) * physical_center[0] -
                     2 * sin(phi) * sin(theta) * physical_center[1] -
                     2 * cos(theta) * physical_center[2];
    const double c = physical_center[0] * physical_center[0] +
                     physical_center[1] * physical_center[1] +
                     physical_center[2] * physical_center[2] - radius * radius;
    auto roots = real_roots(a, b, c);
    r[s] = std::max(roots[0], roots[1]);
  }
  // Construct a new Strahlkorper sk_test with the radius computed
  // above, centered at expansion_center, so that
  // sk_test.physical_center() should recover the physical center of
  // this surface.
  Strahlkorper<Frame::Inertial> sk_test(r, l_max, l_max, expansion_center);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(approx(gsl::at(physical_center, i)) ==
          gsl::at(sk_test.physical_center(), i));
  }
}

void test_point_is_contained() {
  // Construct a spherical Strahlkorper
  const double radius = 2.;
  const std::array<double, 3> center = {{-1.2, 3., 4.}};
  const Strahlkorper<Frame::Inertial> sphere(3, 2, radius, center);

  // Check whether two known points are contained.
  const std::array<double, 3> point_inside = {{-1.2, 1.01, 4.}};
  const std::array<double, 3> point_outside = {{-1.2, 3., 6.01}};
  CHECK(sphere.point_is_contained(point_inside));
  CHECK(not sphere.point_is_contained(point_outside));
}

void test_constructor_with_different_coefs() {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  Strahlkorper<Frame::Inertial> strahlkorper(4, 4, r, center);

  // Modify the 0,0 coefficient to add a constant to the radius.
  const double add_to_r = 1.34;
  auto coefs = strahlkorper.coefficients();
  coefs[0] += sqrt(8.0) * add_to_r;

  Strahlkorper<Frame::Inertial> strahlkorper_test1(4, 4, r + add_to_r, center);
  Strahlkorper<Frame::Inertial> strahlkorper_test2(strahlkorper, coefs);
  CHECK(strahlkorper_test1.ylm_spherepack().spectral_size() ==
        strahlkorper_test2.ylm_spherepack().spectral_size());
  for (size_t s = 0; s < strahlkorper_test1.ylm_spherepack().spectral_size();
       ++s) {
    CHECK(strahlkorper_test1.coefficients()[s] ==
          approx(strahlkorper_test2.coefficients()[s]));
  }
}

void test_radius_and_derivs() {
  {
    // Create spherical Strahlkorper
    const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
    const double r = 3.0;
    Strahlkorper<Frame::Inertial> s(4, 4, r, center);
    CHECK(s.average_radius() == approx(r));
  }

  // Create a strahlkorper with a Im(Y11) dependence.
  const size_t l_max = 4, m_max = 4;
  const double radius = 2.0;
  const double y11_amplitude = 1.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};

  Strahlkorper<Frame::Inertial> strahlkorper_sphere(l_max, m_max, radius,
                                                    center);
  auto coefs = strahlkorper_sphere.coefficients();
  SpherepackIterator it(l_max, m_max);
  // Conversion between SPHEREPACK b_lm and real valued harmonic coefficients:
  // b_lm = (-1)^{m+1} sqrt(1/2pi) d_lm
  coefs[it.set(1, -1)()] = y11_amplitude * sqrt(0.5 / M_PI);
  Strahlkorper<Frame::Inertial> strahlkorper(strahlkorper_sphere, coefs);

  // Now construct a Y00 + Im(Y11) surface by hand.
  const auto& theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto& theta_points = strahlkorper.ylm_spherepack().theta_points();
  const auto& phi_points = strahlkorper.ylm_spherepack().phi_points();
  const auto n_pts = theta_phi[0].size();
  YlmTestFunctions::Y11 func;
  DataVector test_radius(n_pts);
  func.func(&test_radius, 1, 0, theta_points, phi_points);
  for (size_t s = 0; s < n_pts; ++s) {
    test_radius[s] *= y11_amplitude;
    test_radius[s] += radius;
  }

  // Create DataBox
  auto box =
      db::create<db::AddTags<StrahlkorperDB::TagList<Frame::Inertial>::Tags>,
                 db::AddComputeItemsTags<StrahlkorperDB::TagList<
                     Frame::Inertial>::ComputeItemsTags>>(strahlkorper);

  // Test radius
  const auto& strahlkorper_radius =
      db::get<StrahlkorperDB::Radius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_radius, test_radius);

  // Test derivative of radius
  StrahlkorperDB::types<Frame::Inertial>::OneForm test_dx_radius(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    // Analytic solution I computed in Mathematica
    const double theta = theta_phi[0][s];
    const double phi = theta_phi[1][s];
    const double r = test_radius[s];
    const double sin_phi = sin(phi);
    const double cos_phi = cos(phi);
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;

    test_dx_radius.get(0)[s] = -((cos_phi * sin_phi) / r) +
                               (cos_phi * square(cos_theta) * sin_phi) / r;
    test_dx_radius.get(1)[s] =
        square(cos_phi) / r + (square(cos_theta) * square(sin_phi)) / r;
    test_dx_radius.get(2)[s] = -((cos_theta * sin_phi * sin_theta) / r);
    for (auto& a : test_dx_radius) {
      a[s] *= amp;
    }
  }
  const auto& strahlkorper_dx_radius =
      db::get<StrahlkorperDB::DxRadius<Frame::Inertial>>(box);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(strahlkorper_dx_radius.get(i), test_dx_radius.get(i));
  }

  // Test second derivatives.
  StrahlkorperDB::types<Frame::Inertial>::SecondDeriv test_d2x_radius(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    // Messy analytic solution I computed in Mathematica
    const double theta = theta_phi[0][s];
    const double phi = theta_phi[1][s];
    const double r = test_radius[s];
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double sin_phi = sin(phi);
    const double cos_phi = cos(phi);
    const double cos_2_theta = cos(2 * theta);
    const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;
    test_d2x_radius.get(0, 0)[s] =
        (9 * sin(3 * phi) * sin_theta -
         sin_phi * (7 * sin_theta + 12 * square(cos_phi) * sin(3 * theta))) /
        (16. * square(r));
    test_d2x_radius.get(0, 1)[s] =
        -((cos_phi *
           (square(cos_phi) + ((-1 + 3 * cos_2_theta) * square(sin_phi)) / 2.) *
           sin_theta) /
          square(r));
    test_d2x_radius.get(0, 2)[s] =
        (3 * cos_phi * cos_theta * sin_phi * square(sin_theta)) / square(r);
    test_d2x_radius.get(1, 1)[s] =
        (-3 * (7 * sin_phi * sin_theta + 3 * sin(3 * phi) * sin_theta +
               4 * pow<3>(sin_phi) * sin(3 * theta))) /
        (16. * square(r));
    test_d2x_radius.get(1, 2)[s] =
        -((cos_theta * (square(cos_phi) +
                        ((-1 + 3 * cos_2_theta) * square(sin_phi)) / 2.)) /
          square(r));
    test_d2x_radius.get(2, 2)[s] =
        ((1 + 3 * cos_2_theta) * sin_phi * sin_theta) / (2. * square(r));
    for (auto& a : test_d2x_radius) {
      a[s] *= amp;
    }
  }
  const auto& strahlkorper_d2x_radius =
      db::get<StrahlkorperDB::D2xRadius<Frame::Inertial>>(box);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(test_d2x_radius.get(i, j),
                            strahlkorper_d2x_radius.get(i, j));
    }
  }

  // Test nabla squared
  DataVector test_nabla_squared(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    func.scalar_laplacian(&test_nabla_squared, 1, s, {theta_phi[0][s]},
                          {theta_phi[1][s]});
    test_nabla_squared[s] *= y11_amplitude;
  }
  const auto& strahlkorper_nabla_squared =
      db::get<StrahlkorperDB::NablaSquaredRadius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_nabla_squared, test_nabla_squared);
}

void test_normals() {
  // Create a strahlkorper with a Im(Y11) dependence.
  const size_t l_max = 4, m_max = 4;
  const double radius = 2.0;
  const double y11_amplitude = 1.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};

  Strahlkorper<Frame::Inertial> strahlkorper_sphere(l_max, m_max, radius,
                                                    center);
  auto coefs = strahlkorper_sphere.coefficients();
  SpherepackIterator it(l_max, m_max);
  // Conversion between SPHEREPACK b_lm and real valued harmonic coefficients:
  // b_lm = (-1)^{m+1} sqrt(1/2pi) d_lm
  coefs[it.set(1, -1)()] = y11_amplitude * sqrt(0.5 / M_PI);
  Strahlkorper<Frame::Inertial> strahlkorper(strahlkorper_sphere, coefs);

  const auto& theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto n_pts = theta_phi[0].size();

  // Test surface_tangents

  auto test_surface_tangents =
      make_array<2>(StrahlkorperDB::types<Frame::Inertial>::Vector(n_pts));
  const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataVector cos_phi = cos(phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta = sin(theta);

  test_surface_tangents[0].get(0) =
      cos_phi * cos_theta * (radius + 2. * amp * sin_phi * sin_theta);
  test_surface_tangents[0].get(1) =
      cos_theta * sin_phi * (radius + 2. * amp * sin_phi * sin_theta);
  test_surface_tangents[0].get(2) =
      amp * square(cos_theta) * sin_phi -
      sin_theta * (radius + amp * sin_phi * sin_theta);
  test_surface_tangents[1].get(0) =
      -radius * sin_phi + amp * sin_theta * (2. * square(cos_phi) - 1);
  test_surface_tangents[1].get(1) =
      cos_phi * (radius + 2. * amp * sin_phi * sin_theta);
  test_surface_tangents[1].get(2) = amp * cos_phi * cos_theta;

  // Create DataBox
  auto box =
      db::create<db::AddTags<StrahlkorperDB::TagList<Frame::Inertial>::Tags>,
                 db::AddComputeItemsTags<StrahlkorperDB::TagList<
                     Frame::Inertial>::ComputeItemsTags>>(strahlkorper);

  const auto& surface_tangents =
      db::get<StrahlkorperDB::Tangents<Frame::Inertial>>(box);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(gsl::at(surface_tangents, i).get(j),
                            gsl::at(test_surface_tangents, i).get(j));
    }
  }

  // Test surface_cartesian_coordinates
  StrahlkorperDB::types<Frame::Inertial>::Vector test_cart_coords(n_pts);

  {
    const auto temp = radius + amp * sin_phi * sin_theta;
    test_cart_coords.get(0) = cos_phi * sin_theta * temp + center[0];
    test_cart_coords.get(1) = sin_phi * sin_theta * temp + center[1];
    test_cart_coords.get(2) = cos_theta * temp + center[2];
  }
  const auto& cart_coords =
      db::get<StrahlkorperDB::CartesianCoords<Frame::Inertial>>(box);
  for (size_t j = 0; j < 3; ++j) {
    CHECK_ITERABLE_APPROX(test_cart_coords.get(j), cart_coords.get(j));
  }

  // Test surface_normal_one_form
  StrahlkorperDB::types<Frame::Inertial>::OneForm test_normal_one_form(n_pts);
  {
    const auto& r = db::get<StrahlkorperDB::Radius<Frame::Inertial>>(box);
    const DataVector temp = r + amp * sin_phi * sin_theta;
    const DataVector one_over_r = 1.0 / r;
    test_normal_one_form.get(0) = cos_phi * sin_theta * temp * one_over_r;
    test_normal_one_form.get(1) =
        (sin_phi * sin_theta * temp - amp) * one_over_r;
    test_normal_one_form.get(2) = cos_theta * temp * one_over_r;
  }
  const auto& normal_one_form =
      db::get<StrahlkorperDB::NormalOneForm<Frame::Inertial>>(box);
  for (size_t j = 0; j < 3; ++j) {
    CHECK_ITERABLE_APPROX(test_normal_one_form.get(j), normal_one_form.get(j));
  }

  // Test surface_normal_magnitude.
  tnsr::II<DataVector, 3, Frame::Inertial> invg(n_pts);
  invg.get(0, 0) = 1.0;
  invg.get(1, 0) = 0.1;
  invg.get(2, 0) = 0.2;
  invg.get(1, 1) = 2.0;
  invg.get(1, 2) = 0.3;
  invg.get(2, 2) = 3.0;

  DataVector test_normal_mag(n_pts);
  {
    const auto& r = db::get<StrahlkorperDB::Radius<Frame::Inertial>>(box);

    // Nasty expression I computed in mathematica.
    const DataVector normsquared =
        (-0.30000000000000004 * cos_theta * (r + amp * sin_phi * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (amp * (-0.6666666666666667 +
                                0.6666666666666667 * square(cos_theta)) *
                             sin_phi -
                         0.6666666666666667 * r * sin_theta) +
              cos_theta * (-10. * r - 10. * amp * sin_phi * sin_theta)) +
         0.1 * cos_phi * (amp * (-1. + 1. * square(cos_theta)) * sin_phi -
                          1. * r * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (amp * (-10. + 10. * square(cos_theta)) * sin_phi -
                         10. * r * sin_theta) +
              cos_theta * (-2. * r - 2. * amp * sin_phi * sin_theta)) +
         2. * (amp * square(cos_phi) +
               sin_phi *
                   (amp * square(cos_theta) * sin_phi - 1. * r * sin_theta)) *
             (amp * square(cos_phi) +
              amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (amp * (-0.05 + 0.05 * square(cos_theta)) * sin_phi -
                         0.05 * r * sin_theta) +
              cos_theta * (-0.15 * r - 0.15 * amp * sin_phi * sin_theta))) /
        square(r);
    test_normal_mag = sqrt(normsquared);
  }
  const auto& normal_mag = magnitude(normal_one_form, invg);
  CHECK_ITERABLE_APPROX(test_normal_mag, normal_mag);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper",
                  "[ApparentHorizons][Unit]") {
  test_invert_spec_phys_transform();
  test_copy_and_move();
  test_average_radius();
  test_physical_center();
  test_point_is_contained();
  test_constructor_with_different_coefs();
  test_radius_and_derivs();
  test_normals();
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper.Serialization",
                  "[ApparentHorizons][Unit]") {
  Strahlkorper<Frame::Inertial> s(4, 4, 2.0, {{1.0, 2.0, 3.0}});
  test_serialization(s);
}
