// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperDataBox.hpp"

namespace StrahlkorperDB {

template <typename Frame>
typename types<Frame>::ThetaPhi ThetaPhi<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  typename types<Frame>::ThetaPhi theta_phi;
  auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
  theta_phi.get(0) = std::move(temp[0]);
  theta_phi.get(1) = std::move(temp[1]);
  return theta_phi;
}

template <typename Frame>
typename types<Frame>::OneForm Rhat<Frame>::compute(
    const typename types<Frame>::ThetaPhi& theta_phi) noexcept {
  typename types<Frame>::OneForm r_hat;

  const auto& theta = theta_phi.get(0);
  const auto& phi = theta_phi.get(1);

  const DataVector sin_theta = sin(theta);
  r_hat.get(0) = sin_theta * cos(phi);
  r_hat.get(1) = sin_theta * sin(phi);
  r_hat.get(2) = cos(theta);
  return r_hat;
}

template <typename Frame>
typename types<Frame>::Jacobian Jacobian<Frame>::compute(
    const typename types<Frame>::ThetaPhi& theta_phi) noexcept {
  const auto& theta = theta_phi.get(0);
  const auto& phi = theta_phi.get(1);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);

  typename types<Frame>::Jacobian jac;
  jac.get(0, 0) = cos_theta * cos_phi;          // 1/R dx/dth
  jac.get(1, 0) = cos_theta * sin_phi;          // 1/R dy/dth
  jac.get(2, 0) = -sin_theta;                   // 1/R dz/dth
  jac.get(0, 1) = -sin_phi;                     // 1/(R sin(th)) dx/dph
  jac.get(1, 1) = cos_phi;                      // 1/(R sin(th)) dy/dph
  jac.get(2, 1) = DataVector(phi.size(), 0.0);  // 1/(R sin(th)) dz/dph

  return jac;
}

template <typename Frame>
typename types<Frame>::InvJacobian InvJacobian<Frame>::compute(
    const typename types<Frame>::ThetaPhi& theta_phi) noexcept {
  const auto& theta = theta_phi.get(0);
  const auto& phi = theta_phi.get(1);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);

  typename types<Frame>::InvJacobian inv_jac;
  inv_jac.get(0, 0) = cos_theta * cos_phi;          // R dth/dx
  inv_jac.get(0, 1) = cos_theta * sin_phi;          // R dth/dy
  inv_jac.get(0, 2) = -sin_theta;                   // R dth/dz
  inv_jac.get(1, 0) = -sin_phi;                     // R sin(th) dph/dx
  inv_jac.get(1, 1) = cos_phi;                      // R sin(th) dph/dy
  inv_jac.get(1, 2) = DataVector(phi.size(), 0.0);  // R sin(th) dph/dz

  return inv_jac;
}

template <typename Frame>
typename types<Frame>::InvHessian InvHessian<Frame>::compute(
    const typename types<Frame>::ThetaPhi& theta_phi) noexcept {
  const auto& theta = theta_phi.get(0);
  const auto& phi = theta_phi.get(1);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);

  typename types<Frame>::InvHessian inv_hess;
  const DataVector sin_sq_theta = square(sin_theta);
  const DataVector cos_sq_theta = square(cos_theta);
  const DataVector sin_theta_cos_theta = sin_theta * cos_theta;
  const DataVector sin_sq_phi = square(sin_phi);
  const DataVector cos_sq_phi = square(cos_phi);
  const DataVector sin_phicos_phi = sin_phi * cos_phi;
  const DataVector csc_theta = 1.0 / sin_theta;
  const DataVector f1 = 1.0 + 2.0 * sin_sq_theta;
  const DataVector cot_theta = cos_theta * csc_theta;

  // R^2 d^2 th/(dx^2)
  inv_hess.get(0, 0, 0) = cot_theta * (1.0 - cos_sq_phi * f1);
  // R^2 d^2 th/(dxdy)
  inv_hess.get(0, 0, 1) = -cot_theta * sin_phicos_phi * f1;
  // R^2 d^2 th/(dxdz)
  inv_hess.get(0, 0, 2) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dydx)
  inv_hess.get(0, 1, 0) = -cot_theta * sin_phicos_phi * f1;
  // R^2 d^2 th/(dy^2)
  inv_hess.get(0, 1, 1) = cot_theta * (1.0 - sin_sq_phi * f1);
  // R^2 d^2 th/(dydz)
  inv_hess.get(0, 1, 2) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dzdx)
  inv_hess.get(0, 2, 0) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dzdy)
  inv_hess.get(0, 2, 1) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dz^2)
  inv_hess.get(0, 2, 2) = 2.0 * sin_theta_cos_theta;
  // R^2 d/dx (sin(th) dph/dx)
  inv_hess.get(1, 0, 0) = sin_phicos_phi * (1 + sin_sq_theta) * csc_theta;
  // R^2 d/dx (sin(th) dph/dy)
  inv_hess.get(1, 0, 1) = (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
  // R^2 d/dx (sin(th) dph/dz)
  inv_hess.get(1, 0, 2) = DataVector(phi.size(), 0.0);
  // R^2 d/dy (sin(th) dph/dx)
  inv_hess.get(1, 1, 0) = (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dy)
  inv_hess.get(1, 1, 1) = -sin_phicos_phi * (1 + sin_sq_theta) * csc_theta;
  // R^2 d/dy (sin(th) dph/dz)
  inv_hess.get(1, 1, 2) = DataVector(phi.size(), 0.0);
  // R^2 d/dz (sin(th) dph/dx)
  inv_hess.get(1, 2, 0) = cos_theta * sin_phi;
  // R^2 d/dz (sin(th) dph/dy)
  inv_hess.get(1, 2, 1) = -cos_theta * cos_phi;
  // R^2 d/dz (sin(th) dph/dz)
  inv_hess.get(1, 2, 2) = DataVector(phi.size(), 0.0);

  return inv_hess;
}

template <typename Frame>
typename types<Frame>::Vector CartesianCoords<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const typename types<Frame>::OneForm& r_hat) noexcept {
  typename types<Frame>::Vector coords;
  for (size_t d = 0; d < 3; ++d) {
    coords.get(d) = gsl::at(strahlkorper.center(), d) + r_hat.get(d) * radius;
  }
  return coords;
}

template <typename Frame>
typename types<Frame>::OneForm DxRadius<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const typename types<Frame>::InvJacobian& inv_jac) noexcept {
  typename types<Frame>::OneForm dx_radius;
  const DataVector one_over_r = 1.0 / radius;
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  dx_radius.get(0) =
      (inv_jac.get(0, 0) * dr.get(0) + inv_jac.get(1, 0) * dr.get(1)) *
      one_over_r;
  dx_radius.get(1) =
      (inv_jac.get(0, 1) * dr.get(0) + inv_jac.get(1, 1) * dr.get(1)) *
      one_over_r;
  dx_radius.get(2) = inv_jac.get(0, 2) * dr.get(0) * one_over_r;
  return dx_radius;
}

template <typename Frame>
typename types<Frame>::SecondDeriv D2xRadius<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const typename types<Frame>::InvJacobian& inv_jac,
    const typename types<Frame>::InvHessian& inv_hess) noexcept {
  typename types<Frame>::SecondDeriv d2x_radius;
  const DataVector one_over_r_squared = 1.0 / square(radius);
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);

  for (size_t i = 0; i < 3; ++i) {
    // Diagonal terms.  Divide by square(r) later.
    d2x_radius.get(i, i) = DataVector(one_over_r_squared.size(), 0.0);
    for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
      d2x_radius.get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
      for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
        d2x_radius.get(i, i) +=
            derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
      }
    }
    d2x_radius.get(i, i) *= one_over_r_squared;
    // off_diagonal terms.  Symmetrize over i and j.
    // Divide by 2*square(r) later.
    for (size_t j = i + 1; j < 3; ++j) {
      d2x_radius.get(i, j) = DataVector(one_over_r_squared.size(), 0.0);
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_radius.get(i, j) += derivs.first.get(k) *
                                (inv_hess.get(k, i, j) + inv_hess.get(k, j, i));
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_radius.get(i, j) +=
              derivs.second.get(l, k) * (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                         inv_jac.get(k, j) * inv_jac.get(l, i));
        }
      }
      d2x_radius.get(i, j) *= 0.5 * one_over_r_squared;
    }
  }
  return d2x_radius;
}

template <typename Frame>
DataVector NablaSquaredRadius<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const typename types<Frame>::ThetaPhi& theta_phi) noexcept {
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);
  return derivs.second.get(0, 0) + derivs.second.get(1, 1) +
         derivs.first.get(0) / tan(theta_phi.get(0));
}

template <typename Frame>
typename types<Frame>::OneForm NormalOneForm<Frame>::compute(
    const typename types<Frame>::OneForm& dx_radius,
    const typename types<Frame>::OneForm& r_hat) noexcept {
  typename types<Frame>::OneForm one_form;
  for (size_t d = 0; d < 3; ++d) {
    one_form.get(d) = r_hat.get(d) - dx_radius.get(d);
  }
  return one_form;
}

template <typename Frame>
std::array<typename types<Frame>::Vector, 2> Tangents<Frame>::compute(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const typename types<Frame>::OneForm& r_hat,
    const typename types<Frame>::Jacobian& jac) noexcept {
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  std::array<typename types<Frame>::Vector, 2> tangents;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(tangents, i).get(j) =
          dr.get(i) * r_hat.get(j) + radius * jac.get(j, i);
    }
  }
  return tangents;
}

}  // namespace StrahlkorperDB

// ================================================================

namespace StrahlkorperDB {
template struct ThetaPhi<Frame::Inertial>;
template struct Rhat<Frame::Inertial>;
template struct Jacobian<Frame::Inertial>;
template struct InvJacobian<Frame::Inertial>;
template struct InvHessian<Frame::Inertial>;
template struct CartesianCoords<Frame::Inertial>;
template struct DxRadius<Frame::Inertial>;
template struct D2xRadius<Frame::Inertial>;
template struct NablaSquaredRadius<Frame::Inertial>;
template struct NormalOneForm<Frame::Inertial>;
template struct Tangents<Frame::Inertial>;
}  // namespace StrahlkorperDB
