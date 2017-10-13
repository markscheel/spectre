// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/SpherepackIterator.hpp"
#include "Utilities/ConstantExpressions.hpp"

template <typename Fr>
Strahlkorper<Fr>::Strahlkorper(const size_t l_max, const size_t m_max,
                               const double radius,
                               std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(center)),  // NOLINT
      strahlkorper_coefs_(ylm_.spectral_size(), 0.0) {
  ylm_.add_constant(&strahlkorper_coefs_, radius);
}

template <typename Fr>
Strahlkorper<Fr>::Strahlkorper(
    const size_t l_max, const size_t m_max,
    const Strahlkorper& another_strahlkorper) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(another_strahlkorper.ylm_.prolong_or_restrict(
          another_strahlkorper.strahlkorper_coefs_, ylm_)) {}

template <typename Fr>
Strahlkorper<Fr>::Strahlkorper(const Strahlkorper& another_strahlkorper,
                               DataVector coefs) noexcept
    : l_max_(another_strahlkorper.l_max_),
      m_max_(another_strahlkorper.m_max_),
      ylm_(another_strahlkorper.ylm_),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(std::move(coefs)) {
  ASSERT(
      strahlkorper_coefs_.size() == another_strahlkorper.ylm_.spectral_size(),
      "Bad size " << strahlkorper_coefs_.size() << ", expected "
                  << another_strahlkorper.ylm_.spectral_size());
}

template <typename Fr>
Strahlkorper<Fr>::Strahlkorper(const DataVector& radius_at_collocation_points,
                               const size_t l_max, const size_t m_max,
                               std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(center),  // NOLINT
      strahlkorper_coefs_(ylm_.phys_to_spec(radius_at_collocation_points)) {
  ASSERT(radius_at_collocation_points.size() == ylm_.physical_size(),
         "Bad size " << radius_at_collocation_points.size() << ", expected "
                     << ylm_.physical_size());
}

template <typename Fr>
void Strahlkorper<Fr>::pup(PUP::er& p) {
  p | l_max_;
  p | m_max_;
  p | center_;
  p | strahlkorper_coefs_;
  // All variables other than the above are cached quantities that
  // can be recomputed from the above.

  if (p.isUnpacking()) {
    ylm_ = YlmSpherepack(l_max_, m_max_);
  }
}

template <typename Fr>
std::array<double, 3> Strahlkorper<Fr>::physical_center() const noexcept {
  // For better accuracy, one should do an integral of
  // x^i * AreaElement divided by Area.  The method here is an approximation
  // using only the l=1 modes.
  // See Eq. (37) of Hemberger et al, arXiv:1211.6079 for the form of the
  // exact physical center that we approximate here.
  std::array<double, 3> result = center_;
  SpherepackIterator it(l_max_, m_max_);
  result[0] += strahlkorper_coefs_[it.set(1, 1)()] * sqrt(0.75);
  result[1] -= strahlkorper_coefs_[it.set(1, -1)()] * sqrt(0.75);
  result[2] += strahlkorper_coefs_[it.set(1, 0)()] * sqrt(0.375);
  return result;
}

template <typename Fr>
double Strahlkorper<Fr>::average_radius() const noexcept {
  return ylm_.average(coefficients());
}

template <typename Fr>
double Strahlkorper<Fr>::radius(const double theta, const double phi) const
    noexcept {
  return ylm_.interpolate_from_coefs(strahlkorper_coefs_, {{{theta, phi}}})[0];
}

template <typename Fr>
bool Strahlkorper<Fr>::point_is_contained(const std::array<double, 3>& x) const
    noexcept {
  // The point is assumed to be in Cartesian coords in the Strahlkorper frame.
  // Make the point relative to the center of the Strahlkorper.
  auto xmc = x;
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(xmc, d) -= gsl::at(center_, d);
  }

  // Convert the point from Cartesian to spherical coordinates.
  const double r = sqrt(square(xmc[0]) + square(xmc[1]) + square(xmc[2]));
  const double theta = acos(xmc[2] / r);
  double phi = atan2(xmc[1], xmc[0]);
  // Range of atan2 is [-pi,pi],
  // So adjust to get range [0,2pi]
  if (phi < 0.0) {
    phi += 2.0 * M_PI;
  }

  // Is the point inside the surface?
  return r < radius(theta, phi);
}

template <typename Fr>
void Strahlkorper<Fr>::initialize_jac_and_hess() const noexcept {
  if (jac_.get(0, 0).size() > 0) {
    return;
  }
  initialize_mesh_quantities();

  const auto theta_phi = ylm_.theta_phi_points();
  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const auto sin_phi = sin(phi);
  const auto cos_phi = cos(phi);
  const auto sin_theta = sin(theta);
  const auto cos_theta = cos(theta);

  jac_.get(0, 0) = cos_theta * cos_phi;          // 1/R dx/dth
  jac_.get(1, 0) = cos_theta * sin_phi;          // 1/R dy/dth
  jac_.get(2, 0) = -sin_theta;                   // 1/R dz/dth
  jac_.get(0, 1) = -sin_phi;                     // 1/(R sin(th)) dx/dph
  jac_.get(1, 1) = cos_phi;                      // 1/(R sin(th)) dy/dph
  jac_.get(2, 1) = DataVector(phi.size(), 0.0);  // 1/(R sin(th)) dz/dph

  inv_jac_.get(0, 0) = cos_theta * cos_phi;          // R dth/dx
  inv_jac_.get(0, 1) = cos_theta * sin_phi;          // R dth/dy
  inv_jac_.get(0, 2) = -sin_theta;                   // R dth/dz
  inv_jac_.get(1, 0) = -sin_phi;                     // R sin(th) dph/dx
  inv_jac_.get(1, 1) = cos_phi;                      // R sin(th) dph/dy
  inv_jac_.get(1, 2) = DataVector(phi.size(), 0.0);  // R sin(th) dph/dz

  const auto sin_sq_theta = square(sin_theta);
  const auto cos_sq_theta = square(cos_theta);
  const auto sin_theta_cos_theta = sin_theta * cos_theta;
  const auto sin_sq_phi = square(sin_phi);
  const auto cos_sq_phi = square(cos_phi);
  const auto sin_phicos_phi = sin_phi * cos_phi;
  const auto csc_theta = 1.0 / sin_theta;
  const auto f1 = 1.0 + 2.0 * sin_sq_theta;
  const auto cot_theta = cos_theta * csc_theta;
  // R^2 d^2 th/(dx^2)
  inv_hess_.get(0, 0, 0) = cot_theta * (1.0 - cos_sq_phi * f1);
  // R^2 d^2 th/(dxdy)
  inv_hess_.get(0, 0, 1) = -cot_theta * sin_phicos_phi * f1;
  // R^2 d^2 th/(dxdz)
  inv_hess_.get(0, 0, 2) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dydx)
  inv_hess_.get(0, 1, 0) = -cot_theta * sin_phicos_phi * f1;
  // R^2 d^2 th/(dy^2)
  inv_hess_.get(0, 1, 1) = cot_theta * (1.0 - sin_sq_phi * f1);
  // R^2 d^2 th/(dydz)
  inv_hess_.get(0, 1, 2) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dzdx)
  inv_hess_.get(0, 2, 0) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dzdy)
  inv_hess_.get(0, 2, 1) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dz^2)
  inv_hess_.get(0, 2, 2) = 2.0 * sin_theta_cos_theta;
  // R^2 d/dx (sin(th) dph/dx)
  inv_hess_.get(1, 0, 0) = sin_phicos_phi * (1 + sin_sq_theta) * csc_theta;
  // R^2 d/dx (sin(th) dph/dy)
  inv_hess_.get(1, 0, 1) = (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
  // R^2 d/dx (sin(th) dph/dz)
  inv_hess_.get(1, 0, 2) = DataVector(phi.size(), 0.0);
  // R^2 d/dy (sin(th) dph/dx)
  inv_hess_.get(1, 1, 0) = (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dy)
  inv_hess_.get(1, 1, 1) = -sin_phicos_phi * (1 + sin_sq_theta) * csc_theta;
  // R^2 d/dy (sin(th) dph/dz)
  inv_hess_.get(1, 1, 2) = DataVector(phi.size(), 0.0);
  // R^2 d/dz (sin(th) dph/dx)
  inv_hess_.get(1, 2, 0) = cos_theta * sin_phi;
  // R^2 d/dz (sin(th) dph/dy)
  inv_hess_.get(1, 2, 1) = -cos_theta * cos_phi;
  // R^2 d/dz (sin(th) dph/dz)
  inv_hess_.get(1, 2, 2) = DataVector(phi.size(), 0.0);
}

template <typename Fr>
void Strahlkorper<Fr>::initialize_mesh_quantities() const noexcept {
  if (theta_phi_.get(0).size() > 0) {
    return;
  }

  auto theta_phi = ylm_.theta_phi_points();
  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  theta_phi_.get(0) = theta;
  theta_phi_.get(1) = phi;

  const auto sin_theta = sin(theta);

  r_hat_.get(0) = sin_theta * cos(phi);
  r_hat_.get(1) = sin_theta * sin(phi);
  r_hat_.get(2) = cos(theta);
}

template <typename Fr>
const typename Strahlkorper<Fr>::ThetaPhiVector& Strahlkorper<Fr>::theta_phi()
    const noexcept {
  initialize_mesh_quantities();
  return theta_phi_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::OneForm& Strahlkorper<Fr>::r_hat() const
    noexcept {
  initialize_mesh_quantities();
  return r_hat_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::Jacobian& Strahlkorper<Fr>::jacobian() const
    noexcept {
  initialize_jac_and_hess();
  return jac_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::InvJacobian& Strahlkorper<Fr>::inv_jacobian()
    const noexcept {
  initialize_jac_and_hess();
  return inv_jac_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::InvHessian& Strahlkorper<Fr>::inv_hessian()
    const noexcept {
  initialize_jac_and_hess();
  return inv_hess_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::ThreeVector&
Strahlkorper<Fr>::surface_cartesian_coords() const noexcept {
  if (0 == global_coords_.get(0).size()) {
    const auto& r = radius();
    const auto& n = r_hat();
    for (size_t d = 0; d < 3; ++d) {
      global_coords_.get(d) = gsl::at(center_, d) + n.get(d) * r;
    }
  }
  return global_coords_;
}

template <typename Fr>
const DataVector& Strahlkorper<Fr>::radius() const noexcept {
  if (radius_.size() == 0) {
    radius_ = ylm_.spec_to_phys(strahlkorper_coefs_);
  }
  return radius_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::OneForm& Strahlkorper<Fr>::dx_radius() const
    noexcept {
  if (0 == dx_radius_.get(0).size()) {
    const auto one_over_r = 1.0 / radius();
    const auto dr = ylm_.gradient(radius());
    const auto& inv_jac = inv_jacobian();

    dx_radius_.get(0) =
        (inv_jac.get(0, 0) * dr.get(0) + inv_jac.get(1, 0) * dr.get(1)) *
        one_over_r;
    dx_radius_.get(1) =
        (inv_jac.get(0, 1) * dr.get(0) + inv_jac.get(1, 1) * dr.get(1)) *
        one_over_r;
    dx_radius_.get(2) = inv_jac.get(0, 2) * dr.get(0) * one_over_r;
  }
  return dx_radius_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::SecondDeriv& Strahlkorper<Fr>::d2x_radius()
    const noexcept {
  if (0 == d2x_radius_.get(0, 0).size()) {
    const auto one_over_r_squared = 1.0 / square(radius());
    const auto derivs = ylm_.first_and_second_derivative(radius());
    const auto& inv_jac = inv_jacobian();
    const auto& inv_hess = inv_hessian();

    for (size_t i = 0; i < 3; ++i) {
      // Diagonal terms.  Divide by square(r) later.
      d2x_radius_.get(i, i) = DataVector(one_over_r_squared.size(), 0.0);
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_radius_.get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_radius_.get(i, i) +=
              derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
        }
      }
      d2x_radius_.get(i, i) *= one_over_r_squared;
      // off_diagonal terms.  Symmetrize over i and j.
      // Divide by 2*square(r) later.
      for (size_t j = i + 1; j < 3; ++j) {
        d2x_radius_.get(i, j) = DataVector(one_over_r_squared.size(), 0.0);
        for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
          d2x_radius_.get(i, j) +=
              derivs.first.get(k) *
              (inv_hess.get(k, i, j) + inv_hess.get(k, j, i));
          for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
            d2x_radius_.get(i, j) += derivs.second.get(l, k) *
                                     (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                      inv_jac.get(k, j) * inv_jac.get(l, i));
          }
        }
        d2x_radius_.get(i, j) *= 0.5 * one_over_r_squared;
      }
    }
  }
  return d2x_radius_;
}

template <typename Fr>
const DataVector& Strahlkorper<Fr>::nabla_squared_radius() const noexcept {
  if (0 == nabla_squared_radius_.size()) {
    const auto derivs = ylm_.first_and_second_derivative(radius());

    // this is d2theta(r) + cot(theta)*dtheta(r) + d2phi(r)/sin^2(theta)
    nabla_squared_radius_ = derivs.second.get(0, 0) + derivs.second.get(1, 1) +
                            derivs.first.get(0) / tan(theta_phi().get(0));
  }

  return nabla_squared_radius_;
}

template <typename Fr>
const typename Strahlkorper<Fr>::OneForm&
Strahlkorper<Fr>::surface_normal_one_form() const noexcept {
  if (0 == surface_normal_one_form_.get(0).size()) {
    const auto& dr = dx_radius();
    const auto& n = r_hat();
    for (size_t d = 0; d < 3; ++d) {
      surface_normal_one_form_.get(d) = n.get(d) - dr.get(d);
    }
  }
  return surface_normal_one_form_;
}

template <typename Fr>
DataVector Strahlkorper<Fr>::surface_normal_magnitude(
    const InverseMetric& inv_g) const noexcept {
  const auto& s_i = surface_normal_one_form();
  ASSERT(s_i.get(0).size() == inv_g.get(0, 0).size(),
         "Size mismatch: " << s_i.get(0).size() << " vs "
                           << inv_g.get(0, 0).size());
  DataVector mag_squared(s_i.get(0).size(), 0.);
  for (size_t m = 0; m < 3; ++m) {
    for (size_t n = 0; n < 3; ++n) {
      mag_squared += s_i.get(m) * s_i.get(n) * inv_g.get(m, n);
    }
  }
  return sqrt(mag_squared);
}

template <typename Fr>
const std::array<typename Strahlkorper<Fr>::ThreeVector, 2>&
Strahlkorper<Fr>::surface_tangents() const noexcept {
  if (0 == gsl::at(surface_tangents_, 0).get(0).size()) {
    const auto& r = radius();
    const auto& n = r_hat();
    const auto dr = ylm_.gradient(r);
    const auto& jac = jacobian();

    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(surface_tangents_, i).get(j) =
            dr.get(i) * n.get(j) + r * jac.get(j, i);
      }
    }
  }
  return surface_tangents_;
}

// ================================================================

template class Strahlkorper<Frame::Inertial>;
