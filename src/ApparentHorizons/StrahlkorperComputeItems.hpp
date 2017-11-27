// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace StrahlkorperDataBoxTags {

template <class Fr>
struct Strahlkorper : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Strahlkorper";
  using type = ::Strahlkorper<Fr>;
};

/// \f$(\theta,\phi)\f$ on the grid.
template <class Fr>
struct ThetaPhi : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ThetaPhi";
  static typename ::Strahlkorper<Fr>::ThetaPhiVector compute(
      const ::Strahlkorper<Fr>& strahlkorper) noexcept {
    typename ::Strahlkorper<Fr>::ThetaPhiVector theta_phi;
    auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
    theta_phi.get(0) = std::move(temp[0]);
    theta_phi.get(1) = std::move(temp[1]);
    return theta_phi;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Fr>>;
};

/// \f$x_i/r\f$
template <class Fr>
struct Rhat : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Rhat";
  static typename ::Strahlkorper<Fr>::OneForm compute(
      const typename ::Strahlkorper<Fr>::ThetaPhiVector& theta_phi) noexcept {
    typename ::Strahlkorper<Fr>::OneForm r_hat;

    const auto& theta = theta_phi.get(0);
    const auto& phi = theta_phi.get(1);

    const DataVector sin_theta = sin(theta);
    r_hat.get(0) = sin_theta * cos(phi);
    r_hat.get(1) = sin_theta * sin(phi);
    r_hat.get(2) = cos(theta);
    return r_hat;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Fr>>;
};

/// jacobian(i,j) is \f$\frac{1,r}\partial x^i/\partial\theta\f$,
/// \f$\frac{1,r\sin\theta}\partial x^i/\partial\phi\f$
template <class Fr>
struct Jacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Jacobian";
  static auto compute(
      const typename ::Strahlkorper<Fr>::ThetaPhiVector& theta_phi) noexcept {
    const auto& theta = theta_phi.get(0);
    const auto& phi = theta_phi.get(1);
    const DataVector sin_phi = sin(phi);
    const DataVector cos_phi = cos(phi);
    const DataVector sin_theta = sin(theta);
    const DataVector cos_theta = cos(theta);

    typename ::Strahlkorper<Fr>::Jacobian jac;
    jac.get(0, 0) = cos_theta * cos_phi;          // 1/R dx/dth
    jac.get(1, 0) = cos_theta * sin_phi;          // 1/R dy/dth
    jac.get(2, 0) = -sin_theta;                   // 1/R dz/dth
    jac.get(0, 1) = -sin_phi;                     // 1/(R sin(th)) dx/dph
    jac.get(1, 1) = cos_phi;                      // 1/(R sin(th)) dy/dph
    jac.get(2, 1) = DataVector(phi.size(), 0.0);  // 1/(R sin(th)) dz/dph

    return jac;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Fr>>;
};

/// inv_jacobian(j,i) is \f$r\partial\theta/\partial x^i\f$,
///    \f$r\sin\theta\partial\phi/\partial x^i\f$
template <class Fr>
struct InvJacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvJacobian";
  static auto compute(
      const typename ::Strahlkorper<Fr>::ThetaPhiVector& theta_phi) noexcept {
    const auto& theta = theta_phi.get(0);
    const auto& phi = theta_phi.get(1);
    const DataVector sin_phi = sin(phi);
    const DataVector cos_phi = cos(phi);
    const DataVector sin_theta = sin(theta);
    const DataVector cos_theta = cos(theta);

    typename ::Strahlkorper<Fr>::InvJacobian inv_jac;
    inv_jac.get(0, 0) = cos_theta * cos_phi;          // R dth/dx
    inv_jac.get(0, 1) = cos_theta * sin_phi;          // R dth/dy
    inv_jac.get(0, 2) = -sin_theta;                   // R dth/dz
    inv_jac.get(1, 0) = -sin_phi;                     // R sin(th) dph/dx
    inv_jac.get(1, 1) = cos_phi;                      // R sin(th) dph/dy
    inv_jac.get(1, 2) = DataVector(phi.size(), 0.0);  // R sin(th) dph/dz

    return inv_jac;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Fr>>;
};

/// inv_hessian(k,i,j) is \f$\partial\f$ijac(k,j)\f$/partial x^i\f$,
/// where \f$ijac\f$ is the inverse Jacobian.
/// It is not symmetric because the Jacobians are Pfaffian.
template <class Fr>
struct InvHessian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvHessian";
  static auto compute(
      const typename ::Strahlkorper<Fr>::ThetaPhiVector& theta_phi) noexcept {
    const auto& theta = theta_phi.get(0);
    const auto& phi = theta_phi.get(1);
    const DataVector sin_phi = sin(phi);
    const DataVector cos_phi = cos(phi);
    const DataVector sin_theta = sin(theta);
    const DataVector cos_theta = cos(theta);

    typename ::Strahlkorper<Fr>::InvHessian inv_hess;
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
    inv_hess.get(1, 0, 1) =
        (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
    // R^2 d/dx (sin(th) dph/dz)
    inv_hess.get(1, 0, 2) = DataVector(phi.size(), 0.0);
    // R^2 d/dy (sin(th) dph/dx)
    inv_hess.get(1, 1, 0) =
        (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
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
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Fr>>;
};

/// (Euclidean) distance r of each grid point from center.
template <class Fr>
struct Radius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Radius";
  static auto compute(const ::Strahlkorper<Fr>& strahlkorper) noexcept {
    return strahlkorper.ylm_spherepack().spec_to_phys(
        strahlkorper.coefficients());
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Fr>>;
};

/// \f$(x,y,z)\f$ of each point on the surface.
template <class Fr>
struct SurfaceCartesianCoords : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "SurfaceCartesianCoords";
  static auto compute(
      const ::Strahlkorper<Fr>& strahlkorper, const DataVector& radius,
      const typename ::Strahlkorper<Fr>::OneForm& r_hat) noexcept {
    typename ::Strahlkorper<Fr>::ThreeVector coords;
    for (size_t d = 0; d < 3; ++d) {
      coords.get(d) = gsl::at(strahlkorper.center(), d) + r_hat.get(d) * radius;
    }
    return coords;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Fr>, Radius<Fr>, Rhat<Fr>>;
};

/// dx_radius(i) is \f$\partial r/\partial x^i\f$
template <class Fr>
struct DxRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "DxRadius";
  static auto compute(
      const ::Strahlkorper<Fr>& strahlkorper, const DataVector& radius,
      const typename ::Strahlkorper<Fr>::InvJacobian& inv_jac) noexcept {
    typename ::Strahlkorper<Fr>::OneForm dx_radius;
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
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Fr>, Radius<Fr>, InvJacobian<Fr>>;
};

/// d2x_radius(i,j) is \f$\partial^2 r/\partial x^i\partial x^j\f$
template <class Fr>
struct D2xRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "D2xRadius";
  static auto compute(
      const ::Strahlkorper<Fr>& strahlkorper, const DataVector& radius,
      const typename ::Strahlkorper<Fr>::InvJacobian& inv_jac,
      const typename ::Strahlkorper<Fr>::InvHessian& inv_hess) noexcept {
    typename ::Strahlkorper<Fr>::SecondDeriv d2x_radius;
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
          d2x_radius.get(i, j) += derivs.first.get(k) * (inv_hess.get(k, i, j) +
                                                         inv_hess.get(k, j, i));
          for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
            d2x_radius.get(i, j) += derivs.second.get(l, k) *
                                    (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                     inv_jac.get(k, j) * inv_jac.get(l, i));
          }
        }
        d2x_radius.get(i, j) *= 0.5 * one_over_r_squared;
      }
    }
    return d2x_radius;
  }
  static constexpr auto function = compute;
  using argument_tags =
      typelist<Strahlkorper<Fr>, Radius<Fr>, InvJacobian<Fr>, InvHessian<Fr>>;
};

/// \f$\nabla^2 radius\f$
template <class Fr>
struct NablaSquaredRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "NablaSquaredRadius";
  static DataVector compute(
      const ::Strahlkorper<Fr>& strahlkorper, const DataVector& radius,
      const typename ::Strahlkorper<Fr>::ThetaPhiVector& theta_phi) noexcept {
    const auto derivs =
        strahlkorper.ylm_spherepack().first_and_second_derivative(radius);
    return derivs.second.get(0, 0) + derivs.second.get(1, 1) +
           derivs.first.get(0) / tan(theta_phi.get(0));
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Fr>, Radius<Fr>, ThetaPhi<Fr>>;
};

/// Cartesian components of (unnormalized) one-form defining the surface.
/// This is computed by \f$x_i/r-\partial r/\partial x^i\f$,
/// where \f$x_i/r\f$ is `r_hat` and
/// \f$\partial r/\partial x^i\f$ is `dx_radius`.
template <class Fr>
struct SurfaceNormalOneForm : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "SurfaceNormalOneForm";
  static auto compute(
      const typename ::Strahlkorper<Fr>::OneForm& dx_radius,
      const typename ::Strahlkorper<Fr>::OneForm& r_hat) noexcept {
    typename ::Strahlkorper<Fr>::OneForm one_form;
    for (size_t d = 0; d < 3; ++d) {
      one_form.get(d) = r_hat.get(d) - dx_radius.get(d);
    }
    return one_form;
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<DxRadius<Fr>, Rhat<Fr>>;
};

/// surface_tangents[j](i) is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the surface
/// (i.e. `surface_cartesian_coords`)
/// and are considered functions of \f$(\theta,\phi)\f$,
/// \f$\partial/\partial q^0\f$ means \f$\partial/\partial\theta\f$,
/// and \f$\partial/\partial q^1\f$ means
/// \f$\csc\theta\partial/\partial\phi\f$.
template <class Fr>
struct SurfaceTangents : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "SurfaceTangents";
  static auto compute(
      const ::Strahlkorper<Fr>& strahlkorper, const DataVector& radius,
      const typename ::Strahlkorper<Fr>::OneForm& r_hat,
      const typename ::Strahlkorper<Fr>::Jacobian& jac) noexcept {
    const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
    std::array<typename ::Strahlkorper<Fr>::ThreeVector, 2> tangents;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(tangents, i).get(j) =
            dr.get(i) * r_hat.get(j) + radius * jac.get(j, i);
      }
    }
    return tangents;
  }
  static constexpr auto function = compute;
  using argument_tags =
      typelist<Strahlkorper<Fr>, Radius<Fr>, Rhat<Fr>, Jacobian<Fr>>;
};

template <class Fr>
struct TagList {
  using Tags = typelist<Strahlkorper<Fr>>;
  using ComputeItemsTags =
      typelist<ThetaPhi<Fr>, Rhat<Fr>, Jacobian<Fr>, InvJacobian<Fr>,
               InvHessian<Fr>, Radius<Fr>, SurfaceCartesianCoords<Fr>,
               DxRadius<Fr>, D2xRadius<Fr>, NablaSquaredRadius<Fr>,
               SurfaceNormalOneForm<Fr>, SurfaceTangents<Fr>>;
};

}  // StrahlkorperDataBoxTags
