// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperFunctions.hpp"

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace StrahlkorperFunctions {

template <typename Frame>
void compute_radius(const gsl::not_null<Scalar<DataVector>*> radius,
                    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  get(*radius).destructive_resize(
      strahlkorper.ylm_spherepack().physical_size());
  get(*radius) =
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients());
}

template <typename Frame>
void compute_theta_phi(
    const gsl::not_null<tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>*>
        theta_phi,
    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  // If ylm_spherepack ever gets a not-null version of theta_phi_points,
  // then that can be used here to avoid an allocation.
  auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
  destructive_resize_components(theta_phi, temp[0].size());
  get<0>(*theta_phi) = temp[0];
  get<1>(*theta_phi) = temp[1];
}

template <typename Frame>
void compute_rhat(const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> r_hat,
                  const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
                      theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  destructive_resize_components(r_hat, theta.size());

  // Use one component of rhat for temporary storage, to avoid allocations.
  get<1>(*r_hat) = sin(theta);

  get<0>(*r_hat) = get<1>(*r_hat) * cos(phi);
  get<1>(*r_hat) *= sin(phi);
  get<2>(*r_hat) = cos(theta);
}

template <typename Frame>
void compute_cartesian_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame>*> coords,
    const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept {
  destructive_resize_components(coords, get(radius_of_strahlkorper).size());
  for (size_t d = 0; d < 3; ++d) {
    coords->get(d) = gsl::at(strahlkorper.center(), d) +
                     r_hat.get(d) * get(radius_of_strahlkorper);
  }
}

template <typename Frame>
void compute_cartesian_derivs_of_scalar(
    const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> dx_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const StrahlkorperTags::aliases::InvJacobian<Frame>& inv_jac) noexcept {
  destructive_resize_components(dx_scalar, get(scalar).size());

  // use dx_scalar component as temp to avoid allocation.
  get<2>(*dx_scalar) = 1.0 / get(radius_of_strahlkorper);  // 1/r

  const auto spherical_gradient_of_scalar =
      strahlkorper.ylm_spherepack().gradient(get(scalar));
  get<0>(*dx_scalar) =
      (get<0, 0>(inv_jac) * get<0>(spherical_gradient_of_scalar) +
       get<1, 0>(inv_jac) * get<1>(spherical_gradient_of_scalar)) *
      get<2>(*dx_scalar);
  get<1>(*dx_scalar) =
      (get<0, 1>(inv_jac) * get<0>(spherical_gradient_of_scalar) +
       get<1, 1>(inv_jac) * get<1>(spherical_gradient_of_scalar)) *
      get<2>(*dx_scalar);
  get<2>(*dx_scalar) *=
      get<0, 2>(inv_jac) * get<0>(spherical_gradient_of_scalar);
}

template <typename Frame>
void compute_cartesian_second_derivs_of_scalar(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> d2x_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const StrahlkorperTags::aliases::InvJacobian<Frame>& inv_jac,
    const StrahlkorperTags::aliases::InvHessian<Frame>& inv_hess) noexcept {
  destructive_resize_components(d2x_scalar, get(scalar).size());
  for (auto& component : *d2x_scalar) {
    component = 0.0;
  }
  const DataVector one_over_r_squared =
      1.0 / square(get(radius_of_strahlkorper));
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(get(scalar));

  for (size_t i = 0; i < 3; ++i) {
    // Diagonal terms.  Divide by square(r) later.
    for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
      d2x_scalar->get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
      for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
        d2x_scalar->get(i, i) +=
            derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
      }
    }
    d2x_scalar->get(i, i) *= one_over_r_squared;
    // off_diagonal terms.  Symmetrize over i and j.
    // Divide by 2*square(r) later.
    for (size_t j = i + 1; j < 3; ++j) {
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_scalar->get(i, j) += derivs.first.get(k) * (inv_hess.get(k, i, j) +
                                                        inv_hess.get(k, j, i));
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_scalar->get(i, j) +=
              derivs.second.get(l, k) * (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                         inv_jac.get(k, j) * inv_jac.get(l, i));
        }
      }
      d2x_scalar->get(i, j) *= 0.5 * one_over_r_squared;
    }
  }
}

template <typename Frame>
void compute_laplacian_of_scalar(
    const gsl::not_null<Scalar<DataVector>*> laplacian_of_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept {
  get(*laplacian_of_scalar).destructive_resize(get(scalar).size());
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(get(scalar));
  get(*laplacian_of_scalar) = get<0, 0>(derivs.second) +
                              get<1, 1>(derivs.second) +
                              get<0>(derivs.first) / tan(get<0>(theta_phi));
}

template <typename Frame>
void compute_normal_one_form(
    const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> normal_one_form,
    const tnsr::i<DataVector, 3, Frame>& cartesian_derivs_of_radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept {
  destructive_resize_components(normal_one_form, r_hat.begin()->size());
  for (size_t d = 0; d < 3; ++d) {
    normal_one_form->get(d) = r_hat.get(d) - cartesian_derivs_of_radius.get(d);
  }
}

template <typename Frame>
void compute_tangents(
    const gsl::not_null<StrahlkorperTags::aliases::Jacobian<Frame>*> tangents,
    const ::Strahlkorper<Frame>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jac) noexcept {
  destructive_resize_components(tangents, get(radius).size());
  const auto dr = strahlkorper.ylm_spherepack().gradient(get(radius));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      tangents->get(j, i) =
          dr.get(i) * r_hat.get(j) + get(radius) * jac.get(j, i);
    }
  }
}

template <typename Frame>
void compute_jacobian(
    const gsl::not_null<StrahlkorperTags::aliases::Jacobian<Frame>*> jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  destructive_resize_components(jac, theta.size());

  // Use 1,0 component of jac for temporary storage, to avoid allocations.
  get<1, 0>(*jac) = cos(theta);

  // Fill in components in carefully chosen order so as to avoid allocations.
  get<0, 1>(*jac) = -sin(phi);                          // 1/(R sin(th)) dx/dph
  get<1, 1>(*jac) = cos(phi);                           // 1/(R sin(th)) dy/dph
  get<2, 1>(*jac) = 0.0;                                // 1/(R sin(th)) dz/dph
  get<0, 0>(*jac) = get<1, 0>(*jac) * get<1, 1>(*jac);  // 1/R dx/dth
  get<1, 0>(*jac) *= -get<0, 1>(*jac);                  // 1/R dy/dth
  get<2, 0>(*jac) = -sin(theta);                        // 1/R dz/dth
}

template <typename Frame>
void compute_inverse_jacobian(
    const gsl::not_null<StrahlkorperTags::aliases::InvJacobian<Frame>*> inv_jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  destructive_resize_components(inv_jac, theta.size());

  // Use 0,1 component of inv_jac for temporary storage, to avoid allocations.
  get<0, 1>(*inv_jac) = cos(theta);

  // Fill in components in carefully chosen order so as to avoid allocations.
  get<1, 0>(*inv_jac) = -sin(phi);  // R sin(th) dph/dx
  get<1, 1>(*inv_jac) = cos(phi);   // R sin(th) dph/dy
  get<1, 2>(*inv_jac) = 0.0;        // R sin(th) dph/dz
  get<0, 0>(*inv_jac) = get<0, 1>(*inv_jac) * get<1, 1>(*inv_jac);  // R dth/dx
  get<0, 1>(*inv_jac) *= -get<1, 0>(*inv_jac);                      // R dth/dy
  get<0, 2>(*inv_jac) = -sin(theta);                                // R dth/dz
}

template <typename Frame>
void compute_inverse_hessian(
    const gsl::not_null<StrahlkorperTags::aliases::InvHessian<Frame>*> inv_hess,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  destructive_resize_components(inv_hess, theta.size());

  // Use some components of inv_hess for temporary storage, to avoid
  // allocations.
  get<1, 2, 2>(*inv_hess) = cos(theta);
  get<1, 1, 2>(*inv_hess) = sin(theta);
  get<1, 0, 2>(*inv_hess) = cos(phi);
  get<0, 2, 1>(*inv_hess) = sin(phi);
  get<1, 1, 1>(*inv_hess) = square(get<1, 1, 2>(*inv_hess));  // sin^2 theta
  get<0, 2, 0>(*inv_hess) = 1.0 / get<1, 1, 2>(*inv_hess);    // csc theta
  // cos(phi) sin(phi)
  get<0, 2, 1>(*inv_hess) = get<1, 0, 2>(*inv_hess) * get<0, 2, 1>(*inv_hess);
  get<0, 2, 2>(*inv_hess) = square(get<1, 0, 2>(*inv_hess));  // cos^2 phi
  get<1, 2, 1>(*inv_hess) = square(get<0, 2, 1>(*inv_hess));  // sin^2 phi

  // Fill in components of inv_hess in carefully chosen order,
  // to avoid allocations.

  // R^2 d/dx (sin(th) dph/dx)
  get<1, 0, 0>(*inv_hess) = get<0, 2, 1>(*inv_hess) *
                            (1.0 + get<1, 1, 1>(*inv_hess)) *
                            get<0, 2, 0>(*inv_hess);
  // R^2 d/dx (sin(th) dph/dy)
  get<1, 0, 1>(*inv_hess) =
      (get<1, 2, 1>(*inv_hess) -
       get<1, 1, 1>(*inv_hess) * get<0, 2, 2>(*inv_hess)) *
      get<0, 2, 0>(*inv_hess);
  // R^2 d/dy (sin(th) dph/dx)
  get<1, 1, 0>(*inv_hess) = (get<1, 1, 1>(*inv_hess) * get<1, 2, 1>(*inv_hess) -
                             get<0, 2, 2>(*inv_hess)) *
                            get<0, 2, 0>(*inv_hess);

  // More temps: now don't need csc theta anymore
  get<0, 2, 0>(*inv_hess) *= get<1, 2, 2>(*inv_hess);  // cot(theta)
  // 1 + 2 sin^2(theta)
  get<0, 0, 1>(*inv_hess) = 1.0 + 2.0 * get<1, 1, 1>(*inv_hess);

  // Fill in more inv_hess components in careful order, since some
  // of those components still contain temporaries.

  // R^2 d^2 th/(dy^2)
  get<0, 1, 1>(*inv_hess) =
      get<0, 2, 0>(*inv_hess) *
      (1.0 - get<1, 2, 1>(*inv_hess) * get<0, 0, 1>(*inv_hess));
  // R^2 d^2 th/(dx^2)
  get<0, 0, 0>(*inv_hess) =
      get<0, 2, 0>(*inv_hess) *
      (1.0 - get<0, 2, 2>(*inv_hess) * get<0, 0, 1>(*inv_hess));
  // R^2 d^2 th/(dxdy)
  get<0, 0, 1>(*inv_hess) = -get<0, 2, 0>(*inv_hess) * get<0, 2, 1>(*inv_hess) *
                            get<0, 0, 1>(*inv_hess);
  // R^2 d^2 th/(dxdz)
  get<0, 0, 2>(*inv_hess) =
      (2.0 * get<1, 1, 1>(*inv_hess) - 1.0) * get<1, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dydz)
  get<0, 1, 2>(*inv_hess) =
      (2.0 * get<1, 1, 1>(*inv_hess) - 1.0) * get<0, 2, 1>(*inv_hess);
  // R^2 d/dz (sin(th) dph/dx)
  get<1, 2, 0>(*inv_hess) = get<1, 2, 2>(*inv_hess) * get<0, 2, 1>(*inv_hess);
  // R^2 d/dz (sin(th) dph/dy)
  get<1, 2, 1>(*inv_hess) = -get<1, 2, 2>(*inv_hess) * get<1, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dz^2)
  get<0, 2, 2>(*inv_hess) =
      2.0 * get<1, 2, 2>(*inv_hess) * get<1, 1, 2>(*inv_hess);

  // R^2 d^2 th/(dydx)
  get<0, 1, 0>(*inv_hess) = get<0, 0, 1>(*inv_hess);
  // R^2 d^2 th/(dzdx)
  get<0, 2, 0>(*inv_hess) = get<0, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dzdy)
  get<0, 2, 1>(*inv_hess) = get<0, 1, 2>(*inv_hess);
  // R^2 d/dx (sin(th) dph/dz)
  get<1, 0, 2>(*inv_hess) = 0.0;
  // R^2 d/dy (sin(th) dph/dy)
  get<1, 1, 1>(*inv_hess) = -get<1, 0, 0>(*inv_hess);
  // R^2 d/dy (sin(th) dph/dz)
  get<1, 1, 2>(*inv_hess) = 0.0;
  // R^2 d/dz (sin(th) dph/dz)
  get<1, 2, 2>(*inv_hess) = 0.0;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                   \
  template void compute_radius(                                                \
      const gsl::not_null<Scalar<DataVector>*> radius,                         \
      const ::Strahlkorper<FRAME(data)>& strahlkorper) noexcept;               \
  template void compute_theta_phi(                                             \
      const gsl::not_null<                                                     \
          tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>*>            \
          theta_phi,                                                           \
      const ::Strahlkorper<FRAME(data)>& strahlkorper) noexcept;               \
  template void compute_rhat(                                                  \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> r_hat,         \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&           \
          theta_phi) noexcept;                                                 \
  template void compute_cartesian_coordinates(                                 \
      const gsl::not_null<tnsr::I<DataVector, 3, FRAME(data)>*> coords,        \
      const ::Strahlkorper<FRAME(data)>& strahlkorper,                         \
      const Scalar<DataVector>& radius_of_strahlkorper,                        \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat) noexcept;              \
  template void compute_cartesian_derivs_of_scalar(                            \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> dx_scalar,     \
      const Scalar<DataVector>& scalar,                                        \
      const ::Strahlkorper<FRAME(data)>& strahlkorper,                         \
      const Scalar<DataVector>& radius_of_strahlkorper,                        \
      const StrahlkorperTags::aliases::InvJacobian<FRAME(data)>&               \
          inv_jac) noexcept;                                                   \
  template void compute_cartesian_second_derivs_of_scalar(                     \
      const gsl::not_null<tnsr::ii<DataVector, 3, FRAME(data)>*> d2x_scalar,   \
      const Scalar<DataVector>& scalar,                                        \
      const ::Strahlkorper<FRAME(data)>& strahlkorper,                         \
      const Scalar<DataVector>& radius_of_strahlkorper,                        \
      const StrahlkorperTags::aliases::InvJacobian<FRAME(data)>& inv_jac,      \
      const StrahlkorperTags::aliases::InvHessian<FRAME(data)>&                \
          inv_hess) noexcept;                                                  \
  template void compute_laplacian_of_scalar(                                   \
      const gsl::not_null<Scalar<DataVector>*> laplacian_of_scalar,            \
      const Scalar<DataVector>& scalar,                                        \
      const ::Strahlkorper<FRAME(data)>& strahlkorper,                         \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&           \
          theta_phi) noexcept;                                                 \
  template void compute_normal_one_form(                                       \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*>                \
          normal_one_form,                                                     \
      const tnsr::i<DataVector, 3, FRAME(data)>& cartesian_derivs_of_radius,   \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat) noexcept;              \
  template void compute_tangents(                                              \
      const gsl::not_null<StrahlkorperTags::aliases::Jacobian<FRAME(data)>*>   \
          tangents,                                                            \
      const ::Strahlkorper<FRAME(data)>& strahlkorper,                         \
      const Scalar<DataVector>& radius,                                        \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat,                        \
      const StrahlkorperTags::aliases::Jacobian<FRAME(data)>& jac) noexcept;   \
  template void compute_jacobian(                                              \
      const gsl::not_null<StrahlkorperTags::aliases::Jacobian<FRAME(data)>*>   \
          jac,                                                                 \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&           \
          theta_phi) noexcept;                                                 \
  template void compute_inverse_jacobian(                                      \
      const gsl::not_null<                                                     \
          StrahlkorperTags::aliases::InvJacobian<FRAME(data)>*>                \
          inv_jac,                                                             \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&           \
          theta_phi) noexcept;                                                 \
  template void compute_inverse_hessian(                                       \
      const gsl::not_null<StrahlkorperTags::aliases::InvHessian<FRAME(data)>*> \
          inv_hess,                                                            \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&           \
          theta_phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME

}  // namespace StrahlkorperFunctions
