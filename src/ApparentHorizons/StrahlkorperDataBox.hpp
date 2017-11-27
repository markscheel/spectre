// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// Holds tags and ComputeItems associated with a Strahlkorper.
namespace StrahlkorperDB {

// This struct supplies shorter names for longer types used below.
template <typename Frame>
struct types {
  using ThetaPhi = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
  using OneForm = tnsr::i<DataVector, 3, Frame>;
  using Vector = tnsr::I<DataVector, 3, Frame>;
  using Jacobian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Up, Frame>,
                        SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<Frame>>>>;
  using InvJacobian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                        SpatialIndex<3, UpLo::Lo, Frame>>>;
  using InvHessian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
             index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                        SpatialIndex<3, UpLo::Lo, Frame>,
                        SpatialIndex<3, UpLo::Lo, Frame>>>;
  using SecondDeriv = tnsr::ii<DataVector, 3, Frame>;
};

template <typename Frame>
struct Strahlkorper : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Strahlkorper";
  using type = ::Strahlkorper<Frame>;
};

/// \f$(\theta,\phi)\f$ on the grid.
template <typename Frame>
struct ThetaPhi : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ThetaPhi";
  static typename types<Frame>::ThetaPhi compute(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Frame>>;
};

/// \f$x_i/r\f$
template <typename Frame>
struct Rhat : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Rhat";
  static typename types<Frame>::OneForm compute(
      const typename types<Frame>::ThetaPhi& theta_phi) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// jacobian(i,j) is \f$\frac{1,r}\partial x^i/\partial\theta\f$,
/// \f$\frac{1,r\sin\theta}\partial x^i/\partial\phi\f$
template <typename Frame>
struct Jacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Jacobian";
  static typename types<Frame>::Jacobian compute(
      const typename types<Frame>::ThetaPhi& theta_phi) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// inv_jacobian(j,i) is \f$r\partial\theta/\partial x^i\f$,
///    \f$r\sin\theta\partial\phi/\partial x^i\f$
template <typename Frame>
struct InvJacobian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvJacobian";
  static typename types<Frame>::InvJacobian compute(
      const typename types<Frame>::ThetaPhi& theta_phi) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// inv_hessian(k,i,j) is \f$\partial\f$ijac(k,j)\f$/partial x^i\f$,
/// where \f$ijac\f$ is the inverse Jacobian.
/// It is not symmetric because the Jacobians are Pfaffian.
template <typename Frame>
struct InvHessian : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InvHessian";
  static typename types<Frame>::InvHessian compute(
      const typename types<Frame>::ThetaPhi& theta_phi) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<ThetaPhi<Frame>>;
};

/// (Euclidean) distance r of each grid point from center.
template <typename Frame>
struct Radius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Radius";
  SPECTRE_ALWAYS_INLINE static auto compute(
      const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    return strahlkorper.ylm_spherepack().spec_to_phys(
        strahlkorper.coefficients());
  }
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Frame>>;
};

/// \f$(x,y,z)\f$ of each point on the surface.
template <typename Frame>
struct CartesianCoords : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "CartesianCoords";
  static typename types<Frame>::Vector compute(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const typename types<Frame>::OneForm& r_hat) noexcept;
  static constexpr auto function = compute;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};

/// dx_radius(i) is \f$\partial r/\partial x^i\f$
template <typename Frame>
struct DxRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "DxRadius";
  static typename types<Frame>::OneForm compute(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const typename types<Frame>::InvJacobian& inv_jac) noexcept;
  static constexpr auto function = compute;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, InvJacobian<Frame>>;
};

/// d2x_radius(i,j) is \f$\partial^2 r/\partial x^i\partial x^j\f$
template <typename Frame>
struct D2xRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "D2xRadius";
  static typename types<Frame>::SecondDeriv compute(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const typename types<Frame>::InvJacobian& inv_jac,
      const typename types<Frame>::InvHessian& inv_hess) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Frame>, Radius<Frame>,
                                 InvJacobian<Frame>, InvHessian<Frame>>;
};

/// \f$\nabla^2 radius\f$
template <typename Frame>
struct NablaSquaredRadius : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "NablaSquaredRadius";
  static DataVector compute(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const typename types<Frame>::ThetaPhi& theta_phi) noexcept;
  static constexpr auto function = compute;
  using argument_tags =
      typelist<Strahlkorper<Frame>, Radius<Frame>, ThetaPhi<Frame>>;
};

/// Cartesian components of (unnormalized) one-form defining the surface.
/// This is computed by \f$x_i/r-\partial r/\partial x^i\f$,
/// where \f$x_i/r\f$ is `r_hat` and
/// \f$\partial r/\partial x^i\f$ is `dx_radius`.
template <typename Frame>
struct NormalOneForm : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "NormalOneForm";
  static typename types<Frame>::OneForm compute(
      const typename types<Frame>::OneForm& dx_radius,
      const typename types<Frame>::OneForm& r_hat) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<DxRadius<Frame>, Rhat<Frame>>;
};

/// tangents[j](i) is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the surface
/// (i.e. `cartesian_coords`)
/// and are considered functions of \f$(\theta,\phi)\f$,
/// \f$\partial/\partial q^0\f$ means \f$\partial/\partial\theta\f$,
/// and \f$\partial/\partial q^1\f$ means
/// \f$\csc\theta\partial/\partial\phi\f$.
template <typename Frame>
struct Tangents : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Tangents";
  static std::array<typename types<Frame>::Vector, 2> compute(
      const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
      const typename types<Frame>::OneForm& r_hat,
      const typename types<Frame>::Jacobian& jac) noexcept;
  static constexpr auto function = compute;
  using argument_tags = typelist<Strahlkorper<Frame>, Radius<Frame>,
                                 Rhat<Frame>, Jacobian<Frame>>;
};

template <typename Frame>
struct TagList {
  using Tags = typelist<Strahlkorper<Frame>>;
  using ComputeItemsTags =
      typelist<ThetaPhi<Frame>, Rhat<Frame>, Jacobian<Frame>,
               InvJacobian<Frame>, InvHessian<Frame>, Radius<Frame>,
               CartesianCoords<Frame>, DxRadius<Frame>, D2xRadius<Frame>,
               NablaSquaredRadius<Frame>, NormalOneForm<Frame>,
               Tangents<Frame>>;
};

}  // namespace StrahlkorperDB
