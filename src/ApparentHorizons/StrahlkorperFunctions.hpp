// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/Tensor/IndexType.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
class DataVector;
template <typename Frame>
class Strahlkorper;
/// \endcond

/// Functions defined on a Strahlkorper.
namespace StrahlkorperFunctions {

/// (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the
/// center to each point of the Strahlkorper surface.
template <typename Frame>
void compute_radius(const gsl::not_null<Scalar<DataVector>*> radius,
                    const ::Strahlkorper<Frame>& strahlkorper) noexcept;

/// \f$(\theta,\phi)\f$ on the Strahlkorper surface.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
void compute_theta_phi(
    const gsl::not_null<tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>*>
        theta_phi,
    const ::Strahlkorper<Frame>& strahlkorper) noexcept;

/// `r_hat(i)` is \f$\hat{r}^i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the
/// Strahlkorper surface.  Doesn't depend on the shape of the surface.
template <typename Frame>
void compute_rhat(const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> r_hat,
                  const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
                      theta_phi) noexcept;

/// `coords(i)` is \f$x_{\rm surf}^i\f$, the vector of \f$(x,y,z)\f$
/// coordinates of each point on the Strahlkorper surface.
template <typename Frame>
void compute_cartesian_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame>*> coords,
    const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept;

/// `dx_scalar(i)` is \f$\partial S/\partial x^i\f$ where \f$S\f$ is a
/// scalar defined on the surface.  Here \f$S=S(\theta,\phi)\f$ is
/// considered a function of Cartesian coordinates,
/// i.e. \f$S=S(\theta(x,y,z),\phi(x,y,z))\f$ for this operation.
/// Also `radius_of_strahlkorper` is
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function describing
/// the surface, which is considered a function of Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$ for this
/// operation.
template <typename Frame>
void compute_cartesian_derivs_of_scalar(
    const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> dx_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const StrahlkorperTags::aliases::InvJacobian<Frame>& inv_jac) noexcept;

/// `d2x_scalar(i,j)` is \f$\partial^2 S/\partial x^i\partial x^j\f$,
/// where \f$S\f$ is a scalar defined on the surface.  Here
/// \f$S=S(\theta,\phi)\f$ is considered a function of Cartesian
/// coordinates, i.e. \f$S=S(\theta(x,y,z),\phi(x,y,z))\f$ for this
/// operation.
/// Also `radius_of_strahlkorper` is
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function describing
/// the surface, which is considered a function of Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$ for this
/// operation.
template <typename Frame>
void compute_cartesian_second_derivs_of_scalar(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> d2x_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const StrahlkorperTags::aliases::InvJacobian<Frame>& inv_jac,
    const StrahlkorperTags::aliases::InvHessian<Frame>& inv_hess) noexcept;

/// \f$\nabla^2 S\f$, the flat Laplacian of a scalar on the surface.
/// This is \f$\eta^{ij}\partial^2 S/\partial x^i\partial x^j\f$,
/// where \f$S=S(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
void compute_laplacian_of_scalar(
    const gsl::not_null<Scalar<DataVector>*> laplacian_of_scalar,
    const Scalar<DataVector>& scalar, const ::Strahlkorper<Frame>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept;

/// `normal_one_form(i)` is \f$s_i\f$, the (unnormalized) normal one-form
/// to the surface, expressed in Cartesian components.
/// This is computed by \f$x_i/r-\partial r_{\rm surf}/\partial x^i\f$,
/// where \f$x_i/r\f$ is `r_hat` and
/// \f$\partial r_{\rm surf}/\partial x^i\f$ is `cartesian_derivs_of_radius`.
/// See Eq. (8) of \cite Baumgarte1996hh.
/// Note on the word "normal": \f$s_i\f$ points in the correct direction
/// (it is "normal" to the surface), but it does not have unit length
/// (it is not "normalized"; normalization requires a metric).
template <typename Frame>
void compute_normal_one_form(
    const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> normal_one_form,
    const tnsr::i<DataVector, 3, Frame>& cartesian_derivs_of_radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept;

/// `tangents(i,j)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the
/// surface (i.e. `cartesian_coords`) and are considered functions of
/// \f$(\theta,\phi)\f$.
///
/// \f$\partial/\partial q^0\f$ means
/// \f$\partial/\partial\theta\f$; and \f$\partial/\partial q^1\f$
/// means \f$\csc\theta\,\,\partial/\partial\phi\f$.  Note that the
/// vectors `tangents(i,0)` and `tangents(i,1)` are orthogonal to the
/// `normal_one_form` \f$s_i\f$, i.e.
/// \f$s_i \partial x_{\rm surf}^i/\partial q^j = 0\f$; this statement
/// is independent of a metric.  Also, `Tangents(i,0)` and
/// `Tangents(i,1)` are not necessarily orthogonal to each other,
/// since orthogonality between 2 vectors (as opposed to a vector and
/// a one-form) is metric-dependent.
template <typename Frame>
void compute_tangents(
    const gsl::not_null<StrahlkorperTags::aliases::Jacobian<Frame>*> tangents,
    const ::Strahlkorper<Frame>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jac) noexcept;

/// `jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$, i.e. the radius of the
/// Strahlkorper surface at each point.
/// `jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
void compute_jacobian(
    const gsl::not_null<StrahlkorperTags::aliases::Jacobian<Frame>*> jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept;

/// `inv_jac(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `inv_jac(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$, i.e. the radius of the
/// Strahlkorper surface at each point.
/// `inv_jac` doesn't depend on the shape of the surface.
template <typename Frame>
void compute_inverse_jacobian(
    const gsl::not_null<StrahlkorperTags::aliases::InvJacobian<Frame>*> inv_jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept;

/// `inv_hess(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `inv_hess` is not symmetric because the Jacobians are Pfaffian.
/// `inv_hess` doesn't depend on the shape of the surface.
template <typename Frame>
void compute_inverse_hessian(
    const gsl::not_null<StrahlkorperTags::aliases::InvHessian<Frame>*> inv_hess,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>&
        theta_phi) noexcept;
}  // namespace StrahlkorperFunctions
