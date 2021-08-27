// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare gr::Tags::SpatialMetric
/// \cond
class DataVector;
class FastFlow;
/// \endcond

namespace ah::Tags {
struct FastFlow : db::SimpleTag {
  using type = ::FastFlow;
};
/// Tag for holding the previously-found value of a Strahlkorper,
/// which is saved for extrapolation for future initial guesses
/// and for recovering the initial guess on failure of the algorithm.
template <typename Frame>
struct PreviousStrahlkorper : db::SimpleTag {
  using type = ::Strahlkorper<Frame>;
};
/// Tag for holding the inertial-frame Strahlkorper,
/// when the apparent horizon is found in a frame other than the
/// inertial frame.
struct InertialStrahlkorper : db::SimpleTag {
  using type = ::Strahlkorper<::Frame::Inertial>;
};
}  // namespace ah::Tags

/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper`.
namespace StrahlkorperTags {

/// Tag referring to a `::Strahlkorper`
template <typename Frame>
struct Strahlkorper : db::SimpleTag {
  using type = ::Strahlkorper<Frame>;
};

/// @{
/// \f$(\theta,\phi)\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct ThetaPhi : db::SimpleTag {
  using type = aliases::ThetaPhi<Frame>;
};

template <typename Frame>
struct ThetaPhiCompute : ThetaPhi<Frame>, db::ComputeTag {
  using base = ThetaPhi<Frame>;
  using return_type = aliases::ThetaPhi<Frame>;
  static void function(gsl::not_null<aliases::ThetaPhi<Frame>*> theta_phi,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};
/// @}

/// @{
/// `Rhat(i)` is \f$\hat{r}^i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the grid.
/// Doesn't depend on the shape of the surface.
template <typename Frame>
struct Rhat : db::SimpleTag {
  using type = aliases::OneForm<Frame>;
};

template <typename Frame>
struct RhatCompute : Rhat<Frame>, db::ComputeTag {
  using base = Rhat<Frame>;
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> r_hat,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
/// and `Jacobian(i,1)`
/// is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `Jacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct Jacobian : db::SimpleTag {
  using type = aliases::Jacobian<Frame>;
};

template <typename Frame>
struct JacobianCompute : Jacobian<Frame>, db::ComputeTag {
  using base = Jacobian<Frame>;
  using return_type = aliases::Jacobian<Frame>;
  static void function(gsl::not_null<aliases::Jacobian<Frame>*> jac,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
/// and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
/// Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
/// `InvJacobian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvJacobian : db::SimpleTag {
  using type = aliases::InvJacobian<Frame>;
};

template <typename Frame>
struct InvJacobianCompute : InvJacobian<Frame>, db::ComputeTag {
  using base = InvJacobian<Frame>;
  using return_type = aliases::InvJacobian<Frame>;
  static void function(gsl::not_null<aliases::InvJacobian<Frame>*> inv_jac,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `InvHessian(k,i,j)` is \f$\partial (J^{-1}){}^k_j/\partial x^i\f$,
/// where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
/// `InvHessian` is not symmetric because the Jacobians are Pfaffian.
/// `InvHessian` doesn't depend on the shape of the surface.
template <typename Frame>
struct InvHessian : db::SimpleTag {
  using type = aliases::InvHessian<Frame>;
};

template <typename Frame>
struct InvHessianCompute : InvHessian<Frame>, db::ComputeTag {
  using base = InvHessian<Frame>;
  using return_type = aliases::InvHessian<Frame>;
  static void function(gsl::not_null<aliases::InvHessian<Frame>*> inv_hess,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags = tmpl::list<ThetaPhi<Frame>>;
};
/// @}

/// @{
/// (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the center to each
/// point of the surface.
template <typename Frame>
struct Radius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct RadiusCompute : Radius<Frame>, db::ComputeTag {
  using base = Radius<Frame>;
  using return_type = Scalar<DataVector>;
  static void function(gsl::not_null<Scalar<DataVector>*> radius,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>>;
};
/// @}

/// @{
/// `CartesianCoords(i)` is \f$x_{\rm surf}^i\f$,
/// the vector of \f$(x,y,z)\f$ coordinates of each point
/// on the surface.
template <typename Frame>
struct CartesianCoords : db::SimpleTag {
  using type = aliases::Vector<Frame>;
};

template <typename Frame>
struct CartesianCoordsCompute : CartesianCoords<Frame>, db::ComputeTag {
  using base = CartesianCoords<Frame>;
  using return_type = aliases::Vector<Frame>;
  static void function(gsl::not_null<aliases::Vector<Frame>*> coords,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& radius,
                       const aliases::OneForm<Frame>& r_hat) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, Rhat<Frame>>;
};
/// @}

/// @{
/// `DxRadius(i)` is \f$\partial r_{\rm surf}/\partial x^i\f$.  Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct DxRadius : db::SimpleTag {
  using type = aliases::OneForm<Frame>;
};

template <typename Frame>
struct DxRadiusCompute : DxRadius<Frame>, db::ComputeTag {
  using base = DxRadius<Frame>;
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> dx_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& radius,
                       const aliases::InvJacobian<Frame>& inv_jac) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, InvJacobian<Frame>>;
};
/// @}

/// @{
/// `D2xRadius(i,j)` is
/// \f$\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$. Here
/// \f$r_{\rm surf}=r_{\rm surf}(\theta,\phi)\f$ is the function
/// describing the surface, which is considered a function of
/// Cartesian coordinates
/// \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$
/// for this operation.
template <typename Frame>
struct D2xRadius : db::SimpleTag {
  using type = aliases::SecondDeriv<Frame>;
};

template <typename Frame>
struct D2xRadiusCompute : D2xRadius<Frame>, db::ComputeTag {
  using base = D2xRadius<Frame>;
  using return_type = aliases::SecondDeriv<Frame>;
  static void function(gsl::not_null<aliases::SecondDeriv<Frame>*> d2x_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& radius,
                       const aliases::InvJacobian<Frame>& inv_jac,
                       const aliases::InvHessian<Frame>& inv_hess) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   InvJacobian<Frame>, InvHessian<Frame>>;
};
/// @}

/// @{
/// \f$\nabla^2 r_{\rm surf}\f$, the flat Laplacian of the surface.
/// This is \f$\eta^{ij}\partial^2 r_{\rm surf}/\partial x^i\partial x^j\f$,
/// where \f$r_{\rm surf}=r_{\rm surf}(\theta(x,y,z),\phi(x,y,z))\f$.
template <typename Frame>
struct LaplacianRadius : db::SimpleTag {
  using type = DataVector;
};

template <typename Frame>
struct LaplacianRadiusCompute : LaplacianRadius<Frame>, db::ComputeTag {
  using base = LaplacianRadius<Frame>;
  using return_type = DataVector;
  static void function(gsl::not_null<DataVector*> lap_radius,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& radius,
                       const aliases::ThetaPhi<Frame>& theta_phi) noexcept;
  using argument_tags =
      tmpl::list<Strahlkorper<Frame>, Radius<Frame>, ThetaPhi<Frame>>;
};
/// @}

/// @{
/// `NormalOneForm(i)` is \f$s_i\f$, the (unnormalized) normal one-form
/// to the surface, expressed in Cartesian components.
/// This is computed by \f$x_i/r-\partial r_{\rm surf}/\partial x^i\f$,
/// where \f$x_i/r\f$ is `Rhat` and
/// \f$\partial r_{\rm surf}/\partial x^i\f$ is `DxRadius`.
/// See Eq. (8) of \cite Baumgarte1996hh.
/// Note on the word "normal": \f$s_i\f$ points in the correct direction
/// (it is "normal" to the surface), but it does not have unit length
/// (it is not "normalized"; normalization requires a metric).
template <typename Frame>
struct NormalOneForm : db::SimpleTag {
  using type = aliases::OneForm<Frame>;
};

template <typename Frame>
struct NormalOneFormCompute : NormalOneForm<Frame>, db::ComputeTag {
  using base = NormalOneForm<Frame>;
  using return_type = aliases::OneForm<Frame>;
  static void function(gsl::not_null<aliases::OneForm<Frame>*> one_form,
                       const aliases::OneForm<Frame>& dx_radius,
                       const aliases::OneForm<Frame>& r_hat) noexcept;
  using argument_tags = tmpl::list<DxRadius<Frame>, Rhat<Frame>>;
};
/// @}

/// The OneOverOneFormMagnitude is the reciprocal of the magnitude of the
/// one-form perpendicular to the horizon
struct OneOverOneFormMagnitude : db::SimpleTag {
  using type = DataVector;
};

/// Computes the reciprocal of the magnitude of the one form perpendicular to
/// the horizon
template <size_t Dim, typename Frame, typename DataType>
struct OneOverOneFormMagnitudeCompute : db::ComputeTag,
                                        OneOverOneFormMagnitude {
  using base = OneOverOneFormMagnitude;
  using return_type = DataVector;
  static void function(
      const gsl::not_null<DataVector*> one_over_magnitude,
      const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
      const tnsr::i<DataType, Dim, Frame>& normal_one_form) noexcept {
    *one_over_magnitude =
        1.0 / get(magnitude(normal_one_form, inverse_spatial_metric));
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                 NormalOneForm<Frame>>;
};

/// The unit normal one-form \f$s_j\f$ to the horizon.
template <typename Frame>
struct UnitNormalOneForm : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame>;
};
/// Computes the unit one-form perpendicular to the horizon
template <typename Frame>
struct UnitNormalOneFormCompute : UnitNormalOneForm<Frame>, db::ComputeTag {
  using base = UnitNormalOneForm<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::i<DataVector, 3, Frame>*>,
      const tnsr::i<DataVector, 3, Frame>&, const DataVector&) noexcept>(
      &::StrahlkorperGr::unit_normal_one_form<Frame>);
  using argument_tags = tmpl::list<StrahlkorperTags::NormalOneForm<Frame>,
                                   OneOverOneFormMagnitude>;
  using return_type = tnsr::i<DataVector, 3, Frame>;
};

/// UnitNormalVector is defined as \f$S^i = \gamma^{ij} S_j\f$,
/// where \f$S_j\f$ is the unit normal one form
/// and \f$\gamma^{ij}\f$ is the inverse spatial metric.
template <typename Frame>
struct UnitNormalVector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame>;
};
/// Computes the UnitNormalVector perpendicular to the horizon.
template <typename Frame>
struct UnitNormalVectorCompute : UnitNormalVector<Frame>, db::ComputeTag {
  using base = UnitNormalVector<Frame>;
  static void function(gsl::not_null<tnsr::I<DataVector, 3, Frame>*> result,
      const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
      const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form) noexcept {
    raise_or_lower_index(result, unit_normal_one_form, inverse_spatial_metric);
  }
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame, DataVector>,
                 UnitNormalOneForm<Frame>>;
  using return_type = tnsr::I<DataVector, 3, Frame>;
};

/// The 3-covariant gradient \f$D_i S_j\f$ of a Strahlkorper's normal
template <typename Frame>
struct GradUnitNormalOneForm : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
};
/// Computes 3-covariant gradient \f$D_i S_j\f$ of a Strahlkorper's normal
template <typename Frame>
struct GradUnitNormalOneFormCompute : GradUnitNormalOneForm<Frame>,
                                      db::ComputeTag {
  using base = GradUnitNormalOneForm<Frame>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*>,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&, const DataVector&,
      const tnsr::Ijj<DataVector, 3, Frame>&) noexcept>(
      &StrahlkorperGr::grad_unit_normal_one_form<Frame>);
  using argument_tags =
      tmpl::list<Rhat<Frame>, Radius<Frame>, UnitNormalOneForm<Frame>,
                 D2xRadius<Frame>, OneOverOneFormMagnitude,
                 gr::Tags::SpatialChristoffelSecondKind<3, Frame, DataVector>>;
  using return_type = tnsr::ii<DataVector, 3, Frame>;
};

/// Extrinsic curvature of a 2D Strahlkorper embedded in a 3D space.
template <typename Frame>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataVector, 3, Frame>;
};
/// Calculates the Extrinsic curvature of a 2D Strahlkorper embedded in a 3D
/// space.
template <typename Frame>
struct ExtrinsicCurvatureCompute : ExtrinsicCurvature<Frame>, db::ComputeTag {
  using base = ExtrinsicCurvature<Frame>;
  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*>,
                           const tnsr::ii<DataVector, 3, Frame>&,
                           const tnsr::i<DataVector, 3, Frame>&,
                           const tnsr::I<DataVector, 3, Frame>&) noexcept>(
          &StrahlkorperGr::extrinsic_curvature<Frame>);
  using argument_tags =
      tmpl::list<GradUnitNormalOneForm<Frame>, UnitNormalOneForm<Frame>,
                 UnitNormalVector<Frame>>;
  using return_type = tnsr::ii<DataVector, 3, Frame>;
};

/// Ricci scalar is the two-dimensional intrinsic Ricci scalar curvature
/// of a Strahlkorper
struct RicciScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Computes the two-dimensional intrinsic Ricci scalar of a Strahlkorper
template <typename Frame>
struct RicciScalarCompute : RicciScalar, db::ComputeTag {
  using base = RicciScalar;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::I<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::II<DataVector, 3, Frame>&) noexcept>(
      &StrahlkorperGr::ricci_scalar<Frame>);
  using argument_tags =
      tmpl::list<gr::Tags::SpatialRicci<3, Frame, DataVector>,
                 UnitNormalVector<Frame>, ExtrinsicCurvature<Frame>,
                 gr::Tags::InverseSpatialMetric<3, Frame, DataVector>>;
  using return_type = Scalar<DataVector>;
};

/// The pointwise maximum of the Strahlkorper's intrinsic Ricci scalar
/// curvature.
struct MaxRicciScalar : db::SimpleTag {
  using type = double;
};

/// Computes the pointwise maximum of the Strahlkorper's intrinsic Ricci
/// scalar curvature.
struct MaxRicciScalarCompute : MaxRicciScalar, db::ComputeTag {
  using base = MaxRicciScalar;
  using return_type = double;
  static void function(const gsl::not_null<double*> max_ricci_scalar,
                       const Scalar<DataVector>& ricci_scalar) noexcept {
    *max_ricci_scalar = max(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

/// The pointwise minimum of the Strahlkorper’s intrinsic Ricci scalar
/// curvature.
struct MinRicciScalar : db::SimpleTag {
  using type = double;
};

/// Computes the pointwise minimum of the Strahlkorper’s intrinsic Ricci
/// scalar curvature.
struct MinRicciScalarCompute : MinRicciScalar, db::ComputeTag {
  using base = MinRicciScalar;
  using return_type = double;
  static void function(const gsl::not_null<double*> min_ricci_scalar,
                       const Scalar<DataVector>& ricci_scalar) noexcept {
    *min_ricci_scalar = min(get(ricci_scalar));
  }
  using argument_tags = tmpl::list<RicciScalar>;
};

/// @{
/// `Tangents(i,j)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
/// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the
/// surface (i.e. `CartesianCoords`) and are considered functions of
/// \f$(\theta,\phi)\f$.
///
/// \f$\partial/\partial q^0\f$ means
/// \f$\partial/\partial\theta\f$; and \f$\partial/\partial q^1\f$
/// means \f$\csc\theta\,\,\partial/\partial\phi\f$.  Note that the
/// vectors `Tangents(i,0)` and `Tangents(i,1)` are orthogonal to the
/// `NormalOneForm` \f$s_i\f$, i.e.
/// \f$s_i \partial x_{\rm surf}^i/\partial q^j = 0\f$; this statement
/// is independent of a metric.  Also, `Tangents(i,0)` and
/// `Tangents(i,1)` are not necessarily orthogonal to each other,
/// since orthogonality between 2 vectors (as opposed to a vector and
/// a one-form) is metric-dependent.
template <typename Frame>
struct Tangents : db::SimpleTag {
  using type = aliases::Jacobian<Frame>;
};

template <typename Frame>
struct TangentsCompute : Tangents<Frame>, db::ComputeTag {
  using base = Tangents<Frame>;
  using return_type = aliases::Jacobian<Frame>;
  static void function(gsl::not_null<aliases::Jacobian<Frame>*> tangents,
                       const ::Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& radius,
                       const aliases::OneForm<Frame>& r_hat,
                       const aliases::Jacobian<Frame>& jac) noexcept;
  using argument_tags = tmpl::list<Strahlkorper<Frame>, Radius<Frame>,
                                   Rhat<Frame>, Jacobian<Frame>>;
};
/// @}

/// @{
/// Computes the Euclidean area element on a Strahlkorper.
/// Useful for flat space integrals.
template <typename Frame>
struct EuclideanAreaElement : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct EuclideanAreaElementCompute : EuclideanAreaElement<Frame>,
                                     db::ComputeTag {
  using base = EuclideanAreaElement<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const StrahlkorperTags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&) noexcept>(
      &::StrahlkorperGr::euclidean_area_element<Frame>);
  using argument_tags = tmpl::list<
      StrahlkorperTags::Jacobian<Frame>, StrahlkorperTags::NormalOneForm<Frame>,
      StrahlkorperTags::Radius<Frame>, StrahlkorperTags::Rhat<Frame>>;
};
/// @}

/// @{
/// Computes the flat-space integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegral : db::SimpleTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralCompute
    : EuclideanSurfaceIntegral<IntegrandTag, Frame>,
      db::ComputeTag {
  using base = EuclideanSurfaceIntegral<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& euclidean_area_element,
                       const Scalar<DataVector>& integrand,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    *surface_integral = ::StrahlkorperGr::surface_integral_of_scalar<Frame>(
        euclidean_area_element, integrand, strahlkorper);
  }
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};
/// @}

/// @{
/// Computes the Euclidean-space integral of a vector over a
/// Strahlkorper, \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2 S\f$,
/// where \f$s_i\f$ is the Strahlkorper surface unit normal and
/// \f$\delta^{ij}\f$ is the Kronecker delta.  Note that \f$s_i\f$ is
/// not assumed to be normalized; the denominator of the integrand
/// effectively normalizes it using the Euclidean metric.
template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVector : db::SimpleTag {
  static std::string name() noexcept {
    return "EuclideanSurfaceIntegralVector(" + db::tag_name<IntegrandTag>() +
           ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct EuclideanSurfaceIntegralVectorCompute
    : EuclideanSurfaceIntegralVector<IntegrandTag, Frame>,
      db::ComputeTag {
  using base = EuclideanSurfaceIntegralVector<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& euclidean_area_element,
                       const tnsr::I<DataVector, 3, Frame>& integrand,
                       const tnsr::i<DataVector, 3, Frame>& normal_one_form,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    *surface_integral =
        ::StrahlkorperGr::euclidean_surface_integral_of_vector<Frame>(
            euclidean_area_element, integrand, normal_one_form, strahlkorper);
  }
  using argument_tags = tmpl::list<EuclideanAreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::NormalOneForm<Frame>,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};
/// @}

template <typename Frame>
using items_tags = tmpl::list<Strahlkorper<Frame>>;

template <typename Frame>
using compute_items_tags =
    tmpl::list<ThetaPhiCompute<Frame>, RhatCompute<Frame>,
               JacobianCompute<Frame>, InvJacobianCompute<Frame>,
               InvHessianCompute<Frame>, RadiusCompute<Frame>,
               CartesianCoordsCompute<Frame>, DxRadiusCompute<Frame>,
               D2xRadiusCompute<Frame>, LaplacianRadiusCompute<Frame>,
               NormalOneFormCompute<Frame>, TangentsCompute<Frame>>;

}  // namespace StrahlkorperTags

namespace StrahlkorperGr {
/// \ingroup SurfacesGroup
/// Holds tags and ComputeItems associated with a `::Strahlkorper` that
/// also need a metric.
namespace Tags {

/// @{
/// Computes the area element on a Strahlkorper. Useful for integrals.
template <typename Frame>
struct AreaElement : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Frame>
struct AreaElementCompute : AreaElement<Frame>, db::ComputeTag {
  using base = AreaElement<Frame>;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const StrahlkorperTags::aliases::Jacobian<Frame>&,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&) noexcept>(&area_element<Frame>);
  using argument_tags = tmpl::list<
      gr::Tags::SpatialMetric<3, Frame>, StrahlkorperTags::Jacobian<Frame>,
      StrahlkorperTags::NormalOneForm<Frame>, StrahlkorperTags::Radius<Frame>,
      StrahlkorperTags::Rhat<Frame>>;
};
/// @}

/// @{
/// Computes the integral of a scalar over a Strahlkorper.
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral : db::SimpleTag {
  static std::string name() noexcept {
    return "SurfaceIntegral(" + db::tag_name<IntegrandTag>() + ")";
  }
  using type = double;
};

template <typename IntegrandTag, typename Frame>
struct SurfaceIntegralCompute : SurfaceIntegral<IntegrandTag, Frame>,
                                db::ComputeTag {
  using base = SurfaceIntegral<IntegrandTag, Frame>;
  using return_type = double;
  static void function(const gsl::not_null<double*> surface_integral,
                       const Scalar<DataVector>& area_element,
                       const Scalar<DataVector>& integrand,
                       const ::Strahlkorper<Frame>& strahlkorper) noexcept {
    *surface_integral = ::StrahlkorperGr::surface_integral_of_scalar<Frame>(
        area_element, integrand, strahlkorper);
  }
  using argument_tags = tmpl::list<AreaElement<Frame>, IntegrandTag,
                                   StrahlkorperTags::Strahlkorper<Frame>>;
};
/// @}

/// Tag representing the surface area of a Strahlkorper
struct Area : db::SimpleTag {
  using type = double;
};

/// Computes the surface area of a Strahlkorer, \f$A = \oint_S dA\f$ given an
/// AreaElement \f$dA\f$ and a Strahlkorper \f$S\f$.
template <typename Frame>
struct AreaCompute : Area, db::ComputeTag {
  using base = Area;
  using return_type = double;
  static void function(const gsl::not_null<double*> result,
                       const Strahlkorper<Frame>& strahlkorper,
                       const Scalar<DataVector>& area_element) noexcept {
    *result = strahlkorper.ylm_spherepack().definite_integral(
        get(area_element).data());
  }
  using argument_tags =
      tmpl::list<StrahlkorperTags::Strahlkorper<Frame>, AreaElement<Frame>>;
};

/// The Irreducible (areal) mass of an apparent horizon
struct IrreducibleMass : db::SimpleTag {
  using type = double;
};

/// Computes the Irreducible mass of an apparent horizon from its area
template <typename Frame>
struct IrreducibleMassCompute : IrreducibleMass, db::ComputeTag {
  using base = IrreducibleMass;
  using return_type = double;
  static void function(const gsl::not_null<double*> result,
                       const double area) noexcept {
    *result = ::StrahlkorperGr::irreducible_mass(area);
  }

  using argument_tags = tmpl::list<Area>;
};

/// The spin function is proportional to the imaginary part of the
/// Strahlkorper’s complex scalar curvature.

struct SpinFunction : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// Calculates the spin function which is proportional to the imaginary part of
/// the Strahlkorper’s complex scalar curvature.
template <typename Frame>
struct SpinFunctionCompute : SpinFunction, db::ComputeTag {
  using base = SpinFunction;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>,
      const StrahlkorperTags::aliases::Jacobian<Frame>&,
      const Strahlkorper<Frame>&, const tnsr::I<DataVector, 3, Frame>&,
      const Scalar<DataVector>&,
      const tnsr::ii<DataVector, 3, Frame>&) noexcept>(
      &StrahlkorperGr::spin_function<Frame>);
  using argument_tags =
      tmpl::list<StrahlkorperTags::Tangents<Frame>,
                 StrahlkorperTags::Strahlkorper<Frame>,
                 StrahlkorperTags::UnitNormalVector<Frame>, AreaElement<Frame>,
                 gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>>;
  using return_type = Scalar<DataVector>;
};

/// The approximate-Killing-Vector quasilocal spin magnitude of a Strahlkorper
/// (see Sec. 2.2 of \cite Boyle2019kee and references therein).
struct DimensionfulSpinMagnitude : db::SimpleTag {
  using type = double;
};

/// Computes the approximate-Killing-Vector quasilocal spin magnitude of a
/// Strahlkorper
template <typename Frame>
struct DimensionfulSpinMagnitudeCompute : DimensionfulSpinMagnitude,
                                          db::ComputeTag {
  using base = DimensionfulSpinMagnitude;
  using return_type = double;
  static constexpr auto function =
      &StrahlkorperGr::dimensionful_spin_magnitude<Frame>;
  using argument_tags =
      tmpl::list<StrahlkorperTags::RicciScalar, SpinFunction,
                 gr::Tags::SpatialMetric<3, Frame>,
                 StrahlkorperTags::Tangents<Frame>,
                 StrahlkorperTags::Strahlkorper<Frame>, AreaElement<Frame>>;
};

/// The dimensionful spin angular momentum vector.
template <typename Frame>
struct DimensionfulSpinVector : db::SimpleTag {
  using type = std::array<double, 3>;
};

/// Computes the dimensionful spin angular momentum vector.
template <typename Frame>
struct DimensionfulSpinVectorCompute : DimensionfulSpinVector<Frame>,
                                       db::ComputeTag {
  using base = DimensionfulSpinVector<Frame>;
  using return_type = std::array<double, 3>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<std::array<double, 3>*>, double,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, 3, Frame>&, const Scalar<DataVector>&,
      const Scalar<DataVector>&, const Strahlkorper<Frame>&) noexcept>(
      &StrahlkorperGr::spin_vector);
  using argument_tags =
      tmpl::list<DimensionfulSpinMagnitude, AreaElement<Frame>,
                 StrahlkorperTags::Radius<Frame>, StrahlkorperTags::Rhat<Frame>,
                 StrahlkorperTags::RicciScalar, SpinFunction,
                 StrahlkorperTags::Strahlkorper<Frame>>;
};

/// The Christodoulou mass, which is a function of the dimensionful spin
/// angular momentum and the irreducible mass of a Strahlkorper.
struct ChristodoulouMass : db::SimpleTag {
  using type = double;
};

/// Computes the Christodoulou mass from the dimensionful spin angular momentum
/// and the irreducible mass of a Strahlkorper.
template <typename Frame>
struct ChristodoulouMassCompute : ChristodoulouMass, db::ComputeTag {
  using base = ChristodoulouMass;
  using return_type = double;
  static constexpr auto function = &StrahlkorperGr::christodoulou_mass;
  using argument_tags = tmpl::list<DimensionfulSpinMagnitude, IrreducibleMass>;
};

}  // namespace Tags
}  // namespace StrahlkorperGr
