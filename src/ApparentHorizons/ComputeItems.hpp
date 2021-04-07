// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
/// \endcond

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Transform.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
namespace Tags {
// @{
/// These ComputeItems are different from those used in
/// GeneralizedHarmonic evolution because these live only on the
/// intrp::Actions::ApparentHorizon DataBox, not in the volume
/// DataBox.  And these ComputeItems can do fewer allocations than the
/// volume ones, because (for example) Lapse, SpaceTimeNormalVector,
/// etc.  can be inlined instead of being allocated as a separate
/// ComputeItem.
template <size_t Dim, typename Frame>
struct InverseSpatialMetricCompute : gr::Tags::InverseSpatialMetric<Dim, Frame>,
                                     db::ComputeTag {
  using return_type = tnsr::II<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::II<DataVector, Dim, Frame>*> result,
      const tnsr::aa<DataVector, Dim, Frame>& psi) noexcept {
    destructive_resize_components(result, psi.begin()->size());
    *result = determinant_and_inverse(gr::spatial_metric(psi)).second;
  };
  using argument_tags = tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame>>;
  using base = gr::Tags::InverseSpatialMetric<Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct ExtrinsicCurvatureCompute : gr::Tags::ExtrinsicCurvature<Dim, Frame>,
                                   db::ComputeTag {
  using return_type = tnsr::ii<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::ii<DataVector, Dim, Frame>*> result,
      const tnsr::aa<DataVector, Dim, Frame>& psi,
      const tnsr::aa<DataVector, Dim, Frame>& pi,
      const tnsr::iaa<DataVector, Dim, Frame>& phi,
      const tnsr::II<DataVector, Dim, Frame>& inv_g) noexcept {
    const auto shift = gr::shift(psi, inv_g);
    destructive_resize_components(result, psi.begin()->size());
    GeneralizedHarmonic::extrinsic_curvature(
        result, gr::spacetime_normal_vector(gr::lapse(shift, psi), shift), pi,
        phi);
  }
  using argument_tags = tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame>,
                                   GeneralizedHarmonic::Tags::Pi<Dim, Frame>,
                                   GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
                                   gr::Tags::InverseSpatialMetric<Dim, Frame>>;
  using base = gr::Tags::ExtrinsicCurvature<Dim, Frame>;
};

template <size_t Dim, typename SrcFrame, typename DestFrame>
struct ExtrinsicCurvatureTransform
    : gr::Tags::ExtrinsicCurvature<Dim, DestFrame>,
      db::ComputeTag {
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataVector, Dim, DestFrame>*>,
      const tnsr::ii<DataVector, Dim, SrcFrame>& src,
      const Jacobian<DataVector, Dim, DestFrame, SrcFrame>&) noexcept>(
      &transform::to_different_frame<Dim, SrcFrame, DestFrame>);

  using return_type = tnsr::ii<DataVector, Dim, DestFrame>;

  using argument_tags =
      tmpl::list<gr::Tags::ExtrinsicCurvature<Dim, SrcFrame>,
                 transform::Tags::Jacobian<Dim, DestFrame, SrcFrame>>;
  using base = gr::Tags::ExtrinsicCurvature<Dim, DestFrame>;
};

template <size_t Dim, typename SrcFrame, typename DestFrame>
struct SpatialMetricTransform
    : gr::Tags::SpatialMetric<Dim, DestFrame>,
      db::ComputeTag {
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataVector, Dim, DestFrame>*>,
      const tnsr::ii<DataVector, Dim, SrcFrame>& src,
      const Jacobian<DataVector, Dim, DestFrame, SrcFrame>&) noexcept>(
      &transform::to_different_frame<Dim, SrcFrame, DestFrame>);

  using return_type = tnsr::ii<DataVector, Dim, DestFrame>;

  using argument_tags =
      tmpl::list<gr::Tags::SpatialMetric<Dim, SrcFrame>,
                 transform::Tags::Jacobian<Dim, DestFrame, SrcFrame>>;
  using base = gr::Tags::SpatialMetric<Dim, DestFrame>;
};

template <size_t Dim, typename Frame>
struct SpatialChristoffelSecondKindCompute
    : ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>,
      db::ComputeTag {
  using return_type = tnsr::Ijj<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame>*> result,
      const tnsr::iaa<DataVector, Dim, Frame>& phi,
      const tnsr::II<DataVector, Dim, Frame>& inv_g) noexcept {
    destructive_resize_components(result, phi.begin()->size());
    raise_or_lower_first_index(
        result,
        gr::christoffel_first_kind(
            GeneralizedHarmonic::deriv_spatial_metric(phi)),
        inv_g);
  }
  using argument_tags = tmpl::list<GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
                                   gr::Tags::InverseSpatialMetric<Dim, Frame>>;
  using base = ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>;
};

template <size_t Dim, typename Frame>
struct SpatialChristoffelSecondKindFromMetricCompute
    : ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>,
      db::ComputeTag {
  using return_type = tnsr::Ijj<DataVector, Dim, Frame>;
  static void function(
      const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame>*> result,
      const tnsr::ii<DataVector, Dim, Frame>& lower_spatial_metric,
      const tnsr::II<DataVector, Dim, Frame>& upper_spatial_metric,
      const InverseJacobian<DataVector, Dim, ::Frame::Logical, Frame>&
          inverse_jacobian,
      const Mesh<Dim>& mesh) noexcept {
    destructive_resize_components(result, lower_spatial_metric.begin()->size());
    auto logical_deriv_spatial_metric =
        logical_partial_derivative(lower_spatial_metric, mesh);
    tnsr::ijj<DataVector, Dim, Frame> deriv_spatial_metric(
        mesh.number_of_grid_points(), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {  // Symmetry
        for (size_t k = 0; k < Dim; ++k) {
          for (size_t p = 0; p < Dim; ++p) {
            deriv_spatial_metric.get(k, i, j) +=
                logical_deriv_spatial_metric.get(p, i, j) *
                inverse_jacobian.get(p, k);
          }
        }
      }
    }
    raise_or_lower_first_index(result,
                               gr::christoffel_first_kind(deriv_spatial_metric),
                               upper_spatial_metric);
  }
  using argument_tags =
      tmpl::list<gr::Tags::SpatialMetric<Dim, Frame>,
                 gr::Tags::InverseSpatialMetric<Dim, Frame>,
                 transform::Tags::InverseJacobian<Dim, ::Frame::Logical, Frame>,
                 domain::Tags::Mesh<Dim>>;
  using base = ::gr::Tags::SpatialChristoffelSecondKind<Dim, Frame>;
};

// }@
}  // namespace Tags
}  // namespace ah
