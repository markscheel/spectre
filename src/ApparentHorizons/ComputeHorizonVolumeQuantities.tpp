// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Transform.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"

namespace detail {

template <typename Tag, typename DestTags, typename TempTags>
typename Tag::type& get_from_target_or_temp(
    const gsl::not_null<Variables<DestTags>*> target_vars,
    const gsl::not_null<TempBuffer<TempTags>*> temp_vars) {
  if constexpr (tmpl::list_contains<DestTags, Tag>::value) {
    return get<Tag>(*target_vars);
  } else {
    return get<Tag>(*temp_vars);
  }
}
}  // namespace detail

namespace ah {

// Single frame case
template <typename DestTagList>
void ComputeHorizonVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<src_tags>& src_vars, const Mesh<3>& /*mesh*/) noexcept {
  static_assert(
      tmpl::list_contains_v<DestTagList,
                            gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>>,
      "Extrinsic curvature must be in the list of dest tags");
  static_assert(
      tmpl::list_contains_v<DestTagList, gr::Tags::SpatialChristoffelSecondKind<
                                             3, Frame::Inertial>>,
      "Christoffel 2nd kind must be in the list of dest tags");

  const auto& psi =
      get<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(src_vars);
  const auto& pi =
      get<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(src_vars);
  const auto& phi =
      get<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(src_vars);

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using metric_tag = gr::Tags::SpatialMetric<3, Frame::Inertial>;
  using inv_metric_tag = gr::Tags::InverseSpatialMetric<3, Frame::Inertial>;
  using lapse_tag = ::Tags::TempScalar<0, DataVector>;
  using shift_tag = ::Tags::TempI<1, 3, Frame::Inertial, DataVector>;
  using spacetime_normal_vector_tag =
      ::Tags::TempA<2, 3, Frame::Inertial, DataVector>;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using full_temp_tags_list =
      tmpl::list<metric_tag, inv_metric_tag, lapse_tag, shift_tag,
                 spacetime_normal_vector_tag>;

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, tmpl::list<DestTagList>>;
  TempBuffer<temp_tags_list> buffer(get<0, 0>(psi).size());

  auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>>(*target_vars);
  auto& spatial_christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
          *target_vars);

  // These are always temporaries.
  auto& lapse = get<lapse_tag>(buffer);
  auto& shift = get<shift_tag>(buffer);
  auto& spacetime_normal_vector = get<spacetime_normal_vector_tag>(buffer);

  // These may or may not be temporaries
  auto& metric = detail::get_from_target_or_temp<metric_tag>(
      target_vars, make_not_null(&buffer));
  auto& inv_metric = detail::get_from_target_or_temp<inv_metric_tag>(
      target_vars, make_not_null(&buffer));

  // Actual computation starts here

  gr::spatial_metric(make_not_null(&metric), psi);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse), make_not_null(&inv_metric),
                          metric);
  gr::shift(make_not_null(&shift), psi, inv_metric);
  gr::lapse(make_not_null(&lapse), shift, psi);
  gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                              shift);
  GeneralizedHarmonic::extrinsic_curvature(make_not_null(&extrinsic_curvature),
                                           spacetime_normal_vector, pi, phi);
  GeneralizedHarmonic::christoffel_second_kind(
      make_not_null(&spatial_christoffel_second_kind), phi, inv_metric);
}

template <typename DestTagList, typename TargetFrame>
void ComputeHorizonVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<src_tags>& src_vars, const Mesh<3>& mesh,
    const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>& jacobian,
    const InverseJacobian<DataVector, 3, Frame::Logical, TargetFrame>&
        inverse_jacobian) noexcept {
  static_assert(
      tmpl::list_contains_v<DestTagList,
                            gr::Tags::ExtrinsicCurvature<3, TargetFrame>>,
      "Extrinsic curvature must be in the list of dest tags");
  static_assert(
      tmpl::list_contains_v<
          DestTagList, gr::Tags::SpatialChristoffelSecondKind<3, TargetFrame>>,
      "Christoffel 2nd kind must be in the list of dest tags");

  const auto& psi =
      get<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(src_vars);
  const auto& pi =
      get<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(src_vars);
  const auto& phi =
      get<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(src_vars);

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using metric_tag = gr::Tags::SpatialMetric<3, TargetFrame>;
  using inv_metric_tag = gr::Tags::InverseSpatialMetric<3, TargetFrame>;
  using lapse_tag = ::Tags::TempScalar<0, DataVector>;
  using shift_tag = ::Tags::TempI<1, 3, Frame::Inertial, DataVector>;
  using spacetime_normal_vector_tag =
      ::Tags::TempA<2, 3, Frame::Inertial, DataVector>;

  // Additional temporary tags used for multiple frames
  using inertial_metric_tag = ::Tags::Tempii<3, 3, Frame::Inertial, DataVector>;
  using inertial_inv_metric_tag =
      ::Tags::TempII<4, 3, Frame::Inertial, DataVector>;
  using inertial_ex_curvature_tag =
      ::Tags::Tempii<5, 3, Frame::Inertial, DataVector>;
  using logical_deriv_metric_tag = ::Tags::TempTensor<
      6, Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                index_list<SpatialIndex<3, UpLo::Lo, Frame::Logical>,
                           SpatialIndex<3, UpLo::Lo, TargetFrame>,
                           SpatialIndex<3, UpLo::Lo, TargetFrame>>>>;
  using deriv_metric_tag = ::Tags::Tempijj<7, 3, TargetFrame, DataVector>;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using full_temp_tags_list =
      tmpl::list<metric_tag, inv_metric_tag, lapse_tag, shift_tag,
                 spacetime_normal_vector_tag, inertial_metric_tag,
                 inertial_inv_metric_tag, inertial_ex_curvature_tag,
                 logical_deriv_metric_tag, deriv_metric_tag>;

  auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, TargetFrame>>(*target_vars);
  auto& spatial_christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<3, TargetFrame>>(*target_vars);

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, tmpl::list<DestTagList>>;
  TempBuffer<temp_tags_list> buffer(get<0, 0>(psi).size());

  // These are always temporaries.
  auto& lapse = get<lapse_tag>(buffer);
  auto& shift = get<shift_tag>(buffer);
  auto& spacetime_normal_vector = get<spacetime_normal_vector_tag>(buffer);
  auto& inertial_metric = get<inertial_metric_tag>(buffer);
  auto& inertial_inv_metric = get<inertial_inv_metric_tag>(buffer);
  auto& inertial_ex_curvature = get<inertial_ex_curvature_tag>(buffer);
  auto& logical_deriv_metric = get<logical_deriv_metric_tag>(buffer);
  auto& deriv_metric = get<deriv_metric_tag>(buffer);

  // These may or may not be temporaries
  auto& metric = detail::get_from_target_or_temp<metric_tag>(
      target_vars, make_not_null(&buffer));
  auto& inv_metric = detail::get_from_target_or_temp<inv_metric_tag>(
      target_vars, make_not_null(&buffer));

  // Actual computation starts here

  gr::spatial_metric(make_not_null(&inertial_metric), psi);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inertial_inv_metric), inertial_metric);

  // Compute inertial extrinsic curvature
  gr::shift(make_not_null(&shift), psi, inertial_inv_metric);
  gr::lapse(make_not_null(&lapse), shift, psi);
  gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                              shift);
  GeneralizedHarmonic::extrinsic_curvature(
      make_not_null(&inertial_ex_curvature), spacetime_normal_vector, pi, phi);

  // Transform spatial metric and extrinsic curvature
  transform::to_different_frame(make_not_null(&metric), inertial_metric,
                                jacobian);
  transform::to_different_frame(make_not_null(&extrinsic_curvature),
                                inertial_ex_curvature, jacobian);

  // Invert transformed 3-metric.
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse), make_not_null(&inv_metric),
                          metric);

  // Differentiate 3-metric.
  logical_partial_derivative(make_not_null(&logical_deriv_metric), metric,
                             mesh);
  transform::first_index_to_different_frame(
      make_not_null(&deriv_metric), logical_deriv_metric, inverse_jacobian);
  gr::christoffel_second_kind(make_not_null(&spatial_christoffel_second_kind),
                              deriv_metric, inv_metric);
}

}  // namespace ah
