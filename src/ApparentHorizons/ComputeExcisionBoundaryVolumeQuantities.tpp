// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.hpp"

#pragma once

#include <cstdint>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {

namespace detail {
template <typename Tag>
using tensor_index_list = typename Tag::type::index_list;

template <typename Tag, typename Frame>
using all_indices_in_frame_impl =
    TensorMetafunctions::all_indices_in_frame<typename Tag::type, Frame>;
};


/// Returns true if all of the tensors in TagList have
/// indices only in the given frame or are scalars.
template <typename TagList, typename Frame>
constexpr bool all_indices_in_frame_v =
    tmpl::any<TagList, tmpl::bind<detail::all_indices_in_frame_impl, tmpl::_1,
                                  Frame>>::value;

/// Time-independent map case.
///
/// Typically in this case everything is in Frame::Inertial.  However,
/// it turns out that for some cases with time-independent maps, it is
/// useful to define tags for multiple frames, like Frame::Inertial
/// and Frame::Grid, with the understanding that these frames are really
/// identical but use different tags.
/// So here we allow the Tensors in DestTagList to be in a different frame
/// (as determined by the Frame tags) than the Inertial frame, but with
/// the understanding that these frames are really the same and that no
/// frame transformations actually occur.
template <typename SrcTagList, typename DestTagList>
void ComputeExcisionBoundaryVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<SrcTagList>& src_vars, const Mesh<3>& /*mesh*/) {
  // dest_tag_frame is the Frame of the first non-scalar Tensor
  // in DestTagList.  (We are assuming that DestTagList contains more
  // than only scalars).
  // Note that below we static_assert that all tensors in DestTagList
  // have the same frame.
  using dest_tag_frame = typename tmpl::front<tmpl::flatten<tmpl::transform<
      DestTagList, tmpl::bind<detail::tensor_index_list, tmpl::_1>>>>::Frame;
  static_assert(all_indices_in_frame_v<DestTagList, dest_tag_frame>,
                "All dest tags must be in the same frame if "
                "the maps are time-independent");

  static_assert(
      std::is_same_v<tmpl::list_difference<SrcTagList, allowed_src_tags>,
                     tmpl::list<>>,
      "Found a src tag that is not allowed");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_src_tags, SrcTagList>,
                     tmpl::list<>>,
      "A required src tag is missing");

  static_assert(
      std::is_same_v<
          tmpl::list_difference<DestTagList, allowed_dest_tags<dest_tag_frame>>,
          tmpl::list<>>,
      "Found a dest tag that is not allowed");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_dest_tags<dest_tag_frame>,
                                           DestTagList>,
                     tmpl::list<>>,
      "A required dest tag is missing");

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using target_spacetime_metric_tag =
      gr::Tags::SpacetimeMetric<3, dest_tag_frame>;
  using target_spatial_metric_tag = gr::Tags::SpatialMetric<3, dest_tag_frame>;
  using target_inv_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<3, dest_tag_frame>;
  using target_lapse_tag = gr::Tags::Lapse<DataVector>;
  using target_shift_tag = gr::Tags::Shift<3, dest_tag_frame>;
  using target_constraint_gamma1_tag =
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using full_temp_tags_list =
      tmpl::list<target_spacetime_metric_tag, target_spatial_metric_tag,
                 target_inv_spatial_metric_tag, target_lapse_tag,
                 target_shift_tag, target_constraint_gamma1_tag>;

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, DestTagList>;
  TempBuffer<temp_tags_list> buffer(src_vars.number_of_grid_points());

  // These may or may not be temporaries
  auto& lapse = *(get<target_lapse_tag>(target_vars, make_not_null(&buffer)));
  auto& shift = *(get<target_shift_tag>(target_vars, make_not_null(&buffer)));
  auto& spatial_metric =
      *(get<target_spatial_metric_tag>(target_vars, make_not_null(&buffer)));
  auto& spacetime_metric =
      *(get<target_spacetime_metric_tag>(target_vars, make_not_null(&buffer)));
  auto& inv_spatial_metric = *(
      get<target_inv_spatial_metric_tag>(target_vars, make_not_null(&buffer)));
  auto& constraint_gamma1 =
      *(get<target_constraint_gamma1_tag>(target_vars, make_not_null(&buffer)));

  // Actual computation starts here

  if constexpr (std::is_same_v<dest_tag_frame, Frame::Inertial>) {
    spacetime_metric =
        get<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(src_vars);
  } else {
    // Tags are different so we need to copy each component separately.
    const auto& src_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(src_vars);
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {  // symmetry
        spacetime_metric.get(a, b) = src_spacetime_metric.get(a, b);
      }
    }
  }

  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inv_spatial_metric),
                          spatial_metric);
  gr::shift(make_not_null(&shift), spacetime_metric, inv_spatial_metric);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);

  // constraint_gamma1 is a scalar so no worrying about frames.
  constraint_gamma1 =
  get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>(src_vars);
}

/// Dual frame case
template <typename SrcTagList, typename DestTagList, typename TargetFrame>
void ComputeExcisionBoundaryVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<SrcTagList>& src_vars, const Mesh<3>& /*mesh*/,
    const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
      jac_target_to_inertial,
    const InverseJacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
      invjac_target_to_inertial,
    const Jacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
      /*jac_logical_to_target*/,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
      /*invjac_logical_to_target*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_mesh_velocity) {
  static_assert(
      std::is_same_v<tmpl::list_difference<SrcTagList, allowed_src_tags>,
                     tmpl::list<>>,
      "Found a src tag that is not allowed");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_src_tags, SrcTagList>,
                     tmpl::list<>>,
      "A required src tag is missing");

  static_assert(
      std::is_same_v<
          tmpl::list_difference<DestTagList, allowed_dest_tags<TargetFrame>>,
          tmpl::list<>>,
      "Found a dest tag that is not allowed");
  static_assert(
      std::is_same_v<
          tmpl::list_difference<required_dest_tags<TargetFrame>, DestTagList>,
          tmpl::list<>>,
      "A required dest tag is missing");

  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(src_vars);

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using spatial_metric_tag = gr::Tags::SpatialMetric<3, TargetFrame>;
  using inv_spatial_metric_tag = gr::Tags::InverseSpatialMetric<3, TargetFrame>;
  using lapse_tag = gr::Tags::Lapse<DataVector>;
  using shift_tag = gr::Tags::Shift<3, TargetFrame>;
  using inertial_shift_tag = gr::Tags::Shift<3, Frame::Inertial>;
  using constraint_gamma1_tag =
    GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1;

  // Additional temporary tags used for multiple frames
  using inertial_spatial_metric_tag =
      gr::Tags::SpatialMetric<3, Frame::Inertial>;
  using inertial_inv_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial>;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using full_temp_tags_list =
      tmpl::list<spatial_metric_tag, inv_spatial_metric_tag,
                 lapse_tag, shift_tag, inertial_shift_tag,
                 inertial_spatial_metric_tag,
                 inertial_inv_spatial_metric_tag, constraint_gamma1_tag>;

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, DestTagList>;
  TempBuffer<temp_tags_list> buffer(get<0, 0>(spacetime_metric).size());

  // These may or may not be temporaries, depending on if they are asked for
  // in target_vars.
  auto& lapse = *(get<lapse_tag>(
      target_vars, make_not_null(&buffer)));
  auto& shift = *(get<shift_tag>(
      target_vars, make_not_null(&buffer)));
  auto& inertial_shift = *(get<inertial_shift_tag>(
      target_vars, make_not_null(&buffer)));
  auto& inertial_spatial_metric = *(get<inertial_spatial_metric_tag>(
      target_vars, make_not_null(&buffer)));
  auto& inertial_inv_spatial_metric =
      *(get<inertial_inv_spatial_metric_tag>(
          target_vars, make_not_null(&buffer)));
  auto& spatial_metric = *(get<spatial_metric_tag>(
      target_vars, make_not_null(&buffer)));
  auto& inv_spatial_metric = *(get<inv_spatial_metric_tag>(
      target_vars, make_not_null(&buffer)));
  auto& constraint_gamma1 = *(get<constraint_gamma1_tag>(
      target_vars, make_not_null(&buffer)));

  // Actual computation starts here

  gr::spatial_metric(make_not_null(&inertial_spatial_metric), spacetime_metric);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inertial_inv_spatial_metric),
                          inertial_spatial_metric);

  // Transform spatial metric
  transform::to_different_frame(make_not_null(&spatial_metric),
                                inertial_spatial_metric,
                                jac_target_to_inertial);

  // Invert transformed 3-metric.
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inv_spatial_metric),
                          spatial_metric);

  // Only the inertial shift is computed at this time, so there is no
  // transformation done.
  gr::shift(make_not_null(&inertial_shift),
                          spacetime_metric,
                          inertial_inv_spatial_metric);
  // We assume the lapse does note transform between the target and
  // inertial frames.
  gr::lapse(make_not_null(&lapse), inertial_shift, spacetime_metric);

  // Transform the shift
  const size_t VolumeDim = 3;
  auto dest = &shift;
  const auto& src = inertial_shift;
  for (size_t i = 0; i < VolumeDim; ++i) {
    dest->get(i) = invjac_target_to_inertial.get(i, 0) *
    (src.get(0) + inertial_mesh_velocity.get(0));
    for (size_t j = 1; j < VolumeDim; ++j) {
          dest->get(i) += invjac_target_to_inertial.get(i, j) *
          (src.get(j) + inertial_mesh_velocity.get(j));
    }
  }
  constraint_gamma1 =
  get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>(src_vars);
}

}  // namespace ah
