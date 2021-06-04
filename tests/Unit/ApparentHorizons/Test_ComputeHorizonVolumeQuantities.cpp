// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace {
template <typename SrcTags, typename DestTags>
void test_compute_horizon_volume_quantities_time_independent() {
  const size_t number_of_grid_points = 8;

  // Create a brick offset from the origin, so a KerrSchild solution
  // doesn't have a singularity or horizon in the domain.
  const domain::creators::Brick domain_creator(
      {{3.1, 3.2, 3.3}}, {{4.1, 4.2, 4.3}}, 0, number_of_grid_points);
  const auto domain = domain_creator.create_domain();
  ASSERT(domain.blocks().size() == 1, "Expected a Domain with one block");

  const auto element_ids = initial_element_ids(
      domain.blocks()[0].id(),
      domain_creator->initial_refinement_levels()[domain.blocks()[0].id()]);
  ASSERT(element_ids.size() == 1, "Expected a Domain with only one element");

  // Set up coordinates
  const Mesh mesh{domain_creator.initial_extents()[element_ids[0].block_id()],
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  ElementMap<3, ::Frame::Inertial> map_logical_to_inertial{
      element_ids[0], domain.blocks()[0].stationary_map().get_clone()};
  const auto analytic_solution_coords =
      map_logical_to_inertial(logical_coordinates(mesh));

  // Set up analytic solution.
  gr::Solutions::KerrSchild solution(1.0, {{0.1, 0.2, 0.3}},
                                     {{0.03, 0.01, 0.02}});
  const auto solution_vars = solution.variables(
      analytic_solution_coords, 0.0,
      typename gr::Solutions::KerrSchild::tags<DataVector, Frame::Inertial>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
  const auto& dt_lapse =
      get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
  const auto& d_lapse = get<typename gr::Solutions::KerrSchild::DerivLapse<
      DataVector, Frame::Inertial>>(solution_vars);
  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(solution_vars);
  const auto& d_shift = get<typename gr::Solutions::KerrSchild::DerivShift<
      DataVector, Frame::Inertial>>(solution_vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
          solution_vars);
  const auto& g = get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
      solution_vars);
  const auto& dt_g =
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          solution_vars);
  const auto& d_g = get<typename gr::Solutions::KerrSchild::DerivSpatialMetric<
      DataVector, Frame::Inertial>>(solution_vars);

  // Fill src vars with analytic solution.
  Variables<SrcTags> src_vars(mesh.number_of_grid_points());
  get<::gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>>(src_vars) =
      gr::spacetime_metric(lapse, shift, g);
  get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(src_vars) =
      GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
  get<::GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(src_vars) =
      GeneralizedHarmonic::pi(
          lapse, dt_lapse, shift, dt_shift, g, dt_g,
          get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
              src_vars));

  if constexpr (tmpl::list_contains_v<
                    SrcTags, Tags::deriv<GeneralizedHarmonic::Tags::Phi<
                                             3, Frame::Inertial>,
                                         tmpl::size_t<3>, Frame::Inertial>>) {
    // Need to compute numerical deriv of Phi.
    // The partial_derivatives function allows differentiating only a
    // subset of a Variables, but in that case, the resulting Variables
    // cannot point into the original Variables, and also the subset must
    // be the tags that occur at the beginning of the Variables.  Because
    // this is only a test, we just create new Variables and copy.

    // vars to be differentiated
    using tags_before_differentiation =
        tmpl::list<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>;
    Variables<tags_before_differentiation> vars_before_differentiation(
        mesh.number_of_grid_points());
    get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
        vars_before_differentiation) =
        get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(src_vars);

    // differentiate
    const auto inv_jacobian =
        map_logical_to_inertial.inv_jacobian(logical_coordinates(mesh));
    using tags_after_differentiation = tmpl::list<
        Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>>;
    const auto vars_after_differentiation =
        partial_derivatives<tags_after_differentiation>(
            vars_before_differentiation, mesh, inv_jacobian);

    // copy to src_vars.
    get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>>(src_vars) =
        get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(
            vars_after_differentiation);
  }

  // Compute dest_vars
  Variables<DestTags> dest_vars(mesh.number_of_grid_points());
  ComputeHorizonVolumeQuantities::apply(make_not_null(&dest_vars), src_vars,
                                        mesh);

  // Now make sure that dest vars are correct.
  const auto expected_christoffel_second_kind = raise_or_lower_first_index(
      gr::christoffel_first_kind(d_g), determinant_and_inverse(g).second);
  CHECK_ITERABLE_APPROX(
      expected_christoffel_second_kind,
      get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
          dest_vars));

  const auto expected_extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, d_shift, g, dt_g, d_g);
  CHECK_ITERABLE_APPROX(
      expected_extrinsic_curvature,
      get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>>(dest_vars));

  if constexpr (tmpl::list_contains_v<
                    DestTags, gr::Tags::SpatialMetric<3, Frame::Inertial>>) {
    CHECK_ITERABLE_APPROX(
        g, get<gr::Tags::SpatialMetric<3, Frame::Inertial>>(dest_vars));
  }

  if constexpr (tmpl::list_contains_v<DestTags, gr::Tags::InverseSpatialMetric<
                                                    3, Frame::Inertial>>) {
    CHECK_ITERABLE_APPROX(
        determinant_and_inverse(g).second,
        get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial>>(dest_vars));
  }

  if constexpr (tmpl::list_contains_v<
                    DestTags, gr::Tags::SpatialRicci<3, Frame::Inertial>>) {
    // Compute Ricci and check.
    // Compute derivative of christoffel_2nd_kind, which is different
    // from how Ricci is computed in ComputeHorizonVolumeQuantities, but
    // which should give the same result to numerical truncation error.
    using tags_before_deriv =
        tmpl::list<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>;
    Variables<tags_before_deriv> vars_before_deriv(
        mesh.number_of_grid_points());
    get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
        vars_before_deriv) =
        get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
            dest_vars);

    const auto inv_jacobian =
        map_logical_to_inertial.inv_jacobian(logical_coordinates(mesh));
    using tags_after_deriv = tmpl::list<::Tags::deriv<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>;
    const auto vars_after_deriv = partial_derivatives<tags_after_deriv>(
        vars_before_deriv, mesh, inv_jacobian);
    const auto expected_ricci = gr::ricci_tensor(
        get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
            dest_vars),
        get<::Tags::deriv<
            gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>,
            tmpl::size_t<3>, Frame::Inertial>>(vars_after_deriv));
    CHECK_ITERABLE_APPROX(
        expected_ricci,
        get<gr::Tags::SpatialRicci<3, Frame::Inertial>>(dest_vars));
  }
}

// void crap() {
//   // Compute dest_vars, time-independent case
//   if constexpr (std::is_same_v<TargetFrame Frame::Inertial>) {
//     // Compute dest_vars, time-independent case
//     Variables<DestTags> dest_vars(mesh.number_of_grid_points());
//     ComputeHorizonVolumeQuantities::apply(make_not_null(&dest_vars),
//                                           src_vars,
//                                           mesh);
//   } else {
//     // Compute dest_vars, time-dependent case
//     Variables<DestTags> dest_vars(mesh.number_of_grid_points());
//     ComputeHorizonVolumeQuantities::apply(make_not_null(&dest_vars),
//                                           src_vars,
//                                           mesh,
//                                           jacobian_target_to_inertial,
//                                           inv_jacobian_logical_to_target);
//   }
// }

SPECTRE_TEST_CASE("Unit.ApparentHorizons.ComputeHorizonVolumeQuantities",
                  "[ApparentHorizons][Unit]") {
  test_compute_horizon_volume_quantities_time_independent<
      tmpl::list<gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                 Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                             tmpl::size_t<3>, Frame::Inertial>>,
      tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial>,
                 gr::Tags::InverseSpatialMetric<3, Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>,
                 gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>,
                 gr::Tags::SpatialRicci<3, Frame::Inertial>>>();
}
}  // namespace
