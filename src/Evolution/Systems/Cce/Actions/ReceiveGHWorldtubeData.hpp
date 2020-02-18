// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/WorldtubeInterfaceManager.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Stores the boundary data from the GH evolution in the
 * `Cce::GHWorldtubeInterfaceManager`, and sends to the `EvolutionComponent`
 * (template argument) if the data fulfills a prior request.
 *
 * \details If the new data fulfills a prior request submitted to the
 * `Cce::GHWorldtubeInterfaceManager`, this will dispatch the result to
 * `Cce::Actions::SendToEvolution<GHWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` for sending the processed boundary data to
 * the `EvolutionComponent`.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GHInterfaceManager`
 */
template <typename EvolutionComponent>
struct ReceiveGHWorldtubeData {
  using const_global_cache_tags = tmpl::list<InitializationTags::LMax>;
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>,
                                     Tags::GHInterfaceManager> and
               tmpl2::flat_any_v<cpp17::is_same_v<
                   ::Tags::Variables<
                       typename Metavariables::cce_boundary_communication_tags>,
                   DbTags>...>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const TimeStepId& time,
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
      const tnsr::aa<DataVector, 3>& dt_spacetime_metric =
          tnsr::aa<DataVector, 3>{},
      const tnsr::iaa<DataVector, 3>& dt_phi = tnsr::iaa<DataVector, 3>{},
      const tnsr::aa<DataVector, 3>& dt_pi =
          tnsr::aa<DataVector, 3>{}) noexcept {
    const double mass = 1.0;
    const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
    const std::array<double, 3> center{{0.0, 0.0, 0.0}};
    gr::Solutions::KerrSchild solution{mass, spin, center};
    const double extraction_radius = 48.0;

    const size_t l_max = db::get<Spectral::Swsh::Tags::LMax>(box);
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // create the vector of collocation points that we want to interpolate to
    tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
    const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get<0>(collocation_points)[collocation_point.offset] =
          extraction_radius * sin(collocation_point.theta) *
          cos(collocation_point.phi);
      get<1>(collocation_points)[collocation_point.offset] =
          extraction_radius * sin(collocation_point.theta) *
          sin(collocation_point.phi);
      get<2>(collocation_points)[collocation_point.offset] =
          extraction_radius * cos(collocation_point.theta);
    }

    const auto kerr_schild_variables = solution.variables(
        collocation_points, 0.0, gr::Solutions::KerrSchild::tags<DataVector>{});

    // direct collocation quantities for processing into the GH form of the
    // worldtube function
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(kerr_schild_variables);
    const auto& dt_lapse =
        get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(kerr_schild_variables);
    const auto& d_lapse =
        get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
            kerr_schild_variables);

    const auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
        kerr_schild_variables);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
            kerr_schild_variables);
    const auto& d_shift =
        get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
            kerr_schild_variables);

    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
            kerr_schild_variables);
    const auto& dt_spatial_metric = get<
        ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
        kerr_schild_variables);
    const auto& d_spatial_metric =
        get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
            kerr_schild_variables);

    const auto expected_phi = GeneralizedHarmonic::phi(
        lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
    const auto expected_psi =
        gr::spacetime_metric(lapse, shift, spatial_metric);
    const auto expected_pi =
        GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                spatial_metric, dt_spatial_metric, phi);

    DataVector difference{expected_psi[0].size()};
    for (size_t i = 0; i < expected_psi.size(); ++i) {
      difference = spacetime_metric[i] - expected_psi[i];
      for (size_t j = 0; j < difference.size(); ++j) {
        if (abs(difference[j]) > 1.0e-10) {
          Parallel::printf(
              "spacetime metric component %zu element differed by %e; value : "
              "%e\n",
              i, difference[j], expected_psi[i][j]);
        }
      }
    }
    for (size_t i = 0; i < pi.size(); ++i) {
      difference = pi[i] - expected_pi[i];
      for (size_t j = 0; j < difference.size(); ++j) {
        if (abs(difference[j]) > 1.0e-10) {
          Parallel::printf(
              "pi component %zu element differed by %e; value : %e\n", i,
              difference[j], expected_pi[i][j]);
        }
      }
    }
    for (size_t i = 0; i < phi.size(); ++i) {
      difference = phi[i] - expected_phi[i];
      for (size_t j = 0; j < difference.size(); ++j) {
        if (abs(difference[j]) > 1.0e-10) {
          Parallel::printf(
              "phi component %zu element differed by %e; value : %e\n", i,
              difference[j], expected_pi[i][j]);
        }
      }
    }

    db::mutate<Tags::GHInterfaceManager>(
        make_not_null(&box),
        [
          &spacetime_metric, &phi, &pi, &dt_spacetime_metric, &dt_phi, &dt_pi,
          &time, &cache
        ](const gsl::not_null<db::item_type<Tags::GHInterfaceManager>*>
              interface_manager) noexcept {
          (*interface_manager)
              ->insert_gh_data(time, spacetime_metric, phi, pi,
                               dt_spacetime_metric, dt_phi, dt_pi);
          if (const auto gh_data =
                  (*interface_manager)->try_retrieve_first_ready_gh_data()) {
            Parallel::simple_action<Actions::SendToEvolution<
                GHWorldtubeBoundary<Metavariables>, EvolutionComponent>>(
                Parallel::get_parallel_component<
                    GHWorldtubeBoundary<Metavariables>>(cache),
                get<0>(*gh_data), get<1>(*gh_data), get<2>(*gh_data),
                get<3>(*gh_data));
          }
        });
  }
};
}  // namespace Actions
}  // namespace Cce
