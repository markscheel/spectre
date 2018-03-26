// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "ApparentHorizons/HorizonManager.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
namespace DgElementArray {

struct InitializeElement {
  using return_tag_list = tmpl::list<Tags::Extents<3>, Tags::Element<3>,
                                     ::HorizonManager::InputVars>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const std::vector<std::array<size_t, 3>>& initial_extents,
                    const Domain<3, Frame::Inertial>& domain) noexcept {
    ElementId<3> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Element<3> element = create_initial_element(element_id, my_block);
    ::Index<3> mesh{initial_extents[element_id.block_id()]};

    // Coordinates
    ElementMap<3, Frame::Inertial> map{element_id,
                                       my_block.coordinate_map().get_clone()};
    auto inertial_coords = map(logical_coordinates(mesh));

    EinsteinSolutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});
    const auto input_vars = solution.solution(inertial_coords, 0.0);
    const auto& lapse =
        get<gr::Tags::Lapse<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& dt_lapse =
        get<gr::Tags::DtLapse<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& d_lapse =
        get<EinsteinSolutions::KerrSchild::deriv_lapse<DataVector>>(input_vars);
    const auto& shift =
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& d_shift =
        get<EinsteinSolutions::KerrSchild::deriv_shift<DataVector>>(input_vars);
    const auto& dt_shift =
        get<gr::Tags::DtShift<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& g =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            input_vars);
    const auto& dt_g =
        get<gr::Tags::DtSpatialMetric<3, Frame::Inertial, DataVector>>(
            input_vars);
    const auto& d_g =
        get<EinsteinSolutions::KerrSchild::deriv_spatial_metric<DataVector>>(
            input_vars);

    Variables<::HorizonManager::VariablesTags> output_vars(mesh.product());
    auto& psi = get<::HorizonManager::Psi>(output_vars);
    auto& pi = get<::HorizonManager::Pi>(output_vars);
    auto& phi = get<::HorizonManager::Phi>(output_vars);
    psi = gr::spacetime_metric(lapse, shift, g);
    phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
    pi =
        GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g, phi);

    db::compute_databox_type<return_tag_list> outbox =
        db::create<db::get_items<return_tag_list>>(std::move(mesh),
                                                   std::move(element),
                                                   std::move(output_vars));
    return std::make_tuple(std::move(outbox));
  }
};

template <typename ParallelComponentOfReceiver>
struct SendNumElements {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);

    receiver_proxy.ckLocalBranch()
        ->template simple_action<Actions::HorizonManager::ReceiveNumElements>();
  }
};

//-:template <typename ParallelComponentOfReceiver>
//-:struct BeginHorizonSearch {
//-:  template <typename DbTags, typename... InboxTags, typename Metavariables,
//-:            typename ArrayIndex, typename ActionList,
//-:            typename ParallelComponent>
//-:  static void apply(const db::DataBox<DbTags>& /*box*/,
//-:                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
//-:                    const Parallel::ConstGlobalCache<Metavariables>& cache,
//-:                    const ArrayIndex& /*array_index*/,
//-:                    const ActionList /*meta*/,
//-:                    const ParallelComponent* const /*meta*/) noexcept {
//-:    auto& receiver_proxy =
//-: Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);
//-:
//-:    receiver_proxy.ckLocalBranch()
//-:        ->template
// simple_action<Actions::HorizonManager::GetVolumeDataFromElement>();
//-:  }
//-:};

}  // namespace DgElementArray
}  // namespace Actions
