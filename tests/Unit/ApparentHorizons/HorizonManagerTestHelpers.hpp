// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
namespace DgElementArray {

struct InitializeElement {
  using return_tag_list = tmpl::list<Tags::Extents<3>, Tags::Element<3>>;

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

    db::compute_databox_type<return_tag_list> outbox =
        db::create<db::get_items<return_tag_list>>(std::move(mesh),
                                                   std::move(element));
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

}  // namespace DgElementArray
}  // namespace Actions
