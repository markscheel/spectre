// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {

namespace HorizonManager {
struct PrintNumElements {
  using apply_args = tmpl::list<>;

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename Metavariables::number_of_elements_tag>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using number_of_elements_tag =
        typename Metavariables::number_of_elements_tag;
    const auto& num_elements = db::get<number_of_elements_tag>(box);
    Parallel::printf("Number of elements on proc %d is %d\n",
                     Parallel::my_proc(), num_elements);
    return std::forward_as_tuple(std::move(box));
  }
};

struct ReceiveNumElements {
  using apply_args = tmpl::list<>;

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename Metavariables::number_of_elements_tag>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using number_of_elements_tag =
        typename Metavariables::number_of_elements_tag;
    db::mutate<number_of_elements_tag>(
        box, [](auto& num_elements) noexcept { ++num_elements; });
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename ParallelComponentOfReceiver>
struct SendNumElements {
  using apply_args = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);

    receiver_proxy.ckLocalBranch()
        ->template explicit_single_action<ReceiveNumElements>();

    return std::forward_as_tuple(std::move(box));
  }
};

struct InitNumElements {
  using apply_args = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,  // HorizonManager's box
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using number_of_elements_tag =
        typename Metavariables::number_of_elements_tag;
    return std::make_tuple(
        db::create<tmpl::list<number_of_elements_tag>>(0_st));
  }
};

//-:  template<typename GhVars>
//-:  struct VolumeDataForHorizonManager {
//-:    Element<3> element;
//-:    Index<3> extents;
//-:    GhVars gh_vars;
//-:  };
//-:
//-:struct VolumeDataFromElement {
//-:
//-:  template <typename DbTags, typename... InboxTags, typename Metavariables,
//-:            typename ArrayIndex>
//-:  static bool is_ready(
//-:      const db::DataBox<DbTags>& box,
//-:      const tuples::TaggedTuple<InboxTags...>& inboxes,
//-:      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
//-:      const ArrayIndex& /*array_index*/) noexcept {
//-:
//-:    using horizon_manager_tag = typename Metavariables::horizon_manager_tag;
//-:    using number_of_elements_tag = typename
// Metavariables::number_of_elements_tag;
//-:    auto& inbox = tuples::get<horizon_manager_tag>(inboxes);
//-:    const auto& time_id = db::get<Tags::TimeId>(box);
//-:
//-:    auto receives = inbox.find(time_id);
//-:    const size_t num_receives =
//-:        receives == inbox.end() ? 0 : receives->second.size();
//-:
//-:    const auto& number_of_elements = db::get<number_of_elements_tag>(box);
//-:
//-:    return num_receives == number_of_elements;
//-:  }
//-:
//-:
//-:  // Called by the Element to send volume data to a HorizonManager.
//-:  template <typename DbTags, typename... InboxTags, typename Metavariables,
//-:            typename ArrayIndex, typename ActionList,
//-:            typename ParallelComponent>
//-:  static auto apply(db::DataBox<DbTags>& box, // HorizonManager's box
//-:                    tuples::TaggedTuple<InboxTags...>& inboxes, // Data from
// element.
//-:                    const Parallel::ConstGlobalCache<Metavariables>& cache,
//-:                    const ArrayIndex& /*array_index*/,
//-:                    const ActionList /*meta*/,
//-:                    const ParallelComponent* const /*meta*/) noexcept {
//-:    using gh_vars_tag  = typename Metavariables::gh_vars_tag;
//-:    using horizon_manager_tag = typename Metavariables::horizon_manager_tag;
//-:    auto& inbox = tuples::get<horizon_manager_tag>(inboxes);
//-:    const auto& time_id = db::get<Tags::TimeId>(box);
//-:    auto remote_data = std::move(inbox[time_id]);
//-:    inbox.erase(time_id);
//-:
//-:    // we think remote_data is
// std::unordered_map<element_id(),VolumeDataForHorizonManager>
//-:
//-:    // Here we assume that all the elements are available.
//-:
//-:    // Do everything HorizonManager does, up to calling
//-:    // horizon_proxy.receive_interpolated_volume_data
//-:
//-:  }
//-:};
//-:
//-:  template <typename ParallelComponentOfReceiver>
//-:  struct SendVolumeDataToHorizonManager {
//-:    template <typename DbTags, typename... InboxTags, typename
// Metavariables,
//-:              typename ArrayIndex, typename ActionList,
//-:              typename ParallelComponent>
//-:    static auto apply(db::DataBox<DbTags>& box, // Element's box
//-:                      tuples::TaggedTuple<InboxTags...>& /* inboxes */,
//-:                      const Parallel::ConstGlobalCache<Metavariables>&
// cache,
//-:                      const ArrayIndex& /*array_index*/,
//-:                      const ActionList /*meta*/,
//-:                      const ParallelComponent* const /*meta*/) noexcept {
//-:      using gh_vars_tag  = typename Metavariables::gh_vars_tag;
//-:      using horizon_manager_tag = typename
// Metavariables::horizon_manager_tag;
//-:      const auto& time_id = db::get<Tags::TimeId>(box);
//-:      const auto& element = db::get<Tags::Element<3>>(box);
//-:      const auto& extents = db::get<Tags::Extents<3>>(box);
//-:      const auto& gh_vars = db::get<gh_vars_tag>(box);
//-:
//-:      auto& receiver_proxy =
//-: Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);
//-:
//-:      receiver_proxy.ckLocalBranch().template
//-:        receive_data<horizon_manager_tag>(time_id,
//-:
// std::make_pair(element.id(),
// VolumeDataForHorizonManager<std::remove_cv_t<decltype(gh_vars)>>{
// element,extents,std::move(gh_vars)}));
//-:
//-:      return std::forward_as_tuple(std::move(box));
//-:  };

}  // namespace HorizonManager
}  // namespace Actions
