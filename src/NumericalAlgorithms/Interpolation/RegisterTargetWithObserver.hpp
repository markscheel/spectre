// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace intrp {
template <typename Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget;
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \brief Register an InterpolationTarget that will call zero or
/// more ObserverWriters.
///
/// Note that each of the post_interpolation_callback::observation_types
/// are registered separately, since they might be observed separately.
///
/// Should be invoked on the InterpolationTarget.
template <typename InterpolationTargetTag>
struct RegisterTargetWithObserver {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    tmpl::for_each<typename InterpolationTargetTag::
                       post_interpolation_callback::observation_types>([&cache](
        auto type_v) noexcept {
      using type = typename decltype(type_v)::type;
      const auto initial_observation_id = observers::ObservationId{0., type{}};
      Parallel::simple_action<
          observers::Actions::RegisterSingletonWithObserverWriter>(
          Parallel::get_parallel_component<
              InterpolationTarget<Metavariables, InterpolationTargetTag>>(
              cache),
          initial_observation_id);
    });
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace intrp
