// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Adds `temporal_id`s on which this InterpolationTarget
/// should be triggered.
///
/// Uses:
/// - DataBox:
///   - `Tags::PendingTemporalIds<TemporalId>`
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::PendingTemporalIds<TemporalId>`
template <typename InterpolationTargetTag>
struct AddTemporalIdsToInterpolationTarget {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, typename TemporalId,
      Requires<tmpl::list_contains_v<DbTags, Tags::TemporalIds<TemporalId>>> =
          nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    std::vector<TemporalId>&& temporal_ids) noexcept {
    if (not InterpolationTarget_detail::
                flag_temporal_ids_for_pending_interpolation<
                    InterpolationTargetTag>(make_not_null(&box), temporal_ids)
                    .empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolatorWhenFunctionOfTimeIsOk<
              InterpolationTargetTag>>(my_proxy);
    }
  };
}  // namespace Actions
}  // namespace intrp
