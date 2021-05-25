// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp::Actions {
/// \ingroup ActionsGroup
/// \brief Adds a single `temporal_id` on which this InterpolationTarget
/// should be triggered; the `temporal_id` must correspond to the minimum
/// FunctionOfTime expiration time.
///
/// Invoked on an InterpolationTarget to trigger interpolation that
/// updates a FunctionOfTime.
///
/// Should be called only by an Action that needs to update the control system
/// at exactly the expiration time of the FunctionOfTime, and needs to do
/// interpolation at that time in order to do the update.
///
/// \details
/// AddExpiringTemporalIdToInterpolationTarget is similar to
/// AddTemporalIdsToInterpolationTarget.  However, for
/// AddExpiringTemporalIdToInterpolationTarget:
/// - Only a single `temporal_id` is passed in, and it must correspond
///   to the earliest expiration time of all the FunctionVsTimes.
/// - InterpolationTargetTag::compute_target_points::is_sequential must
///   be true.
///
/// The logic is as follows:
///
/// - If PendingTemporalIds is non-empty on entry, then there are
///   pending temporal_ids waiting inside a
///   VerifyTemporalIdsAndSendPoints callback.  We do a sanity check
///   that the new temporal_id is less than these PendingTemporalIds
///   (otherwise the PendingTemporalIds should not be pending, because
///   they are at times before the expiration time of the FunctionsOfTime).
/// - If TemporalIds is non-empty on entry, then there is an
///   interpolation in progress. We therefore append the new
///   temporal_id to TemporalIds (unless it is the same as all the
///   current TemporalIds) and we exit. Then when the in-progress
///   interpolation finishes, it will call
///   InterpolationTargetReceiveVars which will call
///   SendPointsToInterpolator and start the interpolation on the new
///   temporal_id.
/// - If TemporalIds is empty on entry, meaning that there is no
///   interpolation in progress, then we call SendPointsToInterpolator
///   on the new temporal_id.
///
/// Uses:
/// - DataBox:
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::PendingTemporalIds<TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TemporalIds<TemporalId>`
template <typename InterpolationTargetTag>
struct AddExpiringTemporalIdToInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::TemporalIds<TemporalId>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    TemporalId&& temporal_id) noexcept {
    static_assert(
        InterpolationTargetTag::compute_target_points::is_sequential::value,
        "AddExpiredTemporalIdToInterpolationTarget works only with sequential "
        "InterpolationTargets");
    static_assert(InterpolationTarget_detail::cache_contains_functions_of_time<
                      Metavariables>::value,
                  "AddExpiredTemporalIdToInterpolationTarget works only when "
                  "there are FunctionsOfTime in the GlobalCache");

    const bool temporal_ids_are_initially_empty =
        db::get<Tags::TemporalIds<TemporalId>>(box).empty();
    const auto new_temporal_ids =
        InterpolationTarget_detail::flag_temporal_ids_for_interpolation<
            InterpolationTargetTag>(make_not_null(&box), {{temporal_id}});
    if (new_temporal_ids.empty()) {
      // If we didn't add any temporal_ids that aren't already there,
      // then this is a duplicate call (i.e. it is called from every element
      // but only the first call counts), so do nothing.
      return;
    }

    // Do some sanity checks, only for non-duplicate calls.
    const auto& functions_of_time = db::get<domain::Tags::FunctionsOfTime>(box);
    const double min_expiration_time =
        std::min_element(functions_of_time.begin(), functions_of_time.end(),
                         [](const auto& a, const auto& b) {
                           return a.second->time_bounds()[1] <
                                  b.second->time_bounds()[1];
                         })
            ->second->time_bounds()[1];
    if(min_expiration_time != temporal_id.step_time().value()) {
      ERROR(
          "The temporal_id must correspond to the minimum expiration time. "
          "temporal_id = "
          << temporal_id.step_time().value()
          << ", min expiration time = " << min_expiration_time << ", diff = "
          << temporal_id.step_time().value() - min_expiration_time);
    }

    for (const auto& pending_temporal_id :
         db::get<Tags::PendingTemporalIds<TemporalId>>(box)) {
      if (pending_temporal_id.step_time().value() <=
          temporal_id.step_time().value()) {
        ERROR(
            "Pending temporal_ids must occur later than the passed-in "
            "temporal_id. Pending_id = "
            << pending_temporal_id.step_time().value() << ", temporal_id = "
            << temporal_id.step_time().value() << ", diff = "
            << pending_temporal_id.step_time().value() -
                   temporal_id.step_time().value());
      }
    }

    if (temporal_ids_are_initially_empty) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
          my_proxy, temporal_id);
    }
  }
};
}  // namespace intrp::Actions
