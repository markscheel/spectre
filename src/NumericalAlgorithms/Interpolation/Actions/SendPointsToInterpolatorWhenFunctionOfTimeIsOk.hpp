// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sets up points on an `InterpolationTarget` and sends these
/// points to an `Interpolator`, for possibly multiple temporal_ids,
/// if the FunctionOfTimes are valid for those temporal_ids.
///
/// Uses:
/// - DataBox:
///   - `domain::Tags::Domain<3>`
///   - `Tags::PendingTemporalIds<TemporalId>`
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::PendingTemporalIds<TemporalId>`
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::AwaitingFunctionOfTimeUpdate`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct SendPointsToInterpolatorWhenFunctionOfTimeIsOk {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags,
                Tags::TemporalIds<typename Metavariables::temporal_id::type>>> =
                nullptr>
  // It is important that 'apply' take no extra arguments, so we can
  // use it as a callback.
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    using TemporalId = typename Metavariables::temporal_id::type;

    // The logic in `SendPointsToInterpolatorWhenFunctionOfTimeIsOk`:
    // - Clear `AwaitingFunctionOfTimeUpdate`
    // - Return if `PendingTemporalIds` is empty (nothing to do).
    // - If the earliest FunctionOfTime expiration is less than all pending
    //   TemporalIds, then
    //     - Set `AwaitingFunctionOfTimeUpdate`
    //     - Set `SendPointsToInterpolatorWhenFunctionOfTimeIsOk` as a callback
    //     - Return, because there is nothing to do.
    //   else
    //     - Move all FunctionOfTime-approved temporal_ids from
    //       `PendingTemporalIds` to `TemporalIds`
    //     - If sequential, and if `TemporalIds` was formerly empty, then
    //          - Call SendPointsToInterpolator for the first `TemporalId`
    //       else if not sequential, then:
    //          - Call SendPointsToInterpolator for all `TemporalId`s.

    db::mutate<tmpl::list<Tags::AwaitingFunctionOfTimeUpdate>>(
        [](const gsl::not_null<bool*>
               awaiting_function_of_time_update) noexcept {
          *awaiting_function_of_time_update = false;
        });

    const auto& pending_temporal_ids =
        db::get<Tags::PendingTemporalIds<TemporalId>>(box);
    if (pending_temporal_ids.empty()) {
      return;
    }

    const bool temporal_ids_is_initially_empty =
        db::get<Tags::TemporalIds<TemporalId>>(box).empty();

    std::vector<TemporalId> new_temporal_ids{};
    if (InterpolationTarget_detail::have_time_dependent_block<
            InterpolationTargetTag>(box, tmpl::type_<Metavariables>{})) {
      // Check if FunctionsOfTime are up to date in GlobalCache.
      auto& this_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      auto callback = CkCallback(
          Parallel::index_from_parallel_component<ParallelComponent>::
              simple_action<SendPointsToInterpolatorWhenFunctionOfTimeIsOk<
                  InterpolationTargetTag>>(),
          this_proxy);

      double min_expiration_time = std::numeric_limits<double>::max();
      const bool is_ready =
          ::Parallel::mutable_cache_item_is_ready<Tags::FunctionsOfTime>(
              cache,
              [&callback, &pending_temporal_ids](
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) -> boost::optional<CkCallback> {
                min_expiration_time = std::min_element(
                    functions_of_time.begin(), functions_of_time.end(),
                    [](const auto& a, const auto& b) {
                      return a.second->time_bounds()[1] <
                             b.second->time_bounds()[1];
                    });
                for (const auto& pending_id : pending_temporal_ids) {
                  if (pending_id.step_time().value() <= min_expiration_time) {
                    return boost::none;  // Success, at least one time is ok.
                  }
                }
                return boost::optional<CkCallback>(callback);  // Failure.
              });

      if (is_ready) {
        // Move the qualifying pending_ids to ids.
        // Keep a list of the newly added ids.
        db::mutate<tmpl::list<Tags::TemporalIds<TemporalId>,
                              Tags::PendingTemporalIds<TemporalId>>>(
            [&min_expiration_time](
                const gsl::not_null<std::deque<TemporalId>*> ids,
                const gsl::not_null<std::deque<TemporalId>*>
                    pending_ids) noexcept {
              for (auto& id : pending_ids) {
                if (id.step_time().value() <= min_expiration_time) {
                  ids->push_back(id);
                  new_temporal_ids.push_back(id);
                  pending_ids->erase(id);
                }
              }
            });
      } else {
        // The call to mutable_cache_item_is_ready has set a callback
        // that will call
        // SendPointsToInterpolatorWhenFunctionOfTimeIsOk when the
        // cache item is indeed ready. We set a flag and we exit now.
        db::mutate<tmpl::list<Tags::AwaitingFunctionOfTimeUpdate>>(
            [](const gsl::not_null<bool*>
                   awaiting_function_of_time_update) noexcept {
              *awaiting_function_of_time_update = true;
            });
        return;
      }
    } else {
      // There is no time dependence, so simply move all pending_ids.
      // Keep a list of the newly added ids.
      db::mutate<tmpl::list<Tags::TemporalIds<TemporalId>,
                            Tags::PendingTemporalIds<TemporalId>>>(
          [](const gsl::not_null<std::deque<TemporalId>*> ids,
             const gsl::not_null<std::deque<TemporalId>*>
                 pending_ids) noexcept {
            for (auto& id : pending_ids) {
              ids->push_back(id);
              new_temporal_ids.push_back(id);
              pending_ids->erase(id);
            }
          });
    }

    const auto& ids = db::get<Tags::TemporalIds<TemporalId>>(box);
    if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
      // InterpolationTarget is sequential.  If temporal_ids was
      // initially empty, then we need to start an interpolation on
      // the first temporal_id by calling SendPointsToInterpolator.
      // If temporal_ids was not initially empty, then there is an
      // interpolation already in progress, and as soon as that
      // interpolation finishes, it will call SendPointsToInterpolator
      // (inside InterpolationTargetReceiveVars) and therefore we
      // don't need to do anything here.
      if (temporal_ids_is_initially_empty and not ids.empty()) {
        auto& my_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        Parallel::simple_action<
            Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
            my_proxy, ids.front());
      }
    } else {
      // InterpolationTarget is not sequential. So begin interpolation
      // on every new temporal_id that has just been added.
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& id : new_temporal_ids) {
        Parallel::simple_action<
            Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                       id);
      }
    }
  }
};

}  // namespace Actions
}  // namespace intrp
