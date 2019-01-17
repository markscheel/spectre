// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace intrp {
template <typename Metavariables>
struct Interpolator;
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Invoked on the Interpolator ParallelComponent to register an
/// Element with the Interpolator.
///
/// This is called by RegisterElementWithInterpolator below.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::NumberOfElements`
///
/// For requirements on Metavariables, see InterpolationTarget.
struct RegisterElement {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTags, typename Tags::NumberOfElements>> =
          nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::NumberOfElements>(
        make_not_null(&box), [](const gsl::not_null<
                                 db::item_type<Tags::NumberOfElements>*>
                                    num_elements) noexcept {
          ++(*num_elements);
        });
  }
};

/// \ingroup ActionsGroup
/// \brief Called from DgElementArray::execute_next_phase. This is invoked
/// on the DgElementArray ParallelComponent to register all its Elements with
/// the Interpolator.
///
/// We cannot call the above RegisterElement directly from execute_next_phase
/// because that would result in a single call to RegisterElement rather
/// than one call per Element.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
///
struct RegisterElementWithInterpolator {
  template<
    typename DbTags, typename... InboxTags, typename Metavariables,
    typename ArrayIndex, typename ActionList, typename ParallelComponent>
      static void apply(const db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& interpolator =
        *Parallel::get_parallel_component<::intrp::Interpolator<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<RegisterElement>(interpolator);
  }
};

}  // namespace Actions
}  // namespace intrp
