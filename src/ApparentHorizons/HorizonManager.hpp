// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace HorizonManager {
struct Psi : db::DataBoxTag {
  using type = tnsr::aa<DataVector, 3, Frame::Inertial>;
  static constexpr db::DataBoxString label = "Psi";
};
struct Pi : db::DataBoxTag {
  using type = tnsr::aa<DataVector, 3, Frame::Inertial>;
  static constexpr db::DataBoxString label = "Pi";
};
struct Phi : db::DataBoxTag {
  using type = tnsr::iaa<DataVector, 3, Frame::Inertial>;
  static constexpr db::DataBoxString label = "Phi";
};
using VariablesTags = tmpl::list<Psi, Pi, Phi>;

struct InputVars : db::DataBoxTag {
  using type = Variables<VariablesTags>;
  static constexpr db::DataBoxString label = "Vars";
};
}  // namespace HorizonManager

namespace Actions {

namespace HorizonManager {
struct PrintNumElements {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename Metavariables::number_of_elements_tag>> = nullptr>
  static void apply(const db::DataBox<DbTags>& box,  // HorizonManager's box
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using number_of_elements_tag =
        typename Metavariables::number_of_elements_tag;
    const auto& num_elements = db::get<number_of_elements_tag>(box);
    Parallel::printf("Number of elements on proc %d is %d\n",
                     Parallel::my_proc(), num_elements);
  }
};

struct ReceiveNumElements {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename Metavariables::number_of_elements_tag>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using number_of_elements_tag =
        typename Metavariables::number_of_elements_tag;
    db::mutate<number_of_elements_tag>(
        box, [](auto& num_elements) noexcept { ++num_elements; });
  }
};

struct InitNumElements {
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
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

//-:struct HorizonDataOnEachElement {
//-:  Element<3> element;
//-:  Index<3> mesh;
//-:};
//-:
//-:struct GetVolumeDataFromElement {
//-:  template <
//-:      typename DbTags, typename... InboxTags, typename Metavariables,
//-:      typename ArrayIndex, typename ActionList, typename ParallelComponent,
//-:      Requires<tmpl::list_contains_v<
//-:          DbTags, typename Metavariables::number_of_elements_tag>> =
//nullptr>
//-:  static void apply(db::DataBox<DbTags>& box,
//-:                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
//-:                    const Parallel::ConstGlobalCache<Metavariables>&
///*cache*/,
//-:                    const ArrayIndex& /*array_index*/,
//-:                    const ActionList /*meta*/,
//-:                    const ParallelComponent* const /*meta*/,
//-:                    Element<3> element,
//-:                    Index<3> mesh) noexcept {
//-:
//-:  }
//-:};

}  // namespace HorizonManager
}  // namespace Actions
