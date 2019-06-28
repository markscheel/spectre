// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"                             // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace intrp {

/// Holds Actions for Interpolator and InterpolationTarget.
namespace Actions {

// The purpose of the functions and metafunctions in this
// namespace is to allow InterpolationTarget::compute_target_points
// to omit an initialize function and a initialization_tags
// type alias if it doesn't add anything to the DataBox.
namespace initialize_interpolation_target_detail {

// Sets type to initialization_tags, or
// to empty list if initialization_tags is not defined.
template <typename T, typename = cpp17::void_t<>>
struct initialization_tags {
  using type = tmpl::list<>;
};

template <typename T>
struct initialization_tags<
    T, cpp17::void_t<typename T::compute_target_points::initialization_tags>> {
  using type = typename T::compute_target_points::initialization_tags;
};

// Tests whether T::compute_target_points has a non-empty
// initialization_tags member.
template <typename T>
constexpr bool has_empty_initialization_tags_v =
    tmpl::size<typename initialization_tags<T>::type>::value == 0;

// Calls initialization function only if initialization_tags is defined
// and non-empty; otherwise just moves the box.
template <
    typename InterpolationTargetTag, typename DbTags, typename Metavariables,
    Requires<not has_empty_initialization_tags_v<InterpolationTargetTag>> =
        nullptr>
auto make_initial_box(
    db::DataBox<DbTags>&& box,
    const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  return InterpolationTargetTag::compute_target_points::initialize(
      std::move(box), cache);
}

template <
    typename InterpolationTargetTag, typename DbTags, typename Metavariables,
    Requires<has_empty_initialization_tags_v<InterpolationTargetTag>> = nullptr>
auto make_initial_box(
    db::DataBox<DbTags>&& box,
    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
  return std::move(box);
}

}  // namespace initialize_interpolation_target_detail

/// \ingroup ActionsGroup
/// \brief Initializes an InterpolationTarget
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `Tags::TemporalIds<Metavariables>`
///   - `Tags::CompletedTemporalIds<Metavariables>`
///   - `::Tags::Domain<VolumeDim, Frame>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InitializeInterpolationTarget {
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename Metavariables>
  using return_tag_list_initial = tmpl::list<
      Tags::IndicesOfFilledInterpPoints, Tags::TemporalIds<Metavariables>,
      Tags::CompletedTemporalIds<Metavariables>,
      ::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>;
  template <typename Metavariables>
  using return_tag_list =
      tmpl::append<return_tag_list_initial<Metavariables>,
                   typename initialize_interpolation_target_detail::
                       initialization_tags<InterpolationTargetTag>::type>;

  // Custom tag for Domain that contains convert_from_option, which
  // makes a Domain out of a DomainCreator.
  // This tag will be used at the beginning of the initialization,
  // and will then be replaced with the general Domain tag below.
  template <size_t VolumeDim, typename Frame>
  struct CustomDomainTag : db::SimpleTag {
    static std::string name() noexcept { return "Domain"; }
    using type = ::Domain<VolumeDim, Frame>;

    using option_tag = OptionTags::DomainCreator<VolumeDim, Frame>;
    static ::Domain<VolumeDim, Frame> convert_from_option(
        const std::unique_ptr<DomainCreator<VolumeDim, Frame>>&
            creator) noexcept {
      return creator->create_domain();
    }
  };

  template <typename Metavariables>
  struct AddOptionsToDataBox {
    using simple_tags =
        tmpl::list<CustomDomainTag<Metavariables::domain_dim,
                                   typename Metavariables::domain_frame>>;
    template <typename DbTagsList>
    static auto apply(db::DataBox<DbTagsList>&& box,
                      ::Domain<Metavariables::domain_dim,
                               typename Metavariables::domain_frame>
                          domain) noexcept {
      return db::create_from<db::RemoveTags<>, simple_tags>(std::move(box),
                                                            std::move(domain));
    }
  };

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<
          tmpl::list_contains_v<
              DbTagsList, CustomDomainTag<Metavariables::domain_dim,
                                   typename Metavariables::domain_frame>> and
          not tmpl::list_contains_v<
              DbTagsList, Tags::IndicesOfFilledInterpPoints>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using custom_domain_tag =
        CustomDomainTag<Metavariables::domain_dim,
                        typename Metavariables::domain_frame>;
    using general_domain_tag = ::Tags::Domain<Metavariables::domain_dim,
                                          typename Metavariables::domain_frame>;
    // We needed custom_domain_tag for option parsing reasons, but in
    // the DataBox we return, we want the Domain to be available under
    // the general_domain_tag.

    // So first grab Domain from the DataBox.
    db::item_type<custom_domain_tag> domain{};
    db::mutate<custom_domain_tag>(
        make_not_null(&box), [&domain](const auto domain_ptr) noexcept {
          domain = std::move(*domain_ptr);
        });

    // Second, when constructing the new DataBox, remove
    // custom_domain_tag and insert the Domain under
    // general_domain_tag.
    auto new_box = initialize_interpolation_target_detail::make_initial_box<
        InterpolationTargetTag>(
        db::create_from<
            db::RemoveTags<custom_domain_tag>,
            tmpl::push_back<
                db::get_items<return_tag_list_initial<Metavariables>>,
                general_domain_tag>>(
            std::move(box), db::item_type<Tags::IndicesOfFilledInterpPoints>{},
            db::item_type<Tags::TemporalIds<Metavariables>>{},
            db::item_type<Tags::CompletedTemporalIds<Metavariables>>{},
            db::item_type<
                ::Tags::Variables<typename InterpolationTargetTag::
                                      vars_to_interpolate_to_target>>{},
            std::move(domain)),
        cache);
    // compute_items_on_target will depend on compute items added in
    // make_initial_box, so compute_items_on_target must be added
    // in a separate step.
    return std::make_tuple(
        db::create_from<
            db::RemoveTags<>, db::AddSimpleTags<>,
            db::AddComputeTags<
                typename InterpolationTargetTag::compute_items_on_target>>(
            std::move(new_box)));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTagsList, Tags::IndicesOfFilledInterpPoints>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace intrp
