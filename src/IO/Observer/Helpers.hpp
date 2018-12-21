// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Tags.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TMPL.hpp"

namespace observers {
namespace detail {

template <typename T>
struct make_reduction_data_tags;

template <typename... Ts>
struct make_reduction_data_tags<tmpl::list<Ts...>> {
  using type = typename ::observers::Tags::ReductionData<Ts...>;
};

template <typename T>
struct make_reduction_data;

template <typename... Ts>
struct make_reduction_data<tmpl::list<Ts...>> {
  using type = typename Parallel::ReductionData<Ts...>;
};

template <class Observer, class = cpp17::void_t<>>
struct get_reduction_data_tags_from_observer {
  using type = tmpl::list<>;
};

template <class Observer>
struct get_reduction_data_tags_from_observer<
    Observer, cpp17::void_t<typename Observer::reduction_data_tags>> {
  using type = typename Observer::reduction_data_tags;
};
}  // namespace detail

/// Each Observer should specify a type alias `reduction_data_tags`.
/// Given a list of Observers, this metafunction is used to get the full set of
/// `reduction_data_tags` that should be put into Metavariables.
template <class ObserverList>
using get_reduction_data_tags = tmpl::remove_duplicates<tmpl::transform<
    ObserverList, detail::get_reduction_data_tags_from_observer<tmpl::_1>>>;

/// Given a std::list of ReductionDatums, makes an
/// observers::Tags::ReductionData.
template <typename ReductionDatumList>
using make_reduction_data_tags_t =
    typename detail::make_reduction_data_tags<ReductionDatumList>::type;

/// Given a std::list of ReductionDatums, makes an
/// Parallel::ReductionData.
template <typename ReductionDatumList>
using make_reduction_data_t =
    typename detail::make_reduction_data<ReductionDatumList>::type;
}  // namespace observers
