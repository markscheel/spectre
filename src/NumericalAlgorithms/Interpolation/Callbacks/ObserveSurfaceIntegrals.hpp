// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace StrahlkorperTags {
template <typename Frame>
struct Jacobian;
template <typename Frame>
struct NormalOneForm;
template <typename Frame>
struct Radius;
template <typename Frame>
struct Rhat;
template <typename Frame>
struct Strahlkorper;
}  // namespace StrahlkorperTags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace observers {
namespace ThreadedActions {
struct WriteReductionData;
}  // namespace ThreadedActions
template <class Metavariables>
struct ObserverWriter;
}  // namespace observers
/// \endcond

namespace intrp {
namespace callbacks {

/// \cond
namespace detail {

template <typename T>
struct reduction_data_tag_type;

template <template <typename...> class T, typename... Ts>
struct reduction_data_tag_type<T<Ts...>> {
  // The first argument is for Time, the others are for
  // the list of scalars being integrated.
  using type = observers::Tags::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<typename Ts::type::type::value_type,
                               funcl::AssertEqual<>>...>;
};

template <typename T>
using reduction_data_tag_type_t = typename reduction_data_tag_type<T>::type;

template <typename... Ts>
auto make_legend(tmpl::list<Ts...> /* meta */) {
  return std::vector<std::string>{"Time", Ts::name()...};
}

template <typename DbTags, typename StrahlkorperType, typename AreaElement,
          typename... Ts>
auto make_reduction_data(const db::DataBox<DbTags>& box,
                         const StrahlkorperType& strahlkorper,
                         const AreaElement& area_element, double time,
                         tmpl::list<Ts...> /* meta */) {
  using reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<typename Ts::type::type::value_type,
                               funcl::AssertEqual<>>...>;
  return reduction_data(time, StrahlkorperGr::surface_integral_of_scalar(
                                  area_element, get<Ts>(box), strahlkorper)...);
}

}  // namespace detail
/// \endcond

/// \brief post_interpolation_callback that outputs
/// surface integrals on a Strahlkorper.
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `StrahlkorperTags::items_tags<Frame>`
///   - `StrahlkorperTags::compute_items_tags<Frame>`
///   - `TagsToObserve`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
template <typename TagsToObserve, typename InterpolationTargetTag,
          typename Frame>
struct ObserveSurfaceIntegrals {
  using reduction_data_tags = detail::reduction_data_tag_type_t<TagsToObserve>;

  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    const auto& strahlkorper = get<StrahlkorperTags::Strahlkorper<Frame>>(box);
    const auto area_element = StrahlkorperGr::area_element(
        get<gr::Tags::SpatialMetric<3, Frame>>(box),
        get<StrahlkorperTags::Jacobian<Frame>>(box),
        get<StrahlkorperTags::NormalOneForm<Frame>>(box),
        get<StrahlkorperTags::Radius<Frame>>(box),
        get<StrahlkorperTags::Rhat<Frame>>(box));

    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
        proxy[0], observers::ObservationId(temporal_id.time()),
        std::string{"/" + pretty_type::short_name<InterpolationTargetTag>() +
                    "_integrals"},
        detail::make_legend(TagsToObserve{}),
        detail::make_reduction_data(box, strahlkorper, area_element,
                                    temporal_id.time().value(),
                                    TagsToObserve{}));
  }
};
}  // namespace callbacks
}  // namespace intrp
