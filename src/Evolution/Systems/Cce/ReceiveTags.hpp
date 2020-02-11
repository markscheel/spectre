// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
namespace ReceiveTags {

namespace detail {
struct AllSame {
  template <typename Lhs>
  size_t operator()(const Lhs& /*lhs*/) const noexcept {
    return 0;
  }
};
}  // namespace detail

/// A receive tag for the data sent to the CCE evolution component from the CCE
/// boundary component
template <typename CommunicationTagList>
struct BoundaryData {
  using temporal_id = TimeStepId;
  using type =
      std::unordered_map<temporal_id,
                         // there should only be one of these per time id, so we
                         // don't care what the comparitor does
                         std::unordered_multiset<
                             Variables<CommunicationTagList>, detail::AllSame>>;
};

}  // namespace ReceiveTags
}  // namespace Cce
