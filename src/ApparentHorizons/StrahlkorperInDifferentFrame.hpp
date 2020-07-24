// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

/// \cond
template <typename Frame>
class Strahlkorper;
namespace domain {
template <typename SrcFrame, typename DestFrame, size_t>
class CoordinateMapBase;
}  // namespace domain
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// Maps a Strahlkorper from SrcFrame to DestFrame.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    gsl::not_null<Strahlkorper<DestFrame>*> new_strahlkorper,
    const Strahlkorper<SrcFrame>& strahlkorper,
    const std::unique_ptr<domain::CoordinateMapBase<SrcFrame, DestFrame, 3>>&
        map) noexcept;
