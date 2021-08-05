// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

/// \cond
template <typename Frame>
class Strahlkorper;
namespace domain {
template <typename A, typename B, size_t I>
class CoordinateMapBase;
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// \brief Transforms a Strahlkorper from SrcFrame to DestFrame.
///
/// The destination Strahlkorper has the same l_max() and m_max() as the
/// source Strahlkorper.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    const gsl::not_null<Strahlkorper<DestFrame>*> dest_strahlkorper,
    const Strahlkorper<SrcFrame>& src_strahlkorper,
    const domain::CoordinateMapBase<SrcFrame, DestFrame, 3>& map_src_to_dest,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) noexcept;
