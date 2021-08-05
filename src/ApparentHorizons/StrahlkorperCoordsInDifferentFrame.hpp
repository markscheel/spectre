// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
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

/// \brief Computes Cartesian coordinates of a Strahlkorper surface in a
/// frame different than the Strahlkorper frame.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_coords_in_different_frame(
    gsl::not_null<tnsr::I<DataVector, 3, DestFrame>*> cartesian_coords,
    const Strahlkorper<SrcFrame>& strahlkorper,
    const domain::CoordinateMapBase<SrcFrame, DestFrame, 3>& map_src_to_dest,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) noexcept;
