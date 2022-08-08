// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \brief Computes approximate expansion center of a Strahlkorper, given
/// a set of points on the Strahlkorper surface.
///
/// Used primarily when trying to evaluate a Strahlkorper in a
/// different frame, in which case `cartesian_coords` is typically
/// computed using `strahlkorper_coords_in_different_frame`.
template <typename Frame>
std::array<double, 3> strahlkorper_center_from_coords(
    const tnsr::I<DataVector, 3, Frame>& cartesian_coords);
