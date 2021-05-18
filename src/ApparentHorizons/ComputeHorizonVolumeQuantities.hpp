// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace ah {

/// Given the generalized harmonic variables in the volume, computes
/// the upper 3-metric, lower extrinsic curvature, and 3-Christoffel
/// symbol of the second kind.  These are the variables needed by the
/// horizon finder on the volume.
///
/// For the dual-frame case, takes the Jacobians and does numerical
/// derivatives.
struct ComputeHorizonVolumeQuantities {
  // Single-frame case
  template <typename SrcTagList, typename DestTagList>
  static void apply(const gsl::not_null<Variables<DestTagList>*> target_vars,
                    const Variables<SrcTagList>& src_vars,
                    const Mesh<3>& mesh) noexcept;
  // Dual-frame case
  template <typename SrcTagList, typename DestTagList, typename TargetFrame>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> target_vars,
      const Variables<SrcTagList>& src_vars, const Mesh<3>& mesh,
      const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>& jacobian,
      const InverseJacobian<DataVector, 3, Frame::Logical, TargetFrame>&
          inverse_jacobian) noexcept;
};

}  // namespace ah
