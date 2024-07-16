// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once
#include <algorithm>

namespace control_system::size::States {

double target_speed_for_inward_drift(
    const double avg_distorted_normal_dot_unit_coord_vector,
    const double char_speed, const double inward_drift_velocity) {
  // TargetSpeed should be > 0 (we want DeltaR to increase).  And
  // TargetSpeed must be <
  // char_speed/avg_distorted_normal_dot_unit_coord_vector, because
  // going into DriftInward will make char_speed decrease by
  // TargetSpeed*avg_distorted_normal_dot_unit_coord_vector. The time
  // it takes v to cross zero (assuming v decreases linearly, only a
  // rough approximation) is
  // Tau*char_speed/avg_distorted_normal_dot_unit_coord_vector*TargetSpeed,
  // where Tau is the damping timescale.  Therefore choosing
  // TargetSpeed < fudge *
  // char_speed/avg_distorted_normal_dot_unit_coord_vector should make
  // v decrease only by a factor of fudge, and it should make the
  // crossing time fudge*Tau.
  constexpr double fudge = 0.5;
  return std::min(
      inward_drift_velocity,
      fudge * char_speed / avg_distorted_normal_dot_unit_coord_vector);
}

}  // namespace control_system::size::States
avg_distorted_normal_dot_unit_coord_vector
