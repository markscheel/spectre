// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRDriftInwardHelpers.hpp"

#include <algorithm>

namespace control_system::size::States {

double target_speed_for_inward_drift(
    const double avg_distorted_normal_dot_unit_coord_vector,
    const double min_char_speed, const double inward_drift_velocity) {
  // TargetSpeed should be > 0 (we want DeltaR to increase).  And
  // TargetSpeed must be <
  // min_char_speed/avg_distorted_normal_dot_unit_coord_vector, because
  // going into DriftInward will make min_char_speed decrease by
  // TargetSpeed*avg_distorted_normal_dot_unit_coord_vector. The time
  // it takes v to cross zero (assuming v decreases linearly, only a
  // rough approximation) is
  // Tau*min_char_speed/avg_distorted_normal_dot_unit_coord_vector*TargetSpeed,
  // where Tau is the damping timescale.  Therefore choosing
  // TargetSpeed < fudge *
  // min_char_speed/avg_distorted_normal_dot_unit_coord_vector should make
  // v decrease only by a factor of fudge, and it should make the
  // crossing time fudge*Tau.
  constexpr double fudge = 0.5;
  return std::min(
      inward_drift_velocity,
      fudge * min_char_speed / avg_distorted_normal_dot_unit_coord_vector);
}

bool should_transition_from_state_delta_r_to_inward_drift(
    const std::optional<double>& crossing_time_state_3,
    const double damping_time,
    const std::optional<double>& inward_drift_velocity,
    const bool delta_r_almost_above_inward_drift_limit,
    const bool char_speed_almost_above_inward_drift_limit,
    const bool comoving_char_speed_increasing_inward) {
  if (inward_drift_velocity.has_value() and
      crossing_time_state_3.has_value() and
      crossing_time_state_3.value() < damping_time) {
    return false;
  }
  return should_activate_inward_drift(
      inward_drift_velocity, delta_r_almost_above_inward_drift_limit,
      char_speed_almost_above_inward_drift_limit,
      comoving_char_speed_increasing_inward);
}

bool should_activate_inward_drift(
    const std::optional<double>& inward_drift_velocity,
    const bool delta_r_almost_above_inward_drift_limit,
    const bool char_speed_almost_above_inward_drift_limit,
    const bool comoving_char_speed_increasing_inward) {
  if (inward_drift_velocity.has_value() and
      (not delta_r_almost_above_inward_drift_limit) and
      (not char_speed_almost_above_inward_drift_limit)) {
    return comoving_char_speed_increasing_inward;
  }
  return false;
}

}  // namespace control_system::size::States
