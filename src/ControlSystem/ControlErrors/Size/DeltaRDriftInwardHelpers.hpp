// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace control_system::size::States {

/// Value of target_char_speed when state DeltaRDriftInward is in effect.
double target_speed_for_inward_drift(
    const double avg_distorted_normal_dot_unit_coord_vector,
    const double min_char_speed, const double inward_drift_velocity);

/// Returs true if we should transition from state DeltaR to state
/// DeltaRDriftInward.
bool should_transition_from_state_delta_r_to_inward_drift(
    const std::optional<double>& crossing_time_state_3,
    const double damping_time,
    const std::optional<double>& inward_drift_velocity,
    const bool delta_r_almost_above_inward_drift_limit,
    const bool char_speed_almost_above_inward_drift_limit,
    const bool comoving_char_speed_increasing_inward);

/// Returns true if we should transition from state DeltaRDriftInward
/// to state DeltaRNoDrift.
bool should_transition_from_state_inward_drift_to_delta_r_no_drift(
    const std::optional<double>& crossing_time_state_3,
    const double damping_time,
    const std::optional<double>& inward_drift_velocity,
    const bool delta_r_almost_above_inward_drift_limit,
    const bool char_speed_almost_above_inward_drift_limit,
    const bool comoving_char_speed_increasing_inward) {
  return (not should_transition_from_state_delta_r_to_inward_drift(
      crossing_time_state_3, damping_time, inward_drift_velocity,
      delta_r_almost_above_inward_drift_limit,
      char_speed_almost_above_inward_drift_limit,
      comoving_char_speed_increasing_inward));
}

/// Returns true if we should transition to DeltaRDriftInward rather than
/// to DeltaR.
bool should_activate_inward_drift(
    const std::optional<double>& inward_drift_velocity,
    const bool delta_r_almost_above_inward_drift_limit,
    const bool char_speed_almost_above_inward_drift_limit,
    const bool comoving_char_speed_increasing_inward);

}  // namespace control_system::size::States
