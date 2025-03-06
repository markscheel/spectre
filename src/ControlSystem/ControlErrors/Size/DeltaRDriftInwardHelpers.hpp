// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "ControlSystem/ControlErrors/Size/State.hpp"

namespace control_system::size {
struct StateUpdateArgs;
}  // namespace control_system::size

namespace control_system::size::States {

/// Value of target_char_speed when state DeltaRDriftInward is in effect.
double target_speed_for_inward_drift(
    double avg_distorted_normal_dot_unit_coord_vector, double min_char_speed,
    double inward_drift_velocity);

/// Returs true if we should transition from state DeltaR to state
/// DeltaRDriftInward.
bool should_transition_from_state_delta_r_to_inward_drift(
    const std::optional<double>& crossing_time_state_3,
    double damping_time, const StateUpdateArgs& update_args);

/// Returns true if we should transition from state DeltaRDriftInward
/// to state DeltaRNoDrift.
bool should_transition_from_state_inward_drift_to_delta_r_no_drift(
    const std::optional<double>& crossing_time_state_3,
    double damping_time, const StateUpdateArgs& update_args);

/// Returns true if we should transition to DeltaRDriftInward rather than
/// to DeltaR.
bool should_activate_inward_drift(const StateUpdateArgs& update_args);

/// Returns true if either CharSpeed approaches min_allowed_char_speed
/// or DeltaR approaches min_allowed_radial_distance close enough
/// that it would be ok to turn off state DeltaRNoDrift.
bool ok_to_return_to_state_deltar(const StateUpdateArgs& update_args);

}  // namespace control_system::size::States
