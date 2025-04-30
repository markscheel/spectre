// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRDriftInwardHelpers.hpp"

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
    const std::optional<double>& crossing_time_drift_limit,
    const double damping_time, const StateUpdateArgs& update_args) {
  // This function is called ShouldEnterState3FromState2 in SpEC.
  if (update_args.inward_drift_velocity.has_value() and
      crossing_time_drift_limit.has_value() and
      crossing_time_drift_limit.value() < damping_time) {
    return false;
  }
  return should_activate_inward_drift(update_args);
}

bool should_transition_from_state_inward_drift_to_delta_r_no_drift(
    const std::optional<double>& crossing_time_drift_limit,
    const double damping_time, const StateUpdateArgs& update_args) {
  return (not should_transition_from_state_delta_r_to_inward_drift(
      crossing_time_drift_limit, damping_time, update_args));
}

bool should_activate_inward_drift(const StateUpdateArgs& update_args) {
  // This function is called PreferState3OverState2 in SpEC.

  // This drift factor was chosen in SpEC arbitrarily to be 0.9.
  constexpr double inward_drift_limit_buffer_factor = 0.9;

  // The idea of these variables is to check whether either DeltaR or
  // char speed are close to going above the
  // min_average_radial_distance or min_allowed_char_speed values.  If
  // so, then we don't need state DeltaRDriftInward at the moment.
  // For reference, in SpEC these variables are called
  // "DeltaRAlmostAboveState3Limit" and
  // "CharSpeedAlmostAboveState3Limit".
  const bool delta_r_almost_above_inward_drift_limit =
      update_args.min_allowed_radial_distance.has_value() and
      update_args.average_radial_distance.value() >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_radial_distance.value();
  const bool char_speed_almost_above_inward_drift_limit =
      update_args.min_allowed_char_speed.has_value() and
      update_args.min_char_speed >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_char_speed.value();

  return (update_args.inward_drift_velocity.has_value() and
          update_args.comoving_char_speed_increasing_inward and
          (update_args.min_allowed_char_speed.has_value() or
           update_args.min_allowed_radial_distance.has_value()) and
          (not delta_r_almost_above_inward_drift_limit) and
          (not char_speed_almost_above_inward_drift_limit));
}

bool ok_to_return_to_state_deltar(const StateUpdateArgs& update_args) {
  // The purpose of delta_r_large_enough_to_stop_inward_drift and
  // char_speed_large_enough_to_stop_inward_drift below is to stop
  // the scenario in which either the CharSpeed or DeltaR approaches
  // the limit "min_allowed_radial_distance" and the timescale gets
  // cut down to a ridiculously small value.  When that scenario
  // happens, we simply exit state DeltaRNoDrift and go back to DeltaR.
  // These variables are called "DeltaRLargeEnoughToExitState4"
  // and "CharSpeedLargeEnoughToExitState4" in SpEC.
  constexpr double stop_inward_drift_buffer_factor = 0.99;
  const bool delta_r_large_enough_to_stop_inward_drift =
      update_args.min_allowed_radial_distance.has_value() and
      update_args.average_radial_distance.value() >
          stop_inward_drift_buffer_factor *
              update_args.min_allowed_radial_distance.value();
  const bool char_speed_large_enough_to_stop_inward_drift =
      update_args.min_allowed_char_speed.has_value() and
      update_args.min_char_speed >
          stop_inward_drift_buffer_factor *
              update_args.min_allowed_char_speed.value();
  return delta_r_large_enough_to_stop_inward_drift or
         char_speed_large_enough_to_stop_inward_drift;
}
}  // namespace control_system::size::States
