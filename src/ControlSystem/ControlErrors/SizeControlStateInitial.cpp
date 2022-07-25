// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/SizeControlStateInitial.hpp"

namespace SizeControlStates {
void Initial::update(const gsl::not_null<SizeControlInfo*> info,
                     const SizeControlStateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const {
  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different SizeControlStates.
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed < info->damping_time;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius < info->damping_time and
      not char_speed_is_in_danger;

  // This factor is present in SpEC, but it probably isn't necessary
  // (but it doesn't hurt either).  We keep it here to facilitate
  // comparison with SpEC.  The value of 1.01 was chosen in SpEC, but
  // nothing should be sensitive to small changes in this value as long
  // as it is something slightly greater than unity.
  constexpr double non_oscillation_factor = 1.01;

  if (char_speed_is_in_danger) {
    info->discontinuous_change_has_occurred = true;
    info->state = SizeControlLabel::AhSpeed;
    info->target_char_speed =
        update_args.min_char_speed * non_oscillation_factor;
    info->suggested_time_scale = crossing_time_info.t_char_speed;
  } else if (delta_radius_is_in_danger) {
    info->discontinuous_change_has_occurred = true;
    info->state = SizeControlLabel::DeltaR;
    info->suggested_time_scale = crossing_time_info.t_delta_radius;
    // TODO: Add possible transition to State DeltaRDriftInward.
  } else if (update_args.min_comoving_char_speed > 0.0) {
    // Here the comoving speed is positive, so prefer DeltaR control.
    info->discontinuous_change_has_occurred = true;
    info->state = SizeControlLabel::DeltaR;
    // TODO: Add possible transition to State DeltaRDriftInward.
  }
  // Otherwise, no change.
}

double Initial::control_signal(
    const SizeControlInfo& info,
    const SizeControlStateControlSignalArgs& control_signal_args) const {
  // The return value is the Q that directly controls the speed of the
  // excision boundary in the distorted frame relative to the grid frame.
  return info.target_drift_velocity -
         control_signal_args.time_deriv_of_lambda_00;
}

}  // namespace SizeControlStates
