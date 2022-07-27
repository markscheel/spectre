// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/SizeControlStateDeltaR.hpp"

namespace SizeControlStates {
void DeltaR::update(const gsl::not_null<SizeControlInfo*> info,
                    const SizeControlStateUpdateArgs& update_args,
                    const CrossingTimeInfo& crossing_time_info) const {
  constexpr double delta_r_control_signal_threshold = 1.e-3;

  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different SizeControlStates.
  constexpr double time_tolerance_for_delta_r_in_danger = 0.99;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius <
          info->damping_time * time_tolerance_for_delta_r_in_danger;
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed < info->damping_time and
      not delta_radius_is_in_danger;

  if (char_speed_is_in_danger) {
    if (crossing_time_info.t_comoving_char_speed > 0.0 or
        update_args.min_comoving_char_speed < 0.0) {
      // Comoving char speed is negative or threatening to cross zero, so
      // staying in DeltaR mode will not work.  So switch to AhSpeed mode.

      // This factor prevents oscillating between states Initial and AhSpeed.
      constexpr double non_oscillation_factor = 1.01;
      info->discontinuous_change_has_occurred = true;
      info->state = SizeControlLabel::AhSpeed;
      info->target_char_speed =
          update_args.min_char_speed * non_oscillation_factor;
    }
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaR mode will rescue the speed
    // automatically (since it drives char speed to comoving char
    // speed).  But we should decrease the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius;
  } else if (update_args.min_comoving_char_speed > 0.0 and
             std::abs(update_args.control_error_delta_r) >
                 delta_r_control_signal_threshold) {
    constexpr double delta_r_state_decrease_factor = 0.99;
    info->suggested_time_scale =
        info->damping_time * delta_r_state_decrease_factor;
  }
  // TODO: State3, State4, State5 transitions.
}

double DeltaR::control_signal(
    const SizeControlInfo& /*info*/,
    const SizeControlStateControlSignalArgs& control_signal_args) const {
  return control_signal_args.control_error_delta_r;
}
}  // namespace SizeControlStates
