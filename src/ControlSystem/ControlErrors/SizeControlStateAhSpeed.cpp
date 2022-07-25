// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/SizeControlStateAhSpeed.hpp"

#include <cmath>

namespace SizeControlStates {
void AhSpeed::update(const gsl::not_null<SizeControlInfo*> info,
                     const SizeControlStateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const {
  const double min_char_speed = update_args.min_char_speed;
  const double min_comoving_char_speed = update_args.min_comoving_char_speed;

  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different SizeControlStates.
  //
  // The value of 20 causes the control system to panic easily if
  // delta_radius is decreasing quickly.  The value 20 was chosen
  // by trial-and-error in SpEC.
  constexpr double time_tolerance_for_delta_r_in_danger = 20.0;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius <
          info->damping_time * time_tolerance_for_delta_r_in_danger;

  const bool char_speed_is_in_danger = [&crossing_time_info, &info,
                                        &delta_radius_is_in_danger,
                                        &min_char_speed]() {
    // speed_tolerance_for_char_speed_in_danger is slightly greater
    // than unity so that we don't panic if the actual char speed is
    // large enough compared to the target char speed.  The value 1.1
    // was chosen in SpEC and we haven't had to change it; the behavior
    // should not be sensitive to small changes in this value.
    constexpr double speed_tolerance_for_char_speed_in_danger = 1.1;
    // We don't want to panic unless crossing time is less than
    // the current damping time. The value 0.99 was chosen in SpEC and we
    // haven't had to change it; the behavior should not be sensitive to
    // small changes in this value.
    constexpr double time_tolerance_for_char_speed_in_danger = 0.99;
    if (delta_radius_is_in_danger) {
      return false;
    } else if (crossing_time_info.char_speed_will_hit_zero_first and
               crossing_time_info.t_char_speed <
                   info->damping_time *
                       time_tolerance_for_char_speed_in_danger and
               min_char_speed < info->target_char_speed *
                                    speed_tolerance_for_char_speed_in_danger) {
      return true;
    } else {
      return false;
    }
  }();

  const bool comoving_decreasing_slower_than_char_speeds =
      not(crossing_time_info.t_char_speed > 0.0 and
          crossing_time_info.t_comoving_char_speed > 0.0 and
          update_args.min_comoving_char_speed > 0.0 and
          update_args.min_comoving_char_speed /
                  crossing_time_info.t_comoving_char_speed >
              update_args.min_char_speed / crossing_time_info.t_char_speed);

  if (char_speed_is_in_danger) {
    constexpr double min_char_speed_increase_factor = 1.01;
    if (info->target_char_speed < min_char_speed) {
      // We are already in state AhSpeed, and we are in danger.
      // But target_char_speed is less than min_char_speed, so we don't want
      // to continue to push min_char_speed downward.  Instead, increase
      // target_char_speed to be above min_char_speed.
      // We don't do this if delta_radius_is_in_danger, because for that
      // case we might need to drive min_char_speed to a smaller value.
      info->target_char_speed = min_char_speed * min_char_speed_increase_factor;
    }
    info->suggested_time_scale = crossing_time_info.t_char_speed;
  } else if (delta_radius_is_in_danger) {
    constexpr double target_speed_decrease_factor = 0.125;
    constexpr double time_tolerance_for_delta_r =
        0.25 * time_tolerance_for_delta_r_in_danger;
    if ((crossing_time_info.t_char_speed > 0.0 and
         crossing_time_info.t_delta_radius >
             info->damping_time * time_tolerance_for_delta_r) or
        update_args.min_comoving_char_speed < 0.0) {
      info->discontinuous_change_has_occurred = true;
      info->target_char_speed = min_char_speed * target_speed_decrease_factor;
      info->suggested_time_scale =
          std::min(info->damping_time, crossing_time_info.t_delta_radius);
    } else {
      info->discontinuous_change_has_occurred = true;
      info->state = SizeControlLabel::DeltaR;
      info->suggested_time_scale = crossing_time_info.t_delta_radius;
      // TODO: PreferState3OverState2 goes here.
    }
  } else if (update_args.min_comoving_char_speed > 0.0 and
             update_args.min_char_speed > 0.0 and
             (crossing_time_info.t_comoving_char_speed == 0.0 or
              (crossing_time_info.t_comoving_char_speed >
                   5.0 * info->damping_time and
               comoving_decreasing_slower_than_char_speeds)) and
             (update_args.min_char_speed >= info->target_char_speed or
              min_comoving_char_speed > min_char_speed)) {
    info->discontinuous_change_has_occurred = true;
    info->state = SizeControlLabel::DeltaR;
    // TODO: PreferState3OverState2 goes here.
  }
  // If no 'if's are encountered above, then all the info parameters stay
  // the same as they were.
}

double AhSpeed::control_signal(
    const SizeControlInfo& info,
    const SizeControlStateControlSignalArgs& control_signal_args) const {
  const double Y00 = sqrt(0.25 / M_PI);
  // The return value is Q from Eq. 92 of ArXiv:1211.6079.
  return (info.target_char_speed - control_signal_args.min_char_speed) /
         (Y00 * control_signal_args.avg_distorted_normal_dot_unit_coord_vector);
}
}  // namespace SizeControlStates
