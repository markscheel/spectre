// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/DeltaRDriftInward.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftInwardHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace control_system::size::States {

std::unique_ptr<State> DeltaRDriftInward::get_clone() const {
  return std::make_unique<DeltaRDriftInward>(*this);
}

std::string DeltaRDriftInward::update(
    const gsl::not_null<Info*> info, const StateUpdateArgs& update_args,
    const CrossingTimeInfo& crossing_time_info) const {
  const double Y00 = 0.25 * M_2_SQRTPI;

  // This factor is present in SpEC, and it is used to prevent
  // oscillations between states.  The value was chosen in SpEC, but
  // nothing should be sensitive to small changes in this value as
  // long as it is slightly greater than unity.
  constexpr double non_oscillation_drift_inward_factor = 1.1;

  // Note that delta_radius_is_in_danger and char_speed_is_in_danger
  // can be different for different States.

  // The value of 0.99 was chosen by trial and error in SpEC.
  // It should be slightly less than unity but nothing should be
  // sensitive to small changes in this value.
  constexpr double time_tolerance_for_delta_r_in_danger = 0.99;
  const bool delta_radius_is_in_danger =
      crossing_time_info.horizon_will_hit_excision_boundary_first and
      crossing_time_info.t_delta_radius.value_or(
          std::numeric_limits<double>::infinity()) <
          info->damping_time * time_tolerance_for_delta_r_in_danger;
  const bool char_speed_is_in_danger =
      crossing_time_info.char_speed_will_hit_zero_first and
      crossing_time_info.t_char_speed.value_or(
          std::numeric_limits<double>::infinity()) < info->damping_time and
      not delta_radius_is_in_danger;

  constexpr double inward_drift_limit_buffer_factor = 0.9;
  const bool delta_r_almost_above_inward_drift_limit =
      update_args.min_allowed_radial_distance.has_value() and
      average_radial_distance >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_radial_distance.value();
  const bool char_speed_almost_above_inward_drift_limit =
      update_args.min_allowed_char_speed.has_value() and
      update_args.min_char_speed >
          inward_drift_limit_buffer_factor *
              update_args.min_allowed_char_speed.value();

  std::stringstream ss{};

  if (char_speed_is_in_danger) {
    ss << "Current state DeltaRDriftInward. Char speed in danger."
       << " Switching to AhSpeed.\n";
    // Switch to AhSpeed mode. Note that we don't check ComovingCharSpeed
    // like we do in state DeltaR; this behavior agrees with SpEC.

    // This factor prevents oscillating between states Initial and
    // AhSpeed.  It needs to be slightly greater than unity, but the
    // control system should not be sensitive to the exact
    // value. The value of 1.01 was chosen arbitrarily in SpEC and
    // never needed to be changed.
    constexpr double non_oscillation_factor = 1.01;
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::AhSpeed>();
    info->target_char_speed =
        update_args.min_char_speed * non_oscillation_factor;
    ss << " Target char speed = " << info->target_char_speed << "\n";
    // If the comoving char speed is positive and is not about to
    // cross zero, staying in DeltaRDriftInward mode will rescue the
    // speed automatically (since it drives char speed to comoving
    // char speed, plus a small difference).  But we should decrease
    // the timescale in any case.
    info->suggested_time_scale = crossing_time_info.t_char_speed.value();
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (delta_radius_is_in_danger) {
    info->suggested_time_scale = crossing_time_info.t_delta_radius.value();
    ss << "Current state DeltaRDriftInward. Delta radius in danger. Staying "
          "in DeltaRDriftInward.\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (should_transition_from_state_inward_drift_to_delta_r_no_drift(
                 crossing_time_info.t_drift_limit.value(), info->damping_time,
                 update_args.inward_drift_velocity,
                 delta_r_almost_above_inward_drift_limit,
                 char_speed_almost_above_inward_drift_limit,
                 comoving_char_speed_increasing_inward)) {
    ss << "Current state DeltaRDriftInward. Switching to DeltaRNoDrift.\n";
    info->discontinuous_change_has_occurred = true;
    info->state = std::make_unique<States::DeltaRNoDrift>();
    info->suggested_time_scale =
        crossing_time_info.t_drift_limit.value_or(info->damping_time);
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (crossing_time_info.t_delta_radius.has_value() and
             info->damping_time > 2.0 * update_args.horizon_00 * Y00) {
    // Do we need spherepack_factor above?
    ss << "Current state DeltaRDriftInward. RelativeDeltaR is decreasing, "
          "which is probably because timescale is too big (DeltaRDriftInward "
          "should be increasing RelativeDeltaR if control system is working "
          "properly). Decreasing timescale and staying in DeltaRDriftInward.\n";
    constexpr double delta_r_drift_inward_decrease_factor = 0.99;  // Arbitrary
    info->suggested_time_scale =
        info->damping_time * delta_r_drift_inward_decrease_factor;
    info->target_char_speed = target_speed_for_inward_drift(
        update_args.inward_drift_velocity.value(), update_args.min_char_speed,
        update_args.avg_distorted_normal_dot_unit_coord_vector);
    ss << " Target char speed = " << info->target_char_speed << "\n";
    ss << " Suggested timescale = " << info->suggested_time_scale;
  } else if (update_args.average_radial_distance.has_value() and
             update_args.average_radial_distance.value() >
                 non_oscillation_drift_inward_factor *
                     update_args.max_allowed_radial_distance.value()) {
    info->discontinuous_change_has_occurred = true;
    ss << "Current state DeltaRDriftInward. We have drifted too far, so "
          "we are switching to DeltaRDriftOutward.\n";
    info->state = std::make_unique<States::DeltaRDriftOutward>();
  } else {
    ss << "Current state DeltaRDriftInward. No change necessary. Staying in "
          "DeltaRDriftInward.";
  }

  return ss.str();
}

double DeltaRDriftInward::control_error(
    const Info& info, const ControlErrorArgs& control_error_args) const {
  // We increase the control error by the target speed, so as to make
  // control_error_delta_r more negative, which gives a negative velocity
  // to delta_r (i.e. a positive velocity to the excision boundary).
  return control_error_args.control_error_delta_r + info->target_char_speed;
}

PUP::able::PUP_ID DeltaRDriftInward::my_PUP_ID = 0;
}  // namespace control_system::size::States
