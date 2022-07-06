// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/SizeControlInfo.hpp"

CrossingTimeInfo::CrossingTimeInfo(
    const double char_speed_crossing_time,
    const double comoving_char_speed_crossing_time,
    const double delta_radius_crossing_time)
    : t_char_speed(char_speed_crossing_time),
      t_comoving_char_speed(comoving_char_speed_crossing_time),
      t_delta_radius(delta_radius_crossing_time) {
  if (t_char_speed > 0.0) {
    if (t_delta_radius > 0.0 and t_delta_radius <= t_char_speed) {
      horizon_will_hit_excision_boundary_first = true;
    } else {
      char_speed_will_hit_zero_first = true;
    }
  } else if (t_delta_radius > 0.0) {
    horizon_will_hit_excision_boundary_first = true;
  }
}
