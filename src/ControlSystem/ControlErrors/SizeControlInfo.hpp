// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

/// Labels the different 'state's of size control.
///
/// The different states are:
/// - Initial: drives dr/dt of the excision boundary to
///   SizeControlInfo::target_drift_velocity.
/// - AhSpeed: drives the minimum characteristic speed on the excision boundary
///   to SizeControlInfo::target_char_speed.
/// - DeltaR: drives the minimum distance between the horizon and the excision
///   boundary to be constant in time.
/// - DeltaRDriftInward: Same as DeltaR but the excision boundary has a small
///   velocity inward.  This state is triggered when it is deemed that the
///   excision boundary and the horizon are too close to each other; the
///   small velocity makes the excision boundary and the horizon drift apart.
/// - DeltaRDriftOutward: Same as DeltaR but the excision boundary has a small
///   velocity outward.  This state is triggered when it is deemed that the
///   excision boundary and the horizon are too far apart.
/// - DeltaRTransition: Same as DeltaR except for the logic that
///   determines how DeltaRTransition changes to other states.
///   DeltaRTransition is allowed (under some circumstances) to change
///   to state DeltaR, but DeltaRDriftOutward and DeltaRDriftInward
///   are never allowed to change to state DeltaR.  Instead
///   DeltaRDriftOutward and DeltaRDriftInward are allowed (under
///   some circumstances) to change to state DeltaRTransition.
///
/// The reason that DeltaRDriftInward, DeltaRDriftOutward, and
/// DeltaRTransition are separate states is to simplify the logic.  In
/// principle, all 3 of those states could be merged with state
/// DeltaR, because the control error is the same for all four states
/// (except for a velocity term that could be set to zero).  But if that
/// were done, then there would need to be additional complicated
/// logic in determining transitions between different states, and
/// that logic would depend not only on the current state, but also on
/// the previous state.
enum class SizeControlLabel : size_t {
  Initial,
  AhSpeed,
  DeltaR,
  DeltaRDriftInward,
  DeltaRDriftOutward,
  DeltaRTransition
};

/// Holds information that is saved between calls of SizeControl.
struct SizeControlInfo {
  // SizeControlInfo needs to be serializable because it will be
  // stored inside of a ControlError.
  void pup(PUP::er& p) {
    if(p.isUnpacking()) {
      size_t state_as_size_t;
      p | state_as_size_t;
      state = static_cast<SizeControlLabel>(state_as_size_t);
    } else {
      auto state_as_size_t = static_cast<size_t>(state);
      p | state_as_size_t;
    }
    p | damping_time;
    p | target_char_speed;
    p | target_drift_velocity;
    p | suggested_time_scale;
    p | discontinuous_change_has_occurred;
  }

  SizeControlLabel state{SizeControlLabel::Initial};
  /// The current damping time associated with size control.
  double damping_time;
  /// target_char_speed is what the characteristic speed is driven
  /// toward in state SizeControlLabel::AhSpeed.
  double target_char_speed;
  /// target_drift_velocity is what dr/dt (where r and t are distorted frame
  /// variables) of the excision boundary is driven toward in state
  /// SizeControlLabel::Initial.
  double target_drift_velocity;
  /// Sometimes SizeControlState::update will request that damping_time
  /// be changed; the new suggested value is suggested_time_scale.
  double suggested_time_scale;
  /// discontinuous_change_has_occurred is set to true by
  /// SizeControlState::update if it changes anything in such a way that
  /// the control signal jumps discontinuously in time.
  bool discontinuous_change_has_occurred;
};

/// Holds information about crossing times, as computed by
/// ZeroCrossingPredictors.
struct CrossingTimeInfo {
  CrossingTimeInfo(const double char_speed_crossing_time,
                   const double comoving_char_speed_crossing_time,
                   const double delta_radius_crossing_time);
  /// t_char_speed is the time (relative to the current time) when the
  /// minimum characteristic speed is predicted to cross zero (or zero if
  /// the minimum characteristic speed is increasing).
  double t_char_speed;
  /// t_comoving_char_speed is the time (relative to the current time) when the
  /// minimum comoving characteristic speed is predicted to cross zero
  /// (or zero if the minimum comoving characteristic speed is increasing).
  double t_comoving_char_speed;
  /// t_delta_radius is the time (relative to the current time) when the
  /// minimum distance between the horizon and the excision boundary is
  /// predicted to cross zero (or zero if the minimum distance is
  /// increasing).
  double t_delta_radius;
  /// Extra variables to simplify the logic; these indicate whether
  /// the characteristic speed or the excision boundary (or neither) are
  /// expected to cross zero soon.
  bool char_speed_will_hit_zero_first{false};
  bool horizon_will_hit_excision_boundary_first{false};
};
