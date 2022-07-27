// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/ControlErrors/SizeControlInfo.hpp"
#include "Utilities/Gsl.hpp"

/// Packages some of the inputs to the SizeControlState::update, so
/// that SizeControlState::update doesn't need a large number of
/// arguments.
struct SizeControlStateUpdateArgs {
  /// min_char_speed is the minimum over the excision boundary
  /// of Eq. 89 of \cite Hemberger2012jz.
  double min_char_speed;
  /// min_comoving_char_speed is the minimum over the excision boundary
  /// of Eq. 28 of \cite Hemberger2012jz.
  double min_comoving_char_speed;
  /// control_error_delta_r is the control error when the control system
  /// is in state SizeControlLabel::DeltaR.
  /// This is Q in Eq. 96 of \cite Hemberger2012jz.
  double control_error_delta_r;
};

/// Packages some of the inputs to the SizeControlState::control_signal, so
/// that SizeControlState::control_signal doesn't need a large number of
/// arguments.
struct SizeControlStateControlSignalArgs {
  double min_char_speed;
  double control_error_delta_r;
  /// avg_distorted_normal_dot_unit_coord_vector is the average of
  /// distorted_normal_dot_unit_coord_vector over the excision
  /// boundary.  Here distorted_normal_dot_unit_coord_vector is Eq. 93
  /// of \cite Hemberger2012jz.  distorted_normal_dot_unit_coord_vector is
  /// \f$\hat{n}_i x^i/r\f$ where \f$\nat{n}_i\f$ is the
  /// distorted-frame unit normal to the excision boundary (pointing
  /// INTO the hole, i.e. out of the domain), and \f$x^i/r\f$ is the
  /// distorted-frame (or equivalently the grid frame because it is
  /// invariant between these two frames because of the required
  /// limiting behavior of the map we choose) Euclidean normal vector
  /// from the center of the excision-boundary Strahlkorper to each
  /// point on the excision-boundary Strahlkorper.
  double avg_distorted_normal_dot_unit_coord_vector;
  /// time_deriv_of_lambda_00 is the time derivative of the quantity lambda_00
  /// that appears in \cite Hemberger2012jz.  time_deriv_of_lambda_00 is (minus)
  /// the radial velocity of the excision boundary in the distorted frame with
  /// respect to the grid frame.
  double time_deriv_of_lambda_00;
};

/// Represents a 'state' of the size control system.
///
/// Each 'state' of the size control system has a different control
/// signal, which has a different purpose, even though each state
/// controls the same map quantity, namely the Y00 coefficient of the
/// shape map.  For example, state SizeControlLabel::AhSpeed controls
/// the Y00 coefficient of the shape map so that the minimum
/// characteristic speed is driven towards a target value, and state
/// SizeControlLabel::DeltaR controls the Y00 coefficient of the shape
/// map (or the Y00 coefficient of a separate spherically-symmetric size
/// map) so that the minimum difference between the horizon radius and
/// the excision boundary radius is driven towards a constant.
///
/// Each state has its own logic (the 'update' function) that
/// determines values of certain parameters (i.e. the things in
/// SizeControlInfo), including whether the control system should
/// transition to a different state.
class SizeControlState {
 public:
  virtual ~SizeControlState() = default;
  /// Updates the SizeControlInfo in `info`.  Notice that `info`
  /// includes a state, which might be different than the current
  /// state upon return. It is the caller's responsibility to check
  /// if the current state has changed.
  virtual void update(const gsl::not_null<SizeControlInfo*> info,
                      const SizeControlStateUpdateArgs& update_args,
                      const CrossingTimeInfo& crossing_time_info) const = 0;
  /// Returns the control signal, but does not modify the state or any
  /// parameters.
  virtual double control_signal(
      const SizeControlInfo& info,
      const SizeControlStateControlSignalArgs& control_signal_args) const = 0;
};
