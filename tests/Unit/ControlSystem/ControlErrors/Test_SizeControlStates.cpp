// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "ControlSystem/ControlErrors/SizeControlInfo.hpp"
#include "ControlSystem/ControlErrors/SizeControlState.hpp"
#include "ControlSystem/ControlErrors/SizeControlStateAhSpeed.hpp"
#include "ControlSystem/ControlErrors/SizeControlStateDeltaR.hpp"
#include "ControlSystem/ControlErrors/SizeControlStateInitial.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_size_control_update_work(
    const gsl::not_null<SizeControlInfo*> info,
    const SizeControlInfo& expected_info,
    const SizeControlState& size_control_state,
    const SizeControlStateUpdateArgs& update_args,
    const CrossingTimeInfo& crossing_time_info) {
  size_control_state.update(info, update_args, crossing_time_info);
  CHECK(info->state == expected_info.state);
  CHECK(info->damping_time == expected_info.damping_time);
  CHECK(info->target_char_speed == expected_info.target_char_speed);
  CHECK(info->target_drift_velocity == expected_info.target_drift_velocity);
  CHECK(info->suggested_time_scale == expected_info.suggested_time_scale);
  CHECK(info->discontinuous_change_has_occurred ==
        expected_info.discontinuous_change_has_occurred);
}

void test_size_control_update() {
  // Set reasonable values for quantities that won't change in the various
  // logic tests, but are needed below.
  const double original_target_char_speed = 0.011;
  const double damping_time = 0.1;

  // Values for quantities that we will vary so that the logic makes
  // different decisions.
  double min_char_speed = 0.01;
  double min_comoving_char_speed = -0.02;
  double control_err_delta_r = 0.03;

  // Set all the crossing times to zero (i.e. they never cross zero).
  CrossingTimeInfo crossing_time_info(0.0, 0.0, 0.0);

  // Set up initial state.
  auto initial_state = SizeControlLabel::Initial;

  auto do_test = [&initial_state, &damping_time, &min_comoving_char_speed,
                  &control_err_delta_r, &crossing_time_info, &min_char_speed,
                  &original_target_char_speed](
                     const SizeControlLabel final_state,
                     const bool discontinuous_change_has_occurred,
                     const double suggested_time_scale,
                     const double target_char_speed) {
    // Set reasonable values for quantities that won't change in the various
    // logic tests.
    const double target_drift_velocity = 0.001;
    const double original_suggested_time_scale = 0.0;
    const bool original_discontinuous_change_has_occurred = false;

    const SizeControlStateUpdateArgs update_args{
        min_char_speed, min_comoving_char_speed, control_err_delta_r};
    SizeControlInfo info{initial_state,
                         damping_time,
                         original_target_char_speed,
                         target_drift_velocity,
                         original_suggested_time_scale,
                         original_discontinuous_change_has_occurred};
    SizeControlInfo expected_info = info;
    expected_info.state = final_state;
    expected_info.discontinuous_change_has_occurred =
        discontinuous_change_has_occurred;
    expected_info.suggested_time_scale = suggested_time_scale;
    expected_info.target_char_speed = target_char_speed;

    auto size_control_state =
        [&initial_state]() -> std::unique_ptr<SizeControlState> {
      switch (initial_state) {
        case SizeControlLabel::AhSpeed:
          return std::make_unique<SizeControlStates::AhSpeed>();
        case SizeControlLabel::DeltaR:
          return std::make_unique<SizeControlStates::DeltaR>();
        default:
          return std::make_unique<SizeControlStates::Initial>();
      }
    }();

    test_size_control_update_work(make_not_null(&info), expected_info,
                                  *size_control_state, update_args,
                                  crossing_time_info);
  };

  // The parameters of the tests below are chosen by hand so that the
  // union of all the tests hit all of the 'if' statements in all of
  // the SizeControlState::update functions.
  //
  // Each of the tests below is also done in SpEC (with the same input
  // parameters and the same expected results), to ensure that SpEC
  // and SpECTRE have the same size control logic.

  // First we do tests of state SizeControlLabel::Initial.

  // should do nothing
  do_test(SizeControlLabel::Initial, false, 0.0, original_target_char_speed);

  // should go into DeltaR state
  min_comoving_char_speed = 0.02;
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Make deltar cross zero after damping time.
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 1.1 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Make deltar cross zero before damping time.
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  crossing_time_info =
      CrossingTimeInfo(0.91 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  crossing_time_info =
      CrossingTimeInfo(0.9 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  crossing_time_info =
      CrossingTimeInfo(0.89 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::AhSpeed, true, 0.89 * damping_time,
          min_char_speed * 1.01);

  // Now do DeltaR tests
  initial_state = SizeControlLabel::DeltaR;
  min_comoving_char_speed = -0.02;
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 0.0);

  // Should do nothing
  do_test(SizeControlLabel::DeltaR, false, 0.0, original_target_char_speed);

  // Should change suggested time scale
  min_comoving_char_speed = 0.02;
  do_test(SizeControlLabel::DeltaR, false, 0.99 * damping_time,
          original_target_char_speed);

  // Should do nothing
  control_err_delta_r = 1.e-4;
  do_test(SizeControlLabel::DeltaR, false, 0.0, original_target_char_speed);

  // Make deltar cross zero *slightly* before damping time; should do
  // nothing (depends on tolerance in SizeControlStateDeltaR).
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 0.999 * damping_time);
  do_test(SizeControlLabel::DeltaR, false, 0.0, original_target_char_speed);

  // Make deltar cross zero before damping time.
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, false, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  crossing_time_info =
      CrossingTimeInfo(0.91 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, false, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  crossing_time_info =
      CrossingTimeInfo(0.9 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, false, 0.9 * damping_time,
          original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  crossing_time_info =
      CrossingTimeInfo(0.89 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::DeltaR, false, 0.89 * damping_time,
          original_target_char_speed);

  // Same crossing_time_info, but comoving_char_speed is negative.
  // Should have different result as previous test.
  min_comoving_char_speed = -0.02;
  do_test(SizeControlLabel::AhSpeed, true, 0.89 * damping_time,
          min_char_speed * 1.01);

  // Same as 2 tests ago, but comoving_char_speed will cross zero far
  // in the future.  Should be same result as previous test.
  crossing_time_info =
      CrossingTimeInfo(0.89 * damping_time, 1.e12, 0.9 * damping_time);
  min_comoving_char_speed = 0.02;
  do_test(SizeControlLabel::AhSpeed, true, 0.89 * damping_time,
          min_char_speed * 1.01);

  // Now do AhSpeed tests
  initial_state = SizeControlLabel::AhSpeed;
  min_comoving_char_speed = -0.02;
  control_err_delta_r = 0.03;
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 0.0);

  // Should do nothing.
  do_test(SizeControlLabel::AhSpeed, false, 0.0, original_target_char_speed);

  // Should change to DeltaR state.
  min_comoving_char_speed = 0.02;
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Should do nothing because min_comoving_char_speed is smaller than
  // min_comoving_char_speed.
  min_comoving_char_speed = 0.99 * min_char_speed;
  do_test(SizeControlLabel::AhSpeed, false, 0.0, original_target_char_speed);

  // Now should change to DeltaR state if min_char_speed is larger than
  // target_char_speed
  min_char_speed = 0.012;
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Now it should do nothing because comoving crossing time is very small.
  min_char_speed = 0.01;
  min_comoving_char_speed = 0.02;
  crossing_time_info = CrossingTimeInfo(0.0, 1.e-10, 0.0);
  do_test(SizeControlLabel::AhSpeed, false, 0.0, original_target_char_speed);

  // Now it should go to DeltaR because comoving crossing time is large.
  crossing_time_info = CrossingTimeInfo(0.0, 100.0, 0.0);
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Now it should do nothing because comoving is decreasing faster than
  // charspeeds.
  crossing_time_info = CrossingTimeInfo(1000.0, 100.0, 0.0);
  do_test(SizeControlLabel::AhSpeed, false, 0.0, original_target_char_speed);

  // Now it should think delta_r is in danger,
  // and it should go to DeltaR state.
  crossing_time_info = CrossingTimeInfo(0.0, 0.0, 19.0 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 19.0 * damping_time,
          original_target_char_speed);

  // But now with comoving_char_speed negative it should stay in AhSpeed
  // state, but with a change in target speed.
  min_comoving_char_speed = -0.02;
  do_test(SizeControlLabel::AhSpeed, true, damping_time,
          0.125 * min_char_speed);

  // With min_comoving_char_speed positive, it should still stay in
  // AhSpeed state if char_speed has a positive crossing time.
  min_comoving_char_speed = 0.02;
  crossing_time_info = CrossingTimeInfo(1.e10, 0.0, 19.0 * damping_time);
  do_test(SizeControlLabel::AhSpeed, true, damping_time,
          0.125 * min_char_speed);

  // .. but not if the delta_r crossing time is small enough.
  crossing_time_info = CrossingTimeInfo(1.e10, 0.0, 4.99 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 4.99 * damping_time,
          original_target_char_speed);

  // If it thinks char speed is in danger, and the target char speed is
  // greater than the char speed, it changes the timescale and
  // nothing else.
  crossing_time_info =
      CrossingTimeInfo(0.89 * damping_time, 0.0, 0.9 * damping_time);
  do_test(SizeControlLabel::AhSpeed, false, 0.89 * damping_time,
          original_target_char_speed);

  // ...but in the same situation, if char speed is greater than the
  // target speed, it resets the target speed too.
  min_char_speed = original_target_char_speed * 1.0001;
  do_test(SizeControlLabel::AhSpeed, false, 0.89 * damping_time,
          min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely* in danger.
  min_char_speed = original_target_char_speed * 1.09999;
  crossing_time_info =
      CrossingTimeInfo(0.98999 * damping_time, 0.0, 0.99 * damping_time);
  do_test(SizeControlLabel::AhSpeed, false, 0.98999 * damping_time,
          min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely not* in danger,
  // and DeltaR is also not in danger.  Should go to DeltaR state.
  min_char_speed = original_target_char_speed * 1.10001;
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);

  // Again char speed is *barely not* in danger, but for a different reason.
  min_char_speed = original_target_char_speed * 1.09999;
  crossing_time_info =
      CrossingTimeInfo(0.99001 * damping_time, 0.0, 0.992 * damping_time);
  do_test(SizeControlLabel::DeltaR, true, 0.0, original_target_char_speed);
}

void test_size_control_signal() {
  // This is a very rudimentary test.  It just computes
  // the same thing as the thing it is testing, but coded differently.
  const SizeControlStateControlSignalArgs args{0.01, 0.03, 1.2, 0.33};
  const SizeControlInfo info{
      SizeControlLabel::Initial, 1.1, 0.011, 1.e-3, 2.e-3, false};
  CHECK(SizeControlStates::Initial{}.control_signal(info, args) == -0.329);
  CHECK(SizeControlStates::AhSpeed{}.control_signal(info, args) ==
        approx(0.001 * sqrt(4.0 * M_PI) / 1.2));
  CHECK(SizeControlStates::DeltaR{}.control_signal(info, args) == 0.03);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.SizeControlStates", "[Domain][Unit]") {
  test_size_control_update();
  test_size_control_signal();
}
