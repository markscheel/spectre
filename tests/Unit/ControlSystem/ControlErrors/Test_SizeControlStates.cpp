// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <memory>
#include <optional>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Params passed into each test.
struct TestParams {
  // These are reasonable values for quantities that won't change in
  // the various logic tests.
  const double original_target_char_speed{0.011};
  const double damping_time{0.1};
  const std::optional<double> average_radial_distance{
      0.01};  // This is what SpEC calls DeltaR.
  // The following means that the excision boundary radius in the grid frame
  // is 2.01.
  const double horizon_00{4.02 * sqrt(M_PI)};
  const double avg_distorted_normal_dot_unit_coord_vector{1.0};
  // Defaults are values for quantities that we will vary so that the
  // logic makes different decisions.
  double min_char_speed{0.01};
  double min_comoving_char_speed{-0.02};
  double control_err_delta_r{0.03};
  std::optional<double> max_allowed_radial_distance{1.e100};
  std::optional<double> min_allowed_radial_distance{0.08};
  std::optional<double> min_allowed_char_speed{0.08};
  std::optional<double> inward_drift_velocity{0.005};
  control_system::size::CrossingTimeInfo crossing_time_info{
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
};

template <typename InitialState, typename FinalState>
void do_test(const TestParams& test_params,
             const bool expected_discontinuous_change_has_occurred,
             const std::optional<double> expected_suggested_time_scale,
             const double expected_target_char_speed) {
  const std::string initial_state = InitialState{}.name();
  const std::string final_state = FinalState{}.name();
  CAPTURE(initial_state);
  CAPTURE(final_state);
  CAPTURE(expected_discontinuous_change_has_occurred);
  CAPTURE(expected_suggested_time_scale);
  CAPTURE(expected_target_char_speed);
  CAPTURE(test_params.original_target_char_speed);
  CAPTURE(test_params.damping_time);
  CAPTURE(test_params.min_char_speed);
  CAPTURE(test_params.min_comoving_char_speed);
  CAPTURE(test_params.control_err_delta_r);
  CAPTURE(test_params.crossing_time_info.char_speed_will_hit_zero_first);
  CAPTURE(
      test_params.crossing_time_info.horizon_will_hit_excision_boundary_first);
  CAPTURE(test_params.crossing_time_info.t_char_speed);
  CAPTURE(test_params.crossing_time_info.t_comoving_char_speed);
  CAPTURE(test_params.crossing_time_info.t_delta_radius);

  // Set reasonable values for quantities that won't change in the various
  // logic tests.
  const double target_drift_velocity = 0.001;
  const std::optional<double> original_suggested_time_scale = std::nullopt;
  const bool original_discontinuous_change_has_occurred = false;

  const control_system::size::StateUpdateArgs update_args{
      test_params.min_char_speed,
      test_params.min_comoving_char_speed,
      test_params.horizon_00,
      test_params.control_err_delta_r,
      test_params.average_radial_distance,
      test_params.max_allowed_radial_distance,
      test_params.avg_distorted_normal_dot_unit_coord_vector,
      test_params.inward_drift_velocity,
      test_params.min_allowed_radial_distance,
      test_params.min_allowed_char_speed};
  control_system::size::Info info{
      TestHelpers::test_factory_creation<control_system::size::State,
                                         InitialState>(initial_state),
      test_params.damping_time,
      test_params.original_target_char_speed,
      target_drift_velocity,
      original_suggested_time_scale,
      original_discontinuous_change_has_occurred};

  // Check serialization of info
  const auto info_copy = serialize_and_deserialize(info);
  CHECK_FALSE(info.state == nullptr);
  const auto info_copy2 = info_copy;
  CHECK_FALSE(info_copy2.state == nullptr);
  // Note that there is no equality operator for info.state, because the
  // state contains no data; so here we check that the state can be cast to
  // the type it should be.
  CHECK(dynamic_cast<InitialState*>(info_copy.state.get()) != nullptr);
  CHECK(info_copy.damping_time == info.damping_time);
  CHECK(info_copy.target_char_speed == info.target_char_speed);
  CHECK(info_copy.target_drift_velocity == info.target_drift_velocity);
  CHECK(info_copy.suggested_time_scale == info.suggested_time_scale);
  CHECK(info_copy.discontinuous_change_has_occurred ==
        info.discontinuous_change_has_occurred);

  auto state = info.state->get_clone();
  const std::string update_message = state->update(
      make_not_null(&info), update_args, test_params.crossing_time_info);

  // These messages are hardcoded in the states
  CHECK(update_message.find("Current state " + initial_state) !=
        std::string::npos);
  CHECK(update_message.find_last_of(final_state) != std::string::npos);

  CHECK(dynamic_cast<FinalState*>(info.state.get()) != nullptr);
  CHECK(info.damping_time == test_params.damping_time);
  CHECK(info.target_char_speed == expected_target_char_speed);
  CHECK(info.target_drift_velocity == target_drift_velocity);
  CHECK(info.suggested_time_scale == expected_suggested_time_scale);
  CHECK(info.discontinuous_change_has_occurred ==
        expected_discontinuous_change_has_occurred);

  info.reset();
  CHECK(info.damping_time == test_params.damping_time);
  CHECK(info.target_char_speed == expected_target_char_speed);
  CHECK(info.target_drift_velocity == target_drift_velocity);
  CHECK_FALSE(info.suggested_time_scale.has_value());
  CHECK_FALSE(info.discontinuous_change_has_occurred);
}

void test_size_control_update() {
  TestParams test_params;  // With reasonable default values.

  // The parameters of the tests below are chosen by hand so that the
  // union of all the tests hit all of the 'if' statements in all of
  // the control_system::size::State::update functions.
  //
  // Each of the tests below is also done in SpEC (with the same input
  // parameters and the same expected results), to ensure that SpEC
  // and SpECTRE have the same size control logic.

  // First we do tests of state control_system::size::Label::Initial.

  // should do nothing
  do_test<control_system::size::States::Initial,
          control_system::size::States::Initial>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // should go into DeltaR state
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero after damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 1.1 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.9 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaR>(
      test_params, true, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::Initial,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Trigger DeltaRDriftOutward by changing max_allowed_radial_distance
  test_params.max_allowed_radial_distance = 0.001;
  // Make sure nothing is in danger.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  // Comoving speed should be <0 or else we get state DeltaR and not
  // DeltaRDriftOutward.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::Initial,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  test_params.max_allowed_radial_distance = 1.e100;

  // Now do DeltaR tests
  test_params.min_comoving_char_speed = -0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should do nothing
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Should change suggested time scale
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.99 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Should do nothing
  test_params.control_err_delta_r = 1.e-4;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero *slightly* before damping time; should do
  // nothing (depends on tolerance in control_system::size::StateDeltaR).
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.999 * test_params.damping_time,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, faster than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, same as char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.9 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar cross zero before damping time, slower than char speed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaR>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Same crossing_time_info, but comoving_char_speed is negative.
  // Should have different result as previous test.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same as 2 tests ago, but comoving_char_speed will cross zero far
  // in the future.  Should be same result as previous test.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, 1.e12, 0.9 * test_params.damping_time,
      std::nullopt, std::nullopt);
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Trigger DeltaRDriftOutward by changing max_allowed_radial_distance
  test_params.max_allowed_radial_distance = 0.001;
  // Make sure nothing is in danger.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  // Comoving speed should be <0 or else we get state DeltaR and not
  // DeltaRDriftOutward.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::DeltaR,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
  test_params.max_allowed_radial_distance = 1.e100;

  // Now do AhSpeed tests
  test_params.min_comoving_char_speed = -0.02;
  test_params.control_err_delta_r = 0.03;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should do nothing.
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Should change to DeltaR state.
  test_params.min_comoving_char_speed = 0.02;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Should do nothing because min_comoving_char_speed is smaller than
  // min_char_speed.
  test_params.min_comoving_char_speed = 0.99 * test_params.min_char_speed;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now should change to DeltaR state if min_char_speed is larger than
  // target_char_speed
  test_params.min_char_speed = 0.012;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Now it should do nothing because comoving crossing time is very small.
  test_params.min_char_speed = 0.01;
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, 1.e-10, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now it should go to DeltaR because comoving crossing time is large.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, 100.0, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Now it should do nothing because comoving is decreasing faster than
  // charspeeds.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1000.0, 100.0, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Now it should think delta_r is in danger,
  // and it should go to DeltaR state.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 19.0 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, 19.0 * test_params.damping_time,
      test_params.original_target_char_speed);

  // But now with comoving_char_speed negative it should stay in AhSpeed
  // state, but with a change in target speed.
  test_params.min_comoving_char_speed = -0.02;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.damping_time,
      0.125 * test_params.min_char_speed);

  // With min_comoving_char_speed positive, it should still stay in
  // AhSpeed state if char_speed has a positive crossing time.
  test_params.min_comoving_char_speed = 0.02;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.e10, std::nullopt, 19.0 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, true, test_params.damping_time,
      0.125 * test_params.min_char_speed);

  // .. but not if the delta_r crossing time is small enough.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.e10, std::nullopt, 4.99 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, 4.99 * test_params.damping_time,
      test_params.original_target_char_speed);

  // If it thinks char speed is in danger, and the target char speed is
  // greater than the char speed, it changes the timescale and
  // nothing else.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.original_target_char_speed);

  // ...but in the same situation, if char speed is greater than the
  // target speed, it resets the target speed too.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.0001;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.89 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely* in danger.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.09999;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.98999 * test_params.damping_time, std::nullopt,
      0.99 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::AhSpeed>(
      test_params, false, 0.98999 * test_params.damping_time,
      test_params.min_char_speed * 1.01);

  // Same situation as previous, but char speed is *barely not* in danger,
  // and DeltaR is also not in danger.  Should go to DeltaR state.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.10001;
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Again char speed is *barely not* in danger, but for a different reason.
  test_params.min_char_speed = test_params.original_target_char_speed * 1.09999;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.99001 * test_params.damping_time, std::nullopt,
      0.992 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::AhSpeed,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);

  // Now do DeltaRDriftOutward tests
  test_params.min_comoving_char_speed = -0.02;
  test_params.control_err_delta_r = 0.03;
  test_params.max_allowed_radial_distance = 0.001;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

  // Should do nothing.
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero *slightly* before damping time; it should still do
  // nothing (depends on tolerance in control_system::size::DeltaRDriftOutward).
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.999 * test_params.damping_time,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make charspeed cross zero slightly after damping time; it should
  // still do nothing.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      1.001 * test_params.damping_time, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, std::nullopt, test_params.original_target_char_speed);

  // Make deltar cross zero before damping time. Now it should suggest
  // a new damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, 0.9 * test_params.damping_time, std::nullopt,
      std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar and charspeed cross zero before damping time, deltar
  // faster than char speed.  Should suggest new damping time.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.91 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaRDriftOutward>(
      test_params, false, 0.9 * test_params.damping_time,
      test_params.original_target_char_speed);

  // Make deltar and charspeed cross zero before damping time, deltar
  // slower than char speed.  Should go to AhSpeed.
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      0.89 * test_params.damping_time, std::nullopt,
      0.9 * test_params.damping_time, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::AhSpeed>(
      test_params, true, 0.89 * test_params.damping_time,
      1.01 * test_params.min_char_speed);

  // Should go to state DeltaR because distance < max_allowed_radial_distance.
  test_params.max_allowed_radial_distance = 1.e100;
  test_params.crossing_time_info = control_system::size::CrossingTimeInfo(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  do_test<control_system::size::States::DeltaRDriftOutward,
          control_system::size::States::DeltaR>(
      test_params, true, std::nullopt, test_params.original_target_char_speed);
}

void test_size_control_error() {
  // This is a very rudimentary test.  It just computes
  // the same thing as the thing it is testing, but coded differently.
  const control_system::size::ControlErrorArgs args{0.01, 0.03, 0.04, 1.2,
                                                    0.33};
  const control_system::size::Info info{
      std::make_unique<control_system::size::States::Initial>(),
      1.1,
      0.011,
      1.e-3,
      2.e-3,
      false};
  CHECK(control_system::size::States::Initial{}.control_error(info, args) ==
        -0.329);
  CHECK(control_system::size::States::AhSpeed{}.control_error(info, args) ==
        approx(0.001 * sqrt(4.0 * M_PI) / 1.2));
  CHECK(control_system::size::States::DeltaR{}.control_error(info, args) ==
        0.03);
  CHECK(control_system::size::States::DeltaRDriftOutward{}.control_error(
            info, args) == 0.04);
}

template <typename State>
void test_clone_and_serialization() {
  std::unique_ptr<control_system::size::State> state =
      std::make_unique<State>();

  // Note that we don't check equality here.  None of the derived
  // classes of control_system::size::State actually have data.
  // We just check that the types are correct.
  CHECK(dynamic_cast<State*>(serialize_and_deserialize(state).get()) !=
        nullptr);

  // Note that we don't check equality here.  None of the derived
  // classes of control_system::size::State actually have data.
  // We just check that the types are correct.
  CHECK(dynamic_cast<State*>(state->get_clone().get()) != nullptr);
}

void test_name_and_number() {
  const control_system::size::States::Initial initial{};
  const control_system::size::States::AhSpeed ah_speed{};
  const control_system::size::States::DeltaR delta_r{};
  const control_system::size::States::DeltaRDriftOutward
      delta_r_drift_outward{};

  CHECK(initial.name() == "Initial"s);
  CHECK(initial.number() == 0_st);
  CHECK(ah_speed.name() == "AhSpeed"s);
  CHECK(ah_speed.number() == 1_st);
  CHECK(delta_r.name() == "DeltaR"s);
  CHECK(delta_r.number() == 2_st);
  CHECK(delta_r_drift_outward.name() == "DeltaRDriftOutward"s);
  CHECK(delta_r_drift_outward.number() == 5_st);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.SizeControlStates", "[Domain][Unit]") {
  control_system::size::register_derived_with_charm();
  test_size_control_update();
  test_size_control_error();
  test_clone_and_serialization<control_system::size::States::Initial>();
  test_clone_and_serialization<control_system::size::States::AhSpeed>();
  test_clone_and_serialization<control_system::size::States::DeltaR>();
  test_clone_and_serialization<
      control_system::size::States::DeltaRDriftOutward>();
  test_name_and_number();
}
