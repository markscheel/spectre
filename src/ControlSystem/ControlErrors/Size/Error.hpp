// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace control_system::size {
class Info;
}  // namespace control_system::size
namespace intrp {
class ZeroCrossingPredictor;
}  // namespace intrp
/// \endcond

namespace control_system::size {
/*!
 * \brief A simple struct to hold diagnostic information about computing the
 * size control error.
 */
struct ErrorDiagnostics {
  double control_error;
  size_t state_number;
  double min_delta_r;
  double min_relative_delta_r;
  double min_comoving_char_speed;
  double char_speed_crossing_time;
  double comoving_char_speed_crossing_time;
  double delta_r_crossing_time;
  double target_char_speed;
  double suggested_timescale;
  double damping_timescale;
  ControlErrorArgs control_error_args;
  std::string update_message;
  bool discontinuous_change_has_occurred;
};

/*!
 * \brief Computes the size control error, updating the stored info.
 *
 * \tparam Frame should be ::Frame::Distorted if ::Frame::Distorted exists.
 * \param info struct containing parameters that will be used/filled. Some of
 *        the fields in info will guide the behavior of other control system
 *        components like the averager and (possibly) the time step.
 * \param predictor_char_speed ZeroCrossingPredictor for the characteristic
 *          speed.
 * \param predictor_comoving_char_speed ZeroCrossingPredictor for the
 *        comoving characteristic speed.
 * \param predictor_delta_radius ZeroCrossingPredictor for the difference
 *        in radius between the horizon and the excision boundary.
 * \param time the current time.
 * \param control_error_delta_r the control error for the DeltaR state. This is
 *        used in other states as well.
 * \param control_error_delta_r_outward the control error for the
 *        DeltaRDriftOutward state.  If std::nullopt, then DeltaRDriftOutward
 *        will not be used.
 * \param max_allowed_radial_distance the maximum average radial distance
 *        between the horizon and the excision boundary that is allowed without
 *        triggering the DeltaRDriftOutward state.  If std::nullopt, then
 *        DeltaRDriftOutward will not be used.
 * \param dt_lambda_00 the time derivative of the map parameter lambda_00
 * \param apparent_horizon the current horizon in frame Frame.
 * \param excision_boundary a Strahlkorper representing the excision
 *        boundary in frame Frame.  Note that the excision boundary is assumed
 *        to be a sphere in the grid frame.
 * \param lapse_on_excision_boundary Lapse on the excision boundary.
 * \param frame_components_of_grid_shift The quantity
 *        \f$\beta^i \frac{\partial x^\hat{i}}{\partial x_i}\f$ (see below)
 *        evaluated on the excision boundary.  This is a tensor in frame
 *        Frame.
 * \param spatial_metric_on_excision_boundary metric in frame Frame.
 * \param inverse_spatial_metric_on_excision_boundary metric in frame Frame.
 * \return Returns an `ErrorDiagnostics` object which, in addition to the actual
 *         control error, holds a lot of diagnostic information about how the
 *         control error was calculated. This information could be used to print
 *         to a file if desired.
 *
 * The characteristic speed that is needed here is
 * \f{align}
 *     v &= -\alpha -n_i\beta^i \\
 *     v &= -\alpha -n_\hat{i}\hat{\beta}^\hat{i}
 *           - n_\hat{i}\frac{\partial x^\hat{i}}{\partial t} \\
 *     v &= -\alpha -n_\bar{i}\bar{\beta}^\bar{i}
 *           - n_\bar{i}\frac{\partial x^\bar{i}}{\partial t} \\
 *     v &= -\alpha - n_\hat{i}
 *          \frac{\partial x^\hat{i}}{\partial x^i} \beta^i,
 *  \f}
 *  where we have written many equivalent forms in terms of quantities
 *  defined in different frames.
 *
 *  Here \f$\alpha\f$ is the lapse, which is invariant under frame
 *  transformations, \f$n_i\f$, \f$n_\hat{i}\f$, and \f$n_\bar{i}\f$
 *  are the metric-normalized normal one-form to the Strahlkorper in the
 *  grid, distorted, and inertial frames, and
 *  \f$\beta^i\f$, \f$\hat{\beta}^\hat{i}\f$, and \f$\bar{\beta}^\bar{i}\f$
 *  are the shift in the grid, distorted, and inertial frames.
 *
 *  Note that we decorate the shift with hats and bars in addition to
 *  decorating its index, because the shift transforms in a non-obvious
 *  way under frame transformations so it is easy to make mistakes.
 *  To be clear, these different shifts are defined by
 * \f{align}
 *   \beta^i &= \alpha^2 g^{0i},\\
 *   \hat{\beta}^\hat{i} &= \alpha^2 g^{\hat{0}\hat{i}},\\
 *   \bar{\beta}^\bar{i} &= \alpha^2 g^{\bar{0}\bar{i}},
 * \f}
 *  where \f$g^{ab}\f$ is the spacetime metric, and they transform like
 * \f{align}
 * \hat{\beta}^\hat{i} &= \beta^i \frac{\partial x^\hat{i}}{\partial x^i}-
 *  \frac{\partial x^\hat{i}}{\partial t}.
 * \f}
 *
 * The quantity we pass as frame_components_of_grid_shift is
 * \f{align}
 * \beta^i \frac{\partial x^\hat{i}}{\partial x^i}
 * &= \hat{\beta}^\hat{i} + \frac{\partial x^\hat{i}}{\partial t} \\
 * &= \bar{\beta}^\bar{j}\frac{\partial x^\hat{i}}{\partial x^i}
 *    \frac{\partial x^i}{\partial x^\bar{j}} +
 *    \frac{\partial x^\hat{i}}{\partial x^\bar{j}}
 *    \frac{\partial x^\bar{j}}{\partial t},
 * \f}
 * where we have listed several equivalent formulas that involve quantities
 * in different frames.
 */
template <typename Frame>
ErrorDiagnostics control_error(
    const gsl::not_null<Info*> info,
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*>
        predictor_comoving_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*> predictor_delta_radius,
    const gsl::not_null<intrp::ZeroCrossingPredictor*>
        predictor_drift_limit_char_speed,
    const gsl::not_null<intrp::ZeroCrossingPredictor*>
        predictor_drift_limit_delta_radius,
    double time, double control_error_delta_r,
    std::optional<double> control_error_delta_r_outward,
    std::optional<double> max_allowed_radial_distance,
    std::optional<double> inward_drift_velocity,
    std::optional<double> min_allowed_radial_distance,
    std::optional<double> min_allowed_char_speed, double horizon_00,
    double dt_lambda_00, const ylm::Strahlkorper<Frame>& apparent_horizon,
    const ylm::Strahlkorper<Frame>& excision_boundary,
    const Scalar<DataVector>& lapse_on_excision_boundary,
    const tnsr::I<DataVector, 3, Frame>& frame_components_of_grid_shift,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric_on_excision_boundary,
    const tnsr::II<DataVector, 3, Frame>&
        inverse_spatial_metric_on_excision_boundary);

}  // namespace control_system::size
