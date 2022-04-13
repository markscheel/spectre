// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Interpolation/Targets/WedgeSectionTorus.hpp"

#include <pup.h>

namespace intrp::OptionHolders {

WedgeSectionTorus::WedgeSectionTorus(
    const double min_radius_in, const double max_radius_in,
    const double min_theta_in, const double max_theta_in,
    const size_t number_of_radial_points_in,
    const size_t number_of_theta_points_in,
    const size_t number_of_phi_points_in, const bool use_uniform_radial_grid_in,
    const bool use_uniform_theta_grid_in, ::Verbosity verbosity_in,
    const Options::Context& context)
    : min_radius(min_radius_in),
      max_radius(max_radius_in),
      min_theta(min_theta_in),
      max_theta(max_theta_in),
      number_of_radial_points(number_of_radial_points_in),
      number_of_theta_points(number_of_theta_points_in),
      number_of_phi_points(number_of_phi_points_in),
      use_uniform_radial_grid(use_uniform_radial_grid_in),
      use_uniform_theta_grid(use_uniform_theta_grid_in),
      verbosity(verbosity_in) {
  if (min_radius >= max_radius) {
    PARSE_ERROR(context, "WedgeSectionTorus expects min_radius < max_radius");
  }
  if (min_theta >= max_theta) {
    PARSE_ERROR(context, "WedgeSectionTorus expects min_theta < max_theta");
  }
}

void WedgeSectionTorus::pup(PUP::er& p) {
  p | min_radius;
  p | max_radius;
  p | min_theta;
  p | max_theta;
  p | number_of_radial_points;
  p | number_of_theta_points;
  p | number_of_phi_points;
  p | use_uniform_radial_grid;
  p | use_uniform_theta_grid;
  p | verbosity;
}

bool operator==(const WedgeSectionTorus& lhs, const WedgeSectionTorus& rhs) {
  return lhs.min_radius == rhs.min_radius and
         lhs.max_radius == rhs.max_radius and lhs.min_theta == rhs.min_theta and
         lhs.max_theta == rhs.max_theta and
         lhs.number_of_radial_points == rhs.number_of_radial_points and
         lhs.number_of_theta_points == rhs.number_of_theta_points and
         lhs.number_of_phi_points == rhs.number_of_phi_points and
         lhs.use_uniform_radial_grid == rhs.use_uniform_radial_grid and
         lhs.use_uniform_theta_grid == rhs.use_uniform_theta_grid and
         lhs.verbosity == rhs.verbosity;
}

bool operator!=(const WedgeSectionTorus& lhs, const WedgeSectionTorus& rhs) {
  return not(lhs == rhs);
}

}  // namespace intrp::OptionHolders
