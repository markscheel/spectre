// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/SphericalCompression.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators::time_dependence {

SphericalCompression::SphericalCompression(
    const double initial_time, const double min_radius, const double max_radius,
    const std::array<double, 3> center, const double initial_value,
    const double initial_velocity, const double initial_acceleration,
    const Options::Context& context)
    : initial_time_(initial_time),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center),
      initial_value_grid_to_distorted_(initial_value),
      initial_velocity_grid_to_distorted_(initial_velocity),
      initial_acceleration_grid_to_distorted_(initial_acceleration) {
  if (min_radius >= max_radius) {
    PARSE_ERROR(context,
                "Tried to create a SphericalCompression TimeDependence, but "
                "the minimum radius ("
                    << min_radius << ") is not less than the maximum radius ("
                    << max_radius << ")");
  }
}

SphericalCompression::SphericalCompression(
    const double initial_time, const double min_radius, const double max_radius,
    const std::array<double, 3> center,
    const double initial_value_grid_to_distorted,
    const double initial_velocity_grid_to_distorted,
    const double initial_acceleration_grid_to_distorted,
    const double initial_value_distorted_to_inertial,
    const double initial_velocity_distorted_to_inertial,
    const double initial_acceleration_distorted_to_inertial,
    const Options::Context& context)
    : initial_time_(initial_time),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center),
      initial_value_grid_to_distorted_(initial_value_grid_to_distorted),
      initial_velocity_grid_to_distorted_(initial_velocity_grid_to_distorted),
      initial_acceleration_grid_to_distorted_(
          initial_acceleration_grid_to_distorted),
      initial_value_distorted_to_inertial_(initial_value_distorted_to_inertial),
      initial_velocity_distorted_to_inertial_(
          initial_velocity_distorted_to_inertial),
      initial_acceleration_distorted_to_inertial_(
          initial_acceleration_distorted_to_inertial),
      distorted_and_inertial_frames_are_equal_(false) {
  if (min_radius >= max_radius) {
    PARSE_ERROR(context,
                "Tried to create a SphericalCompression TimeDependence, but "
                "the minimum radius ("
                    << min_radius << ") is not less than the maximum radius ("
                    << max_radius << ")");
  }
}

std::unique_ptr<TimeDependence<3>> SphericalCompression::get_clone() const {
  if (distorted_and_inertial_frames_are_equal_) {
    return std::make_unique<SphericalCompression>(
        initial_time_, min_radius_, max_radius_, center_,
        initial_value_grid_to_distorted_, initial_velocity_grid_to_distorted_,
        initial_acceleration_grid_to_distorted_);
  } else {
    return std::make_unique<SphericalCompression>(
        initial_time_, min_radius_, max_radius_, center_,
        initial_value_grid_to_distorted_, initial_velocity_grid_to_distorted_,
        initial_acceleration_grid_to_distorted_,
        initial_value_distorted_to_inertial_,
        initial_velocity_distorted_to_inertial_,
        initial_acceleration_distorted_to_inertial_);
  }
}

std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
SphericalCompression::block_maps_grid_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      result{number_of_blocks};
  if (distorted_and_inertial_frames_are_equal_) {
    result[0] = std::make_unique<GridToInertialMapSimple>(
        grid_to_inertial_map_simple());
  } else {
    result[0] = std::make_unique<GridToInertialMapCombined>(
        grid_to_inertial_map_combined());
  }
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
SphericalCompression::block_maps_grid_to_distorted(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
      result{number_of_blocks};
  if (not distorted_and_inertial_frames_are_equal_) {
    result[0] = std::make_unique<GridToDistortedMap>(grid_to_distorted_map());
    for (size_t i = 1; i < number_of_blocks; ++i) {
      result[i] = result[0]->get_clone();
    }
  }
  return result;
}

std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
SphericalCompression::block_maps_distorted_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
      result{number_of_blocks};
  if (not distorted_and_inertial_frames_are_equal_) {
    result[0] =
        std::make_unique<DistortedToInertialMap>(distorted_to_inertial_map());
    for (size_t i = 1; i < number_of_blocks; ++i) {
      result[i] = result[0]->get_clone();
    }
  }
  return result;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
SphericalCompression::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Functions of time don't expire by default
  double expiration_time_grid_to_distorted =
      std::numeric_limits<double>::infinity();

  // If we have control systems, overwrite the expiration time with the one
  // supplied by the control system
  if (initial_expiration_times.count(
          function_of_time_name_grid_to_distorted_) == 1) {
    expiration_time_grid_to_distorted =
        initial_expiration_times.at(function_of_time_name_grid_to_distorted_);
  }

  result[function_of_time_name_grid_to_distorted_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_value_grid_to_distorted_},
                                     {initial_velocity_grid_to_distorted_},
                                     {initial_acceleration_grid_to_distorted_},
                                     {0.0}}},
          expiration_time_grid_to_distorted);

  if (not distorted_and_inertial_frames_are_equal_) {
    double expiration_time_distorted_to_inertial =
        std::numeric_limits<double>::infinity();
    if (initial_expiration_times.count(
            function_of_time_name_distorted_to_inertial_) == 1) {
      expiration_time_distorted_to_inertial = initial_expiration_times.at(
          function_of_time_name_distorted_to_inertial_);
    }
    result[function_of_time_name_distorted_to_inertial_] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_,
            std::array<DataVector, 4>{
                {{initial_value_distorted_to_inertial_},
                 {initial_velocity_distorted_to_inertial_},
                 {initial_acceleration_distorted_to_inertial_},
                 {0.0}}},
            expiration_time_distorted_to_inertial);
  }
  return result;
}

auto SphericalCompression::grid_to_inertial_map_simple() const
    -> GridToInertialMapSimple {
  return GridToInertialMapSimple{
      SphericalCompressionMap{function_of_time_name_grid_to_distorted_,
                              min_radius_, max_radius_, center_}};
}

auto SphericalCompression::grid_to_distorted_map() const -> GridToDistortedMap {
  return GridToDistortedMap{
      SphericalCompressionMap{function_of_time_name_grid_to_distorted_,
                              min_radius_, max_radius_, center_}};
}

auto SphericalCompression::distorted_to_inertial_map() const
    -> DistortedToInertialMap {
  return DistortedToInertialMap{
      SphericalCompressionMap{function_of_time_name_distorted_to_inertial_,
                              min_radius_, max_radius_, center_}};
}

auto SphericalCompression::grid_to_inertial_map_combined() const
    -> GridToInertialMapCombined {
  return GridToInertialMapCombined{
      SphericalCompressionMap{function_of_time_name_grid_to_distorted_,
                              min_radius_, max_radius_, center_},
      SphericalCompressionMap{function_of_time_name_distorted_to_inertial_,
                              min_radius_, max_radius_, center_}};
}

bool operator==(const SphericalCompression& lhs,
                const SphericalCompression& rhs) {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.min_radius_ == rhs.min_radius_ and
         lhs.max_radius_ == rhs.max_radius_ and lhs.center_ == rhs.center_ and
         lhs.initial_value_grid_to_distorted_ ==
             rhs.initial_value_grid_to_distorted_ and
         lhs.initial_velocity_grid_to_distorted_ ==
             rhs.initial_velocity_grid_to_distorted_ and
         lhs.initial_acceleration_grid_to_distorted_ ==
             rhs.initial_acceleration_grid_to_distorted_ and
         lhs.distorted_and_inertial_frames_are_equal_ ==
             rhs.distorted_and_inertial_frames_are_equal_ and
         (lhs.distorted_and_inertial_frames_are_equal_ or
          (lhs.initial_value_distorted_to_inertial_ ==
               rhs.initial_value_distorted_to_inertial_ and
           lhs.initial_velocity_distorted_to_inertial_ ==
               rhs.initial_velocity_distorted_to_inertial_ and
           lhs.initial_acceleration_distorted_to_inertial_ ==
               rhs.initial_acceleration_distorted_to_inertial_));
}

bool operator!=(const SphericalCompression& lhs,
                const SphericalCompression& rhs) {
  return not(lhs == rhs);
}
}  // namespace creators::time_dependence

using SphericalCompressionMap3d =
    CoordinateMaps::TimeDependent::SphericalCompression<false>;

INSTANTIATE_MAPS_FUNCTIONS(((SphericalCompressionMap3d)), (Frame::Grid),
                           (Frame::Distorted, Frame::Inertial),
                           (double, DataVector))
INSTANTIATE_MAPS_FUNCTIONS(((SphericalCompressionMap3d)), (Frame::Distorted),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
