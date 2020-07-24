// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"

#include <algorithm>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <limits>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

// Map the Cartesian coordinates of the old Strahlkorper's
// collocation points to the new frame, and compute the approximate
// center in the new coordinates, the spherical coordinates of those
// collocation points with respect to the new center, and the
// min/max radius.
template <typename SrcFrame, typename DestFrame>
void mapped_appx_center_and_spherical_coords(
    const gsl::not_null<std::array<double, 3>*> center,
    const gsl::not_null<std::array<DataVector, 3>*> r_theta_phi,
    const gsl::not_null<double*> r_min, const gsl::not_null<double*> r_max,
    const Strahlkorper<SrcFrame>& strahlkorper,
    const std::unique_ptr<domain::CoordinateMapBase<SrcFrame, DestFrame, 3>>&
        map) noexcept {
  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto radius =
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients());

  const auto r_hat = [&theta_phi]() noexcept {
    const DataVector sin_theta = sin(theta_phi[0]);
    return tnsr::i<DataVector, 3, SrcFrame>{{sin_theta * cos(theta_phi[1]),
                                             sin_theta * sin(theta_phi[1]),
                                             cos(theta_phi[0])}};
  }
  ();

  const tnsr::I<DataVector, 3, SrcFrame> src_coords{
      {strahlkorper.center()[0] + get<0>(r_hat) * radius,
       strahlkorper.center()[1] + get<1>(r_hat) * radius,
       strahlkorper.center()[2] + get<2>(r_hat) * radius}};

  // Cartesian coordinates in the new frame.
  const auto dest_coords = (*map)(src_coords);

  // Compute approximate center by averaging the coordinates.
  *center = [&dest_coords]() noexcept {
    std::array<double, 3> center_l{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(center_l, d) = 0.0;
      for (const auto& coord_pt : dest_coords.get(d)) {
        gsl::at(center_l, d) += coord_pt;
      }
      gsl::at(center_l, d) /= dest_coords.get(d).size();
    }
    return center_l;
  }
  ();

  // Spherical coordinates in the new frame.
  (*r_theta_phi)[0] = square(get<0>(dest_coords) - (*center)[0]);
  for (size_t d = 1; d < 3; ++d) {  // start at 1
    (*r_theta_phi)[0] += square(dest_coords.get(d) - gsl::at(*center, d));
  }
  (*r_theta_phi)[0] = sqrt((*r_theta_phi)[0]);
  (*r_theta_phi)[1] =
      acos((get<2>(dest_coords) - (*center)[2]) / (*r_theta_phi)[0]);
  (*r_theta_phi)[2] = atan2(get<1>(dest_coords) - (*center)[1],
                            get<0>(dest_coords) - (*center)[0]);

  // Minimum and maximum radius of each point with respect to the new
  // center, used for bracketing the roots.
  const auto minmax =
      std::minmax_element((*r_theta_phi)[0].begin(), (*r_theta_phi)[0].end());
  *r_max = *(minmax.first);
  *r_min = *(minmax.second);
}

// Let S be the set of collocation points on the source Strahlkorper,
// mapped into the destination frame; these points are in the array
// called `r_theta_phi`.  The points in S are not necessarily ordered
// in any regular way in the destination frame.  For each (theta, phi)
// collocation point of the new Strahlkorper in the destination frame,
// find the point in S that is closest in theta and phi and return the
// radius of that point.  This radius will serve as the initial guess
// for the radius of new Strahlkorper at (theta,phi).
DataVector radius_at_closest_theta_phi_point(
    const std::array<DataVector, 3>& r_theta_phi,
    const std::array<DataVector, 2>& new_theta_phi) noexcept {
  DataVector new_radius(new_theta_phi[0].size());
  for (size_t i = 0; i < new_radius.size(); ++i) {
    const double theta = new_theta_phi[0][i];
    const double phi = new_theta_phi[1][i];
    const double sin_theta = sin(theta);
    double radius = std::numeric_limits<double>::signaling_NaN();
    double min_distance_squared = std::numeric_limits<double>::max();
    for (size_t j = 0; j < r_theta_phi[0].size(); ++j) {
      double dtheta = fabs(r_theta_phi[1][j] - theta);
      double dphi = fabs(r_theta_phi[2][j] - phi);
      while (dphi > 2.0 * M_PI) {
        dphi -= 2.0 * M_PI;
      }
      while (dtheta > M_PI) {
        dtheta -= M_PI;
      }
      const double distance_squared = square(dtheta) + square(sin_theta * dphi);
      if (distance_squared < min_distance_squared) {
        min_distance_squared = distance_squared;
        radius = r_theta_phi[0][j];
      }
    }
    new_radius[i] = radius;
  }
  return new_radius;
}

// Tries to bracket a root, given a functor, and index, and two arrays
// of x and y (with each y=f(x)) that have been tried.
//
// We need to take into account that y might be undefined (i.e. an invalid
// boost::optional).
//
// We assume that there is only one root in the interval.
// So this means we have only 2 possibilities:
// 1) All points are invalid, e.g. "X X X X X X X X".
//    (here X represents an invalid point)
// 2) All valid points are adjacent, with the same sign, e.g. "X X + + X X".
//    (here + represents a point with y>0).
// Note that "X X + + - + X X" and so forth will not occur because the
// root would have been deemed bracketed before bracket_by_contracting
// was called.
//
// For case 1) above, we need to continue to bisect between each set
// of points and keep searching.  For case 2) we need to check only
// between valid and invalid points.  That is, for "X X + + X X" we
// bisect only between points 1 and 2 and between points 3 and 4.  For
// "+ + X X X" we bisect between points 1 and 2.
template <typename Functor>
std::array<double, 4> bracket_by_contracting(
    const std::vector<double>& x, const std::vector<boost::optional<double>>& y,
    const Functor& f, const size_t index, const size_t level = 0) noexcept {
  // First check if we have any valid points.
  size_t last_valid_index = y.size();
  for (size_t i = y.size(); i >= 1; --i) {
    if (y[i - 1]) {
      last_valid_index = i - 1;
      break;
    }
  }

  if (last_valid_index == y.size()) {
    // No valid points!

    // Create larger arrays with one point between each of the already
    // computed points that we will consider.
    std::vector<double> xx(x.size() * 2 - 1);
    std::vector<boost::optional<double>> yy(y.size() * 2 - 1);

    // Copy all even-numbered points in the range.
    for (size_t i = 0; i < x.size(); ++i) {
      xx[2 * i] = x[i];
      yy[2 * i] = y[i];
    }

    // Fill midpoints and check for bracket on each one.
    for (size_t i = 0; i < x.size() - 1; ++i) {
      xx[2 * i + 1] = x[i] + 0.5 * (x[i + 1] - x[i]);
      yy[2 * i + 1] = f(xx[2 * i + 1], index);
      if (yy[2 * i + 1]) {
        // Valid point! But we know all the other points are invalid,
        // so we need to check only 3 points here: the new point and
        // its neighbors.
        if (level > 6) {
          ERROR("Too many iterations in bracket_by_contracting");
        }
        // Recurse.
        return bracket_by_contracting({{x[i], xx[2 * i + 1], x[i + 1]}},
                                      {{y[i], yy[2 * i + 1], y[i + 1]}}, f,
                                      index, level + 1);
      }
    }
    if (level > 6) {
      ERROR("Too many iterations in bracket_by_contracting");
    }
    // Recurse, using all points.
    return bracket_by_contracting(xx, yy, f, index, level + 1);
  }

  // If we get here, we have a valid point.

  // Check if there is more than one valid point.
  size_t first_valid_index = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    if (y[i]) {
      first_valid_index = i;
      break;
    }
  }

  std::vector<double> xx;
  std::vector<boost::optional<double>> yy;
  if (first_valid_index > 0) {
    // Check for a root between first_valid_index-1 and first_valid_index.
    const double x_test =
        x[first_valid_index - 1] +
        0.5 * (x[first_valid_index] - x[first_valid_index - 1]);
    const auto y_test = f(x_test, index);
    if (y_test and y[first_valid_index].get() * y_test.get() <= 0.0) {
      // Bracketed!
      return std::array<double, 4>{{x_test, x[first_valid_index], y_test.get(),
                                    y[first_valid_index].get()}};
    } else {
      xx.push_back(x[first_valid_index - 1]);
      xx.push_back(x_test);
      xx.push_back(x[first_valid_index]);
      yy.push_back(y[first_valid_index - 1]);
      yy.push_back(y_test);
      yy.push_back(y[first_valid_index]);
    }
  }
  if (last_valid_index < y.size() - 1) {
    // Check for a root between last_valid_index and last_valid_index+1.
    const double x_test = x[last_valid_index] +
                          0.5 * (x[last_valid_index + 1] - x[last_valid_index]);
    const auto y_test = f(x_test, index);
    if (y_test and y[last_valid_index].get() * y_test.get() <= 0.0) {
      // Bracketed!
      return std::array<double, 4>{{x[last_valid_index], x_test,
                                    y[last_valid_index].get(), y_test.get()}};
    } else {
      xx.push_back(x[last_valid_index]);
      xx.push_back(x_test);
      xx.push_back(x[last_valid_index]);
      yy.push_back(y[last_valid_index + 1]);
      yy.push_back(y_test);
      yy.push_back(y[last_valid_index + 1]);
    }
  }

  if (level > 6) {
    ERROR("Too many iterations in bracket_by_contracting");
  }
  // Didn't find a bracket, so recurse.
  return bracket_by_contracting(xx, yy, f, index, level + 1);
}

// Given initial guess for lower_bound and upper_bound, adjusts
// lower_bound and upper_bound so that the root is bracketed, and
// fills f_at_lower_bound and f_at_upper_bound.  Uses guess_value as
// an initial guess.
//
// We assume that lower_bound and upper_bound would normally bracket
// the root.  However, because we are passing the point through a map,
// it is possible that lower_bound or upper_bound are sufficiently far
// from the root that the map is invalid (meaning that the functor
// returns boost::none); this might happen if the Strahlkorper is very
// near an excision boundary. In that case, we assume that the problem
// is that our original bounds are too wide, and reduce them until we
// have bracketed the root.
template <typename Functor>
void bracket(const gsl::not_null<DataVector*> lower_bound,
             const gsl::not_null<DataVector*> upper_bound,
             const gsl::not_null<DataVector*> f_at_lower_bound,
             const gsl::not_null<DataVector*> f_at_upper_bound,
             const DataVector& guess_value, const Functor& f) noexcept {
  for (size_t s = 0; s < lower_bound->size(); ++s) {
    // Initial values of x1,x2,y1,y2.
    // Use guess_value and upper bound.  This is because guess_value
    // typically underestimates the actual radius, and because lower_bound
    // is more likely than upper_bound to be invalid.

    double x1 = guess_value[s];
    double x2 = (*upper_bound)[s];
    auto y1 = f(x1, s);
    auto y2 = f(x2, s);
    if (not(y1 and y2 and y1.get() * y2.get() <= 0.0)) {
      // Root is not bracketed, so try to do so.
      // First try lower_bound.
      const double x3 = (*lower_bound)[s];
      const auto y3 = f(x3, s);
      if (y1 and y3 and y1.get() * y2.get() <= 0.0) {
        // Bracketed! Throw out x2,y2
        x2 = x3;
        y2 = y3;
      } else if (y2 and y3 and y2.get() * y3.get() <= 0.0) {
        // Bracketed! Throw out x1,y1
        x1 = x3;
        y1 = y3;
      } else {
        // Need to do something more sophisticated.
        std::array<double, 4> tmp =
            bracket_by_contracting({{x3, x1, x2}}, {{y3, y1, y2}}, f, s);
        x1 = tmp[0];
        x2 = tmp[1];
        y1 = tmp[2];
        y2 = tmp[3];
      }
    }
    (*f_at_lower_bound)[s] = y1.get();
    (*f_at_upper_bound)[s] = y2.get();
    (*lower_bound)[s] = x1;
    (*upper_bound)[s] = x2;
  }
}
}  // namespace

template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    const gsl::not_null<Strahlkorper<DestFrame>*> new_strahlkorper,
    const Strahlkorper<SrcFrame>& strahlkorper,
    const std::unique_ptr<domain::CoordinateMapBase<SrcFrame, DestFrame, 3>>&
        map) noexcept {
  // The strategy here is to take each (theta,phi) point of the new
  // strahlkorper and compute its radius.  This is done by root
  // finding.  The bracketing here can be tricky, because we need to
  // take into account guesses that might be outside the range of the
  // map.

  // Map the Cartesian coordinates of the old Strahlkorper's
  // collocation points to the new frame, and compute the approximate
  // center in the new coordinates, the spherical coordinates of those
  // collocation points with respect to the new center, and the
  // min/max radius.
  std::array<double, 3> new_center{};
  std::array<DataVector, 3> r_theta_phi{};
  double r_min = 0.0;
  double r_max = 0.0;
  mapped_appx_center_and_spherical_coords(
      make_not_null(&new_center), make_not_null(&r_theta_phi),
      make_not_null(&r_min), make_not_null(&r_max), strahlkorper, map);

  // Collocation points of the new strahlkorper
  // The only thing that is used about the new Strahlkorper here is
  // its l_max, m_max.
  const auto new_theta_phi =
      new_strahlkorper->ylm_spherepack().theta_phi_points();

  // For each (theta,phi) collocation point of the new Strahlkorper,
  // find the theta,phi of the nearest mapped point (the
  // `r_theta_phi`) array, and return the radius of that point.  This
  // serves as an initial guess for the radius associated with that
  // (theta,phi) collocation point. We will compute that radius more
  // precisely using root-finding below.
  const DataVector guess_radius =
      radius_at_closest_theta_phi_point(r_theta_phi, new_theta_phi);

  // Get the r_hat vector of the new Strahlkorper, which defines the
  // collocation points.
  const auto new_r_hat = [&new_theta_phi]() noexcept {
    const DataVector sin_theta = sin(new_theta_phi[0]);
    return tnsr::i<DataVector, 3, SrcFrame>{{sin_theta * cos(new_theta_phi[1]),
                                             sin_theta * sin(new_theta_phi[1]),
                                             cos(new_theta_phi[0])}};
  }
  ();

  // Now we must find the coord radius of the old surface, at each of
  // the collocation points of the new surface.  To do so, for each
  // index 's' find the root 'r' that zeroes this lambda function.
  // Here the lambda function returns boost::none if the supplied
  // point is outside the range of the map.
  const auto radius_function =
      [&map, &new_center, &new_r_hat, &strahlkorper ](
          const double r, const size_t s) noexcept->boost::optional<double> {
    // Get cartesian coordinates of the point in the new frame
    // that corresponds to a radius of 'r' and the angle of the
    // collocation point.
    const tnsr::I<double, 3, DestFrame> x_new{
        {r * get<0>(new_r_hat)[s] + new_center[0],
         r * get<1>(new_r_hat)[s] + new_center[1],
         r * get<2>(new_r_hat)[s] + new_center[2]}};

    // Find cartesian coordinates of the same poit in the old frame,
    // with respect to the old center.
    const auto x_old_test = map->inverse(x_new);
    if (not x_old_test) {
      return boost::none;
    }
    const auto& old_center = strahlkorper.center();
    const std::array<double, 3> x_old{
        {get<0>(x_old_test.get()) - old_center[0],
         get<1>(x_old_test.get()) - old_center[1],
         get<2>(x_old_test.get()) - old_center[2]}};

    // Find (r, theta, phi) of the same point in the old frame.
    const double r_old =
        sqrt(square(x_old[0]) + square(x_old[1]) + square(x_old[2]));
    const double theta_old = acos(x_old[2] / r_old);
    const double phi_old = atan2(x_old[1], x_old[0]);
    // Evaluate the radius of the surface on the old strahlkorper
    // at theta_old, phi_old.
    const double r_surf = strahlkorper.radius(theta_old, phi_old);

    // If r_surf is r_old, then 'r' is on the surface.
    return r_surf - r_old;
  };

  // Bracket the root.
  const double padding = 0.10; // in case r_min and r_max are not enough.
  DataVector lower_bound(get<0>(new_r_hat).size(), r_min * (1.0 - padding));
  DataVector upper_bound(get<0>(new_r_hat).size(), r_max * (1.0 + padding));
  DataVector f_at_lower_bound(get<0>(new_r_hat).size());
  DataVector f_at_upper_bound(get<0>(new_r_hat).size());
  bracket(&lower_bound, &upper_bound, &f_at_lower_bound, &f_at_upper_bound,
          guess_radius, radius_function);

  // Find the root.
  const auto radius_at_each_angle = RootFinder::toms748(
      [&radius_function](const double r, const size_t s) noexcept {
        return radius_function(r, s).get();
      },
      lower_bound, upper_bound, f_at_lower_bound, f_at_upper_bound,
      std::numeric_limits<double>::epsilon() * (r_min + r_max),
      2.0 * std::numeric_limits<double>::epsilon());

  // Now reset the radius and center of the new strahlkorper.
  *new_strahlkorper = Strahlkorper<DestFrame>(new_strahlkorper->l_max(),
                                              new_strahlkorper->m_max(),
                                              radius_at_each_angle, new_center);
}

/// \cond
#define SFRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                             \
  template void strahlkorper_in_different_frame(                         \
      const gsl::not_null<Strahlkorper<DFRAME(data)>*> new_strahlkorper, \
      const Strahlkorper<SFRAME(data)>& strahlkorper,                    \
      const std::unique_ptr<                                             \
          domain::CoordinateMapBase<SFRAME(data), DFRAME(data), 3>>&     \
          map) noexcept;
GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Grid), (::Frame::Inertial))
GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Inertial), (::Frame::Grid))
#undef INSTANTIATE
#undef DFRAME
#undef SFRAME
/// \endcond
