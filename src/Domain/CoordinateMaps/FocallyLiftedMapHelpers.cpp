// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps::FocallyLiftedMapHelpers {

template <typename T>
tt::remove_cvref_wrap_t<T> scale_factor(
    const std::array<T, 3>& src_point, const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    const bool src_is_between_proj_and_target) noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  // quadratic equation is
  // a x^2 + b x + c = 0
  const return_type a = square(src_point[0] - proj_center[0]) +
                        square(src_point[1] - proj_center[1]) +
                        square(src_point[2] - proj_center[2]);
  const return_type b =
      2.0 *
      ((src_point[0] - proj_center[0]) * (proj_center[0] - sphere_center[0]) +
       (src_point[1] - proj_center[1]) * (proj_center[1] - sphere_center[1]) +
       (src_point[2] - proj_center[2]) * (proj_center[2] - sphere_center[2]));
  const double c = square(sphere_center[0] - proj_center[0]) +
                   square(sphere_center[1] - proj_center[1]) +
                   square(sphere_center[2] - proj_center[2]) - square(radius);
  if (src_is_between_proj_and_target) {
    // Here we assume that src_point is between proj_center and
    // target_point.  There are three cases: 1) src and proj are both
    // inside the sphere, 2) src and proj are both outside the sphere,
    // and 3) proj is outside the sphere and src is inside the sphere.
    // To cover all 3 cases, we choose the smallest root that is
    // greater than or equal to unity. This means for case 2) we are
    // choosing the point closest to src.
    return smallest_root_greater_than_value_within_roundoff(
        a, b, make_with_value<return_type>(a, c), 1.0);
  }
  // Here we assume that target_point is between proj_center and
  // src_point. There are three cases: 1) proj is inside the sphere
  // and src is outside the sphere 2) src is inside the sphere and proj
  // is outside the sphere, and 3) the sphere is between src and proj.
  // To cover all 3 cases, we choose the largest root that is less than
  // or equal to unity, and we require that this root is positive.
  // This means that for 3) we are choosing the point closest to src.
  return largest_root_between_values_within_roundoff(
      a, b, make_with_value<return_type>(a, c), 0.0, 1.0);
}

boost::optional<double> try_scale_factor(
    const std::array<double, 3>& src_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    const bool pick_larger_root,
    const bool pick_root_greater_than_one) noexcept {
  // We solve the quadratic for (scale_factor-1) instead of scale_factor to
  // avoid roundoff problems when scale_factor is very nearly equal to unity.
  // Note that scale_factor==1 will occur when src_point is on the sphere, which
  // happens when inverse-mapping the boundaries.

  // quadratic equation is
  // a x^2 + b x + c = 0
  const double a = square(src_point[0] - proj_center[0]) +
                   square(src_point[1] - proj_center[1]) +
                   square(src_point[2] - proj_center[2]);
  const double b =
      2.0 *
      ((src_point[0] - proj_center[0]) * (src_point[0] - sphere_center[0]) +
       (src_point[1] - proj_center[1]) * (src_point[1] - sphere_center[1]) +
       (src_point[2] - proj_center[2]) * (src_point[2] - sphere_center[2]));
  const double c = square(sphere_center[0] - src_point[0]) +
                   square(sphere_center[1] - src_point[1]) +
                   square(sphere_center[2] - src_point[2]) - square(radius);

  double x0 = std::numeric_limits<double>::signaling_NaN();
  double x1 = std::numeric_limits<double>::signaling_NaN();
  const int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
  if (num_real_roots == 2) {
    // We solved for scale_factor-1 above, so add 1 to get scale_factor.
    x0 += 1.0;
    x1 += 1.0;
    if (equal_within_roundoff(x0, 1.0)) {
      x0 = 1.0;
    }
    if (equal_within_roundoff(x1, 1.0)) {
      x1 = 1.0;
    }
    if (pick_root_greater_than_one) {
      // For the inverse map, we want the a scale_factor s such that
      // s >= 1. Note that gsl_poly_solve_quadratic returns x0 < x1.
      // have three cases:
      //  a) x0 < x1 < 1         ->   error
      //  b) x0 < 1 <= x1        ->   Choose x1
      //  c) 1 <= x0 < x1        ->   choose based on pick_larger_root
      if (x0 >= 1.0 and not pick_larger_root) {
        return x0;
      } else if (x1 >= 1.0) {
        return x1;
      } else {
        return boost::none;
      }
    } else {
      // For the inverse map, we want a scale_factor s such that 0 < s <= 1.
      // Note that gsl_poly_solve_quadratic returns x0 < x1.
      // So we have six cases:
      //  a) x0 < x1 <= 0        ->   error
      //  b) x0 <= 0 < x1 <= 1   ->   Choose x1
      //  c) x0 <= 0 and x1 > 1  ->   error
      //  d) 0 < x0 < x1 <= 1      ->   Choose according to pick_larger_root
      //  e) 0 < x0 <= 1 < x1      ->   Choose x0
      //  f) 1 < x0 < x1           ->   error
      if (x0 <= 0.0) {
        if (x1 > 0.0 and x1 <= 1.0) {
          return x1;  // b)
        } else {
          return boost::none;  // a) and c)
        }
      } else if (x0 <= 1.0) {
        if (x1 > 1.0) {
          return x0;  // e)
        } else {
          return pick_larger_root ? x1 : x0;  // d)
        }
      } else {
        return boost::none;  // f)
      }
    }
  } else if (num_real_roots == 1) {
    // We solved for scale_factor-1 above, so add 1 to get scale_factor.
    x0 += 1.0;
    if (equal_within_roundoff(x0, 1.0)) {
      x0 = 1.0;
    }
    if (pick_root_greater_than_one) {
      if (x0 < 1.0) {
        return boost::none;
      } else {
        return x0;
      }
    } else {
      if (x0 <= 0.0 or x0 > 1.0) {
        return boost::none;
      } else {
        return x0;
      }
    }
  } else {
    return boost::none;
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> d_scale_factor_d_src_point(
    const std::array<T, 3>& intersection_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, const T& lambda) noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type lambda_over_denominator =
      square(lambda) / (square(intersection_point[0] - proj_center[0]) +
                        square(intersection_point[1] - proj_center[1]) +
                        square(intersection_point[2] - proj_center[2]) +
                        ((intersection_point[0] - proj_center[0]) *
                             (proj_center[0] - sphere_center[0]) +
                         (intersection_point[1] - proj_center[1]) *
                             (proj_center[1] - sphere_center[1]) +
                         (intersection_point[2] - proj_center[2]) *
                             (proj_center[2] - sphere_center[2])));
  auto result =
      make_with_value<std::array<return_type, 3>>(lambda_over_denominator, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) =
        lambda_over_denominator *
        (gsl::at(sphere_center, i) - gsl::at(intersection_point, i));
  }
  return result;
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template tt::remove_cvref_wrap_t<DTYPE(data)> scale_factor<DTYPE(data)>( \
      const std::array<DTYPE(data), 3>& src_point,                         \
      const std::array<double, 3>& proj_center,                            \
      const std::array<double, 3>& sphere_center, double radius,           \
      bool src_is_between_proj_and_target) noexcept;                       \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>             \
  d_scale_factor_d_src_point<DTYPE(data)>(                                 \
      const std::array<DTYPE(data), 3>& intersection_point,                \
      const std::array<double, 3>& proj_center,                            \
      const std::array<double, 3>& sphere_center,                          \
      const DTYPE(data) & lambda) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef INSTANTIATE
#undef DTYPE
/// \endcond

}  // namespace domain::CoordinateMaps::FocallyLiftedMapHelpers
