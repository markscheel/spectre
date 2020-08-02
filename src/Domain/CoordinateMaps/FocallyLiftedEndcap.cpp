// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {

namespace Endcap_detail {
double sin_ax_over_x(double x, double ax, double a) noexcept {
  return square(ax) < 6.0 * std::numeric_limits<double>::epsilon()
             ? a
             : sin(ax) / x;
}
double sin_ax_over_x(double x, double a) noexcept {
  return sin_ax_over_x(x, a * x, a);
}
DataVector sin_ax_over_x(const DataVector& x, double a) noexcept {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = sin_ax_over_x(x[i], a * x[i], a);
  }
  return result;
}
double dlogx_sin_ax_over_x(double x, double ax, double a) noexcept {
  return square(ax) < 10.0 * std::numeric_limits<double>::epsilon()
             ? -cube(a) / 3.0
             : (a * cos(ax) - sin(ax) / x) / square(x);
}
double dlogx_sin_ax_over_x(double x, double a) noexcept {
  return dlogx_sin_ax_over_x(x, a * x, a);
}
DataVector dlogx_sin_ax_over_x(const DataVector& x, double a) noexcept {
  DataVector result(x);
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = dlogx_sin_ax_over_x(x[i], a * x[i], a);
  }
  return result;
}
}  // namespace Endcap_detail

Endcap::Endcap(const std::array<double, 3>& center, double radius,
               double z_plane) noexcept
    : center_(center),
      radius_([&]() noexcept {
        // The equal_within_roundoff below has an implicit scale of 1,
        // so the ASSERT may trigger in the case where we really
        // want an entire domain that is very small.
        ASSERT(not equal_within_roundoff(radius, 0.0),
               "Cannot have zero radius");
        return radius;
      }()),
      theta_(acos((z_plane - center_[2]) / radius_)) {
  ASSERT(z_plane != center[2],
         "Plane must intersect sphere at more than one point");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Endcap::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  const return_type sin_factor =
      radius_ * Endcap_detail::sin_ax_over_x(rho, theta_);
  const return_type z = radius_ * cos(rho * theta_) + center_[2];
  const return_type x = sin_factor * xbar + center_[0];
  const return_type y = sin_factor * ybar + center_[1];
  return std::array<return_type, 3>{{std::move(x), std::move(y), std::move(z)}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Endcap::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  const return_type sin_factor =
      radius_ * Endcap_detail::sin_ax_over_x(rho, theta_);
  const return_type d_sin_factor =
      radius_ * Endcap_detail::dlogx_sin_ax_over_x(rho, theta_);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dz/dxbar
  get<2, 0>(jacobian_matrix) = -sin_factor * theta_ * xbar;
  // dz/dybar
  get<2, 1>(jacobian_matrix) = -sin_factor * theta_ * ybar;
  // dx/dxbar
  get<0, 0>(jacobian_matrix) = d_sin_factor * square(xbar) + sin_factor;
  // dx/dybar
  get<0, 1>(jacobian_matrix) = d_sin_factor * xbar * ybar;
  // dy/dxbar
  get<1, 0>(jacobian_matrix) = d_sin_factor * ybar * xbar;
  // dy/dybar
  get<1, 1>(jacobian_matrix) = d_sin_factor * square(ybar) + sin_factor;

  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Endcap::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  const return_type& xbar = source_coords[0];
  const return_type& ybar = source_coords[1];
  const return_type rho = sqrt(square(xbar) + square(ybar));
  // Let q = sin(rho theta)/rho
  const return_type q = Endcap_detail::sin_ax_over_x(rho, theta_);
  const return_type dlogrho_q = Endcap_detail::dlogx_sin_ax_over_x(rho, theta_);
  const return_type one_over_r_q = 1.0 / (q * radius_);
  const return_type tmp =
      one_over_r_q * dlogrho_q / (q + square(rho) * dlogrho_q);

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dxbar/dx
  get<0, 0>(inv_jacobian_matrix) = one_over_r_q - square(xbar) * tmp;
  // dxbar/dy
  get<0, 1>(inv_jacobian_matrix) = -xbar * ybar * tmp;
  // dybar/dx
  get<1, 0>(inv_jacobian_matrix) = get<0, 1>(inv_jacobian_matrix);
  // dybar/dy
  get<1, 1>(inv_jacobian_matrix) = one_over_r_q - square(ybar) * tmp;

  return inv_jacobian_matrix;
}

boost::optional<std::array<double, 3>> Endcap::inverse(
    const std::array<double, 3>& target_coords, const double sigma_in) const
    noexcept {
  const double x = target_coords[0] - center_[0];
  const double y = target_coords[1] - center_[1];
  const double z = target_coords[2] - center_[2];
  const double r = sqrt(square(x) + square(y) + square(z));
  // The equal_within_roundoff below has an implicit scale of 1,
  // so the inverse may fail if radius_ is very small on purpose,
  // e.g. if we really want a tiny tiny domain for some reason.
  if (not equal_within_roundoff(r, radius_)) {
    return boost::none;
  }

  // Compute sin^2(rho theta).
  const double sin_squared_rho_theta = (square(x) + square(y)) / square(r);
  // Compute sin(rho theta)/rho.
  // If sin^2(rho theta) is small,
  // use arcsin(q) = q(1 + q^2/6 + 3 q^4/40 + ...)
  // for q = sin(rho theta).
  double sin_rho_theta_over_rho = 0.0;
  if (square(sin_squared_rho_theta) <
      (40.0 / 3.0) * std::numeric_limits<double>::epsilon()) {
    sin_rho_theta_over_rho = theta_ * (1.0 - sin_squared_rho_theta / 6.0);
  } else {
    const double rho = asin(sqrt(sin_squared_rho_theta)) / theta_;
    sin_rho_theta_over_rho = sqrt(sin_squared_rho_theta) / rho;
  }

  // Note about the division in the next line: The above check of r
  // versus radius_ means that r cannot be zero unless the radius of
  // the sphere (a map parameter) is chosen to be zero, which would
  // make the map singular.  Also sin_rho_theta_over_rho cannot be
  // zero unless theta_ (a map parameter) is chosen to be zero, which
  // also would make the map singular.
  const double xbar = x / (r * sin_rho_theta_over_rho);
  const double ybar = y / (r * sin_rho_theta_over_rho);
  const double rho_squared = square(xbar) + square(ybar);
  if (rho_squared > 1.0 and not equal_within_roundoff(rho_squared, 1.0)) {
    return boost::none;
  }

  const double zbar = 2.0 * sigma_in - 1.0;
  if (abs(zbar) > 1.0 and not equal_within_roundoff(abs(zbar), 1.0)) {
    return boost::none;
  }

  return std::array<double, 3>{{xbar, ybar, zbar}};
}

template <typename T>
tt::remove_cvref_wrap_t<T> Endcap::sigma(
    const std::array<T, 3>& source_coords) const noexcept {
  return 0.5 * (source_coords[2] + 1.0);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Endcap::deriv_sigma(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  return std::array<return_type, 3>{
      {make_with_value<return_type>(dereference_wrapper(source_coords[0]), 0.0),
       make_with_value<return_type>(dereference_wrapper(source_coords[0]), 0.0),
       make_with_value<return_type>(dereference_wrapper(source_coords[0]),
                                    0.5)}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Endcap::dxbar_dsigma(
    const std::array<T, 3>& source_coords) const noexcept {
  using return_type = tt::remove_cvref_wrap_t<T>;
  return std::array<return_type, 3>{
      {make_with_value<return_type>(dereference_wrapper(source_coords[0]), 0.0),
       make_with_value<return_type>(dereference_wrapper(source_coords[0]), 0.0),
       make_with_value<return_type>(dereference_wrapper(source_coords[0]),
                                    2.0)}};
}

boost::optional<double> Endcap::lambda_tilde(
    const std::array<double, 3>& parent_mapped_target_coords,
    const std::array<double, 3>& projection_point) const noexcept {
  // Try to find lambda_tilde going from target_coords to sphere.
  // This lambda_tilde should be positive and less than or equal to unity.
  // If there are two such roots, we choose based on where the points are.
  const bool choose_larger_root =
      parent_mapped_target_coords[2] > projection_point[2];
  return FocallyLiftedMapHelpers::try_scale_factor(
      parent_mapped_target_coords, projection_point, center_, radius_,
      choose_larger_root, false);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Endcap::deriv_lambda_tilde(
    const std::array<T, 3>& target_coords, const T& lambda_tilde,
    const std::array<double, 3>& projection_point) const noexcept {
  return FocallyLiftedMapHelpers::d_scale_factor_d_src_point(
      target_coords, projection_point, center_, lambda_tilde);
}

void Endcap::pup(PUP::er& p) noexcept {
  p | center_;
  p | radius_;
  p | theta_;
}

bool operator==(const Endcap& lhs, const Endcap& rhs) noexcept {
  return lhs.center_ == rhs.center_ and lhs.radius_ == rhs.radius_ and
         lhs.theta_ == rhs.theta_;
}

bool operator!=(const Endcap& lhs, const Endcap& rhs) noexcept {
  return not(lhs == rhs);
}
// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> Endcap::       \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Endcap::jacobian(const std::array<DTYPE(data), 3>& source_coords)           \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Endcap::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)       \
      const noexcept;                                                         \
  template tt::remove_cvref_wrap_t<DTYPE(data)> Endcap::sigma(                \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;        \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Endcap::deriv_sigma(const std::array<DTYPE(data), 3>& source_coords)        \
      const noexcept;                                                         \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Endcap::dxbar_dsigma(const std::array<DTYPE(data), 3>& source_coords)       \
      const noexcept;                                                         \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Endcap::deriv_lambda_tilde(const std::array<DTYPE(data), 3>& target_coords, \
                             const DTYPE(data) & lambda_tilde,                \
                             const std::array<double, 3>& projection_point)   \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef INSTANTIATE
#undef DTYPE
/// \endcond

}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
