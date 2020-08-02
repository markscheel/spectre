// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

template <typename InnerMap>
class FocallyLiftedMap;

template <typename InnerMap>
bool operator==(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept;

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Map from \f$(\bar{x},\bar{y},\bar{z})\f$ to the volume
 * contained between a 2D surface and the surface of a 2-sphere.
 *
 * \details We are given the radius \f$R\f$ and
 * center \f$C\f$ of a sphere, and a projection point \f$P\f$.  Also
 * we are given the functions
 * \f$f^i(\bar{x},\bar{y},\bar{z})\f$ and
 * \f$\sigma(\bar{x},\bar{y},\bar{z})\f$ defined below; these
 * functions define the mapping from \f$(\bar{x},\bar{y},\bar{z})\f$
 * to the 2D surface.
 *
 * The input coordinates are labeled \f$(\bar{x},\bar{y},\bar{z})\f$.
 * Let \f$x_0^i = f^i(\bar{x},\bar{y},\bar{z})\f$ be the three coordinates
 * of the 2D surface in 3D space.
 *
 * Now let \f$x_1^i\f$ be a point on the surface of the sphere,
 * constructed so that \f$P\f$, \f$x_0^i\f$, and \f$x_1^i\f$ are
 * co-linear.  In particular, \f$x_1^i\f$ is determined by the
 * equation
 *
 * \f{align} x_1^i = P^i + (x_0^i - P^i) \lambda,\f}
 *
 * where \f$\lambda\f$ is a scalar factor that depends on \f$x_0^i\f$ and
 * that can be computed by solving a quadratic equation.  This quadratic
 * equation is derived by demanding that \f$x_1^i\f$ lies on the sphere:
 *
 * \f{align}
 * |P^i + (x_0^i - P^i) \lambda - C^i |^2 - R^2 = 0.
 * \f}
 *
 * Now assume that \f$x_0^i\f$ corresponds to a level surface defined
 * by some scalar function \f$\sigma(\bar{x},\bar{y},\bar{z})=0\f$,
 * where \f$\sigma\f$ is normalized so that \f$\sigma=1\f$ on the sphere.
 * Then, once \f$\lambda\f$ has been computed and \f$x_1^i\f$ has
 * been determined, the map is given by
 *
 * \f{align}x^i = x_0^i + (x_1^i - x_0^i) \sigma(\bar{x},\bar{y},\bar{z}).\f}
 *
 * ### Jacobian
 *
 * Differentiating Eq. (1) above yields
 *
 * \f[
 * \frac{\partial x_1^i}{\partial x_0^j} = \lambda +
 *  (x_0^i - P^i) \frac{\partial \lambda}{\partial x_0^j}.
 * \f]
 *
 * and differentiating Eq. (2) above yields
 *
 * \f{align}
 * \frac{\partial\lambda}{\partial x_0^j} &=
 * \lambda^2 \frac{C_j - x_1^j}{|x_1^i - P^i|^2
 * + (x_1^i - P^i)(P_i - C_{i})}.
 * \f}
 *
 * The Jacobian can be found by differentiating Eq. (3) above and combining
 * with the last two equations:
 *
 * \f[
 * \frac{\partial x^i}{\partial \bar{x}^j} =
 * \sigma \frac{\partial x_1^i}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial \bar{x}^j} +
 * (1-\sigma)\frac{\partial x_0^i}{\partial \bar{x}^j}
 * + \frac{\partial \sigma}{\partial \bar{x}^j} (x_1^i - x_0^i).
 * \f]
 *
 * The inner map provides the function `deriv_sigma`, which returns
 * \f$\partial \sigma/\partial \bar{x}^j\f$, and the function `jacobian`,
 * which returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 *
 * ### Inverse map.
 *
 * Given \f$x^i\f$, we wish to compute \f$\bar{x}\f$,
 * \f$\bar{y}\f$, and \f$\bar{z}\f$.
 *
 * We first find the coordinates \f$x_0^i\f$ that lie on the 2-surface
 * and are defined such that \f$P\f$, \f$x_0^i\f$, and \f$x^i\f$ are co-linear.
 * \f$x_0^i\f$ is determined by the equation
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda},\f}
 *
 * where \f$\tilde{\lambda}\f$ is a scalar factor that depends on
 * \f$x^i\f$ and is determined by the inner map.  The inner map
 * provides a function `lambda_tilde` that takes \f$x^i\f$ and
 * \f$P^i\f$ as arguments and returns \f$\tilde{\lambda}\f$ (or
 * boost::none if the appropriate \f$\tilde{\lambda}\f$ cannot be
 * found; a value of boost::none indicates that the point \f$x^i\f$ is
 * outside the range of the map).
 *
 * Now consider the coordinates \f$x_1^i\f$ that lie on the sphere
 * and are defined such that \f$P\f$, \f$x_1^i\f$, and \f$x^i\f$ are co-linear.
 * \f$x_1^i\f$ is determined by the equation
 *
 * \f{align} x_1^i = P^i + (x^i - P^i) \bar{\lambda},\f}
 *
 * where \f$\bar{\lambda}\f$ is a scalar factor that depends on \f$x^i\f$ and
 * is the solution of a quadratic
 * that is derived by demanding that \f$x_1^i\f$ lies on the sphere:
 *
 * \f{align}
 * |P^i + (x^i - P^i) \bar{\lambda} - C^i |^2 - R^2 = 0.
 * \f}
 *
 * Note that we don't actually need to compute \f$x_1^i\f$. Instead, we
 * can determine \f$\sigma\f$ by the relation
 *
 * \f{align}
 * \sigma = \frac{\tilde{\lambda}-1}{\tilde{\lambda}-\bar{\lambda}}.
 * \f}
 *
 * Once we have \f$x_0^i\f$ and \f$\sigma\f$, the point
 * \f$(\bar{x},\bar{y},\bar{z})\f$ is uniquely determined by the inner
 * map.
 * The `inverse` function of the inner map takes \f$x_0^i\f$ and \f$\sigma\f$
 * as arguments, and returns
 * \f$(\bar{x},\bar{y},\bar{z})\f$, or boost::none if \f$x_0^i\f$ or
 * \f$\sigma\f$ are outside the range of the map.
 *
 * #### Root polishing
 *
 * The inverse function described above will sometimes have errors that
 * are noticeably larger than roundoff.  Therefore we apply a single
 * Newton-Raphson iteration to refine the result of the inverse map:
 * Suppose we are given \f$x^i\f$, and we have computed \f$\bar{x}^i\f$
 * by the above procedure.  We then correct \f$\bar{x}^i\f$ by adding
 *
 * \f[
 * \delta \bar{x}^i = \left(x^j - x^j(\bar{x})\right)
 * \frac{\partial \bar{x}^i}{\partial x^j},
 * \f]
 *
 * where \f$x^j(\bar{x})\f$ is the result of applying the forward map
 * to \f$\bar{x}^i\f$ and \f$\partial \bar{x}^i/\partial x^j\f$ is the
 * inverse jacobian.
 *
 * ### Inverse jacobian
 *
 * We write the inverse Jacobian as
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial x^j} =
 * \frac{\partial \bar{x}^i}{\partial x_0^k}
 * \frac{\partial x_0^k}{\partial x^j}
 * + \frac{\partial \bar{x}^i}{\partial \sigma}
 *   \frac{\partial \sigma}{\partial x^j},
 * \f}
 *
 * where we have recognized that \f$\bar{x}^i\f$ depends both on
 * \f$x_0^k\f$ (the corresponding point on the 2-surface) and on
 * \f$\sigma\f$ (encoding the distance away from the 2-surface).
 *
 * We now evaluate Eq. (9). The inner map provides a function
 * `inv_jacobian` that returns \f$\partial \bar{x}^i/\partial x_0^k\f$
 * (where \f$\sigma\f$ is held fixed), and a function `dxbar_dsigma`
 * that returns \f$\partial \bar{x}^i/\partial \sigma\f$ (where
 * \f$x_0^i\f$ is held fixed).  The factor \f$\partial x_0^j/\partial
 * x^i\f$ can be computed by differentiating Eq. (5), which yields
 *
 * \f{align}
 * \frac{\partial x_0^j}{\partial x^i} &= \tilde{\lambda} \delta_i^j
 * + \frac{x_0^j-P^j}{\tilde{\lambda}}
 *   \frac{\partial\tilde{\lambda}}{\partial x^i},
 * \f}
 *
 * where \f$\partial \tilde{\lambda}/\partial x^i\f$ is provided
 * by the `deriv_lambda_tilde` function of the inner map.
 *
 * To evaluate the remaining unknown factor in Eq. (9),
 * \f$\partial \sigma/\partial x^j\f$,
 * note that \f$\bar{\lambda}=\tilde{\lambda}\lambda\f$.
 * Therefore Eq. (8) is equivalent to
 *
 * \f{align}
 * \sigma &= \frac{\tilde{\lambda}-1}{\tilde{\lambda}(1-\lambda)}.
 * \f}
 *
 * Differentiating this expression yields
 *
 * \f{align}
 * \frac{\partial \sigma}{\partial x^i} &=
 * \frac{\partial \sigma}{\partial \lambda}
 * \frac{\partial \lambda}{\partial x_0^j}
 * \frac{\partial x_0^j}{\partial x^i}
 * + \frac{\partial \sigma}{\partial \tilde\lambda}
 *   \frac{\partial \tilde\lambda}{\partial x^i}\\
 * &=
 * \frac{\sigma}{1-\lambda}
 * \frac{\partial \lambda}{\partial x_0^j}
 * \frac{\partial x_0^j}{\partial x^i}
 * +
 * \frac{1}{\tilde{\lambda}^2(1-\lambda)}
 * \frac{\partial \tilde\lambda}{\partial x^i},
 * \f}
 *
 * where the second factor in the first term can be evaluated using Eq. (4).
 *
 */
template <typename InnerMap>
class FocallyLiftedMap {
 public:
  static constexpr size_t dim = 3;
  FocallyLiftedMap(const std::array<double, 3>& center,
                   const std::array<double, 3>& proj_center, double radius,
                   InnerMap inner_map) noexcept;

  FocallyLiftedMap() = default;
  ~FocallyLiftedMap() = default;
  FocallyLiftedMap(FocallyLiftedMap&&) = default;
  FocallyLiftedMap(const FocallyLiftedMap&) = default;
  FocallyLiftedMap& operator=(const FocallyLiftedMap&) = default;
  FocallyLiftedMap& operator=(FocallyLiftedMap&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  friend bool operator==
      <InnerMap>(const FocallyLiftedMap<InnerMap>& lhs,
                 const FocallyLiftedMap<InnerMap>& rhs) noexcept;
  std::array<double, 3> center_{}, proj_center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  InnerMap inner_map_;
};
template <typename InnerMap>
bool operator!=(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept;
}  // namespace domain::CoordinateMaps
