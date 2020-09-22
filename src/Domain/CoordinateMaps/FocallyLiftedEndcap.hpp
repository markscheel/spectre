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

/// Contains FocallyLiftedInnerMaps
namespace domain::CoordinateMaps::FocallyLiftedInnerMaps {
/*!
 * \brief A FocallyLiftedInnerMap that maps a 3D unit right cylinder
 *  to a volume that connects portions of two spherical surfaces.
 *
 * \image html FocallyLiftedEndcap.svg "Focally Lifted Endcap."
 *
 * \details The domain of the map is a 3D unit right cylinder with
 * coordinates \f$(\bar{x},\bar{y},\bar{z})\f$ such that
 * \f$-1\leq\bar{z}\leq 1\f$ and \f$\bar{x}^2+\bar{y}^2 \leq
 * 1\f$.  The range of the map has coordinates \f$(x,y,z)\f$.
 *
 * Consider a sphere with center \f$C^i\f$ and radius \f$R\f$ that is
 * intersected by a plane normal to the \f$z\f$ axis and located at
 * \f$z = z_\mathrm{P}\f$.  In the figure above, every point
 * \f$\bar{x}^i\f$ in the blue region \f$\sigma=0\f$ maps to a point
 * \f$x_0^i\f$ on a portion of the surface of the sphere.
 *
 * `Endcap` provides the following functions:
 *
 * ### operator()
 * `operator()` maps \f$(\bar{x},\bar{y},\bar{z}=-1)\f$ to the portion of
 * the sphere with \f$z \geq z_\mathrm{P}\f$.  The arguments to `operator()`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 * `operator()` returns \f$x_0^i\f$,
 * the 3D coordinates on that sphere, which are given by
 *
 * \f{align}
 * x_0^0 &= R \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{x}}{\bar{\rho}} + C^0,\\
 * x_0^1 &= R \frac{\sin(\bar{\rho} \theta_\mathrm{max})
 *              \bar{y}}{\bar{\rho}} + C^1,\\
 * x_0^2 &= R \cos(\bar{\rho} \theta_\mathrm{max}) + C^2.
 * \f}
 *
 * Here \f$\bar{\rho}^2 \equiv (\bar{x}^2+\bar{y}^2)/\bar{R}^2\f$, where
 * \f$\bar{R}\f$ is the radius of the cylinder in barred coordinates,
 * which is always unity,
 * and where
 * \f$\theta_\mathrm{max}\f$ is defined by
 * \f$\cos(\theta_\mathrm{max}) = (z_\mathrm{P}-C^2)/R\f$.
 * Note that when \f$\bar{\rho}=0\f$, we must evaluate
 * \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * as \f$\theta_\mathrm{max}\f$.
 *
 * ### sigma
 *
 * \f$\sigma\f$ is a function that is zero at \f$\bar{z}=-1\f$
 * (which maps onto the sphere \f$x^i=x_0^i\f$) and
 * unity at \f$\bar{z}=+1\f$ (corresponding to the
 * upper surface of the FocallyLiftedMap). We define
 *
 * \f{align}
 *  \sigma &= \frac{\bar{z}+1}{2}.
 * \f}
 *
 * ### deriv_sigma
 *
 * `deriv_sigma` returns
 *
 * \f{align}
 *  \frac{\partial \sigma}{\partial \bar{x}^j} &= (0,0,1/2).
 * \f}
 *
 * ### jacobian
 *
 * `jacobian` returns \f$\partial x_0^k/\partial \bar{x}^j\f$.
 * The arguments to `jacobian`
 * are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * Differentiating Eqs.(1--3) above yields
 *
 * \f{align*}
 * \frac{\partial x_0^2}{\partial \bar{x}} &=
 * - R \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{x}\\
 * \frac{\partial x_0^2}{\partial \bar{y}} &=
 * - R \theta_\mathrm{max}
 * \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\bar{y}\\
 * \frac{\partial x_0^0}{\partial \bar{x}} &=
 * R \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}^2\\
 * \frac{\partial x_0^0}{\partial \bar{y}} &=
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{x}} &=
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{x}\bar{y}\\
 * \frac{\partial x_0^1}{\partial \bar{y}} &=
 * R \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}} +
 * R \frac{1}{\bar{\rho}}\frac{d}{d\bar{\rho}}
 * \left(\frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}\right)
 * \bar{y}^2\\
 * \frac{\partial x_0^i}{\partial \bar{z}} &=0,
 * \f}
 * where care must be taken to evaluate
 * \f$\sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$
 * and its derivatives near \f$\bar{\rho}=0\f$.
 *
 * ### inverse
 *
 * `inverse` takes \f$x_0^i\f$ and \f$\sigma\f$ as arguments, and
 * returns \f$(\bar{x},\bar{y},\bar{z})\f$, or boost::none if
 * \f$x_0^i\f$ or \f$\sigma\f$ are outside the range of the map.
 * For example, if \f$x_0^i\f$ does not lie on the sphere,
 * we return boost::none.
 *
 * The easiest to compute is \f$\bar{z}\f$, which is given by inverting
 * Eq. (4):
 *
 * \f{align}
 *  \bar{z} &= 2\sigma - 1.
 * \f}
 *
 * If \f$\bar{z}\f$ is outside the range \f$[-1,1]\f$ then we return
 * boost::none.
 *
 * To get \f$\bar{x}\f$ and \f$\bar{y}\f$,
 * we invert
 * Eqs (1--3).  If \f$x_0^0=x_0^1=0\f$, then \f$\bar{x}=\bar{y}=0\f$.
 * Otherwise, we compute
 *
 * \f{align}
 *   \bar{\rho} = \theta_\mathrm{max}^{-1}
 *   \tan^{-1}\left(\frac{\rho}{x_0^2-C^2}\right),
 * \f}
 *
 * where \f$\rho^2 = (x_0^0-C^0)^2+(x_0^1-C^1)^2\f$. Then
 *
 * \f{align}
 * \bar{x} &= (x_0^0-C^0)\frac{\bar{\rho}}{\rho},\\
 * \bar{y} &= (x_0^1-C^1)\frac{\bar{\rho}}{\rho}.
 * \f}
 *
 * Note that if \f$\bar{x}^2+\bar{y}^2 > 1\f$, the original point is outside
 * the range of the map so we return boost::none.
 *
 * ### lambda_tilde
 *
 * `lambda_tilde` takes as arguments a point \f$x^i\f$ and a projection point
 *  \f$P^i\f$, and computes \f$\tilde{\lambda}\f$, the solution to
 *
 * \f{align} x_0^i = P^i + (x^i - P^i) \tilde{\lambda}.\f}
 *
 * Since \f$x_0^i\f$ must lie on the sphere, \f$\tilde{\lambda}\f$ is the
 * solution of the quadratic equation
 *
 * \f{align}
 * |P^i + (x^i - P^i) \tilde{\lambda} - C^i |^2 - R^2 = 0.
 * \f}
 *
 * In solving the quadratic, we choose the larger root if
 * \f$x^2>z_\mathrm{P}\f$ and the smaller root otherwise. We demand
 * that the root is greater than unity.  If there is no such root,
 * this means that the point \f$x^i\f$ is not in the range of the map
 * so we return boost::none.
 *
 * ### deriv_lambda_tilde
 *
 * `deriv_lambda_tilde` takes as arguments \f$x_0^i\f$, a projection point
 *  \f$P^i\f$, and \f$\tilde{\lambda}\f$, and
 *  returns \f$\partial \tilde{\lambda}/\partial x^i\f$.
 * By differentiating Eq. (11), we find
 *
 * \f{align}
 * \frac{\partial\tilde{\lambda}}{\partial x^j} &=
 * \tilde{\lambda}^2 \frac{C^j - x_0^j}{|x_0^i - P^i|^2
 * + (x_0^i - P^i)(P_i - C_{i})}.
 * \f}
 *
 * ### inv_jacobian
 *
 * `inv_jacobian` returns \f$\partial \bar{x}^i/\partial x_0^k\f$,
 *  where \f$\sigma\f$ is held fixed.
 *  The arguments to `inv_jacobian`
 *  are \f$(\bar{x},\bar{y},\bar{z})\f$, but \f$\bar{z}\f$ is ignored.
 *
 * Note that \f$\bar{x}\f$ and \f$\bar{y}\f$ can be considered to
 * depend only on \f$x_0^0\f$ and \f$x_0^1\f$ but not on \f$x_0^2\f$,
 * because the point \f$x_0^i\f$ is constrained to lie on a sphere of
 * radius \f$R\f$.  Note that there is an alternative way to compute
 * Eqs. (8) and (9) using only \f$x_0^0\f$ and \f$x_0^1\f$. To do
 * this, define
 *
 * \f{align}
 * \upsilon \equiv \sin(\bar{\rho}\theta_\mathrm{max})
 *        &= \sqrt{\frac{(x_0^0-C^0)^2+(x_0^1-C^1)^2}{R^2}}.
 * \f}
 *
 * Then we can write
 *
 * \f{align}
 * \frac{1}{\bar{\rho}}\sin(\bar{\rho}\theta_\mathrm{max})
 * &= \frac{\theta_\mathrm{max}\upsilon}{\arcsin(\upsilon)},
 * \f}
 *
 * so that
 *
 * \f{align}
 * \bar{x} &= \frac{x_0^0-C^0}{R}\left(\frac{1}{\bar{\rho}}
 *             \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1} \\
 * \bar{y} &= \frac{x_0^1-C^1}{R}\left(\frac{1}{\bar{\rho}}
 *             \sin(\bar{\rho}\theta_\mathrm{max})\right)^{-1}.
 * \f}
 *
 * We will compute \f$\partial \bar{x}^i/\partial
 * x_0^k\f$ by differentiating Eqs. (15) and (16).  Because those equations
 * involve \f$\bar{\rho}\f$, we first establish some relations
 * involving derivatives of \f$\bar{\rho}\f$.  For ease of notation, we define
 *
 * \f[
 * q \equiv \frac{\sin(\bar{\rho}\theta_\mathrm{max})}{\bar{\rho}}.
 * \f]
 *
 * First observe that
 * \f[
 * \frac{dq}{d\upsilon}
 * = \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f]
 *
 * where \f$\upsilon\f$ is the quantity defined by Eq. (13).  Therefore
 *
 * \f{align}
 * \frac{\partial q}{\partial x_0^0} &=
 * \frac{\bar{x}}{\bar{\rho}}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial q}{\partial x_0^1} &=
 * \frac{\bar{y}}{\bar{\rho}}\frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},
 * \f}
 *
 * where we have differentiated Eq. (13), and where we have
 * used Eqs. (15) and (16) to eliminate \f$x_0^0\f$ and
 * \f$x_0^1\f$ in favor of \f$\bar{x}\f$ and
 * \f$\bar{y}\f$ in the final result.
 *
 * By differentiating Eqs. (15) and (16), and using Eqs. (17) and (18), we
 * find
 *
 * \f{align*}
 * \frac{\partial \bar{x}}{\partial x_0^0} &=
 * \frac{1}{R q}
 * - \frac{\bar{x}^2}{R q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial \bar{x}}{\partial x_0^1} &=
 * - \frac{\bar{x}\bar{y}}{R q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial \bar{x}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{y}}{\partial x_0^0} &=
 * \frac{\partial \bar{x}}{\partial x_0^1},\\
 * \frac{\partial \bar{y}}{\partial x_0^1} &=
 * \frac{1}{R q}
 * - \frac{\bar{y}^2}{R q \bar{\rho}} \frac{dq}{d\bar{\rho}}
 * \left(\bar{\rho} \frac{dq}{d\bar{\rho}} + q\right)^{-1},\\
 * \frac{\partial \bar{y}}{\partial x_0^2} &= 0,\\
 * \frac{\partial \bar{z}}{\partial x_0^i} &= 0.
 * \f}
 * Note that care must be taken to evaluate
 * \f$q = \sin(\bar{\rho}\theta_\mathrm{max})/\bar{\rho}\f$ and its
 * derivative near \f$\bar{\rho}=0\f$.
 *
 * ### dxbar_dsigma
 *
 * `dxbar_dsigma` returns \f$\partial \bar{x}^i/\partial \sigma\f$,
 *  where \f$x_0^i\f$ is held fixed.
 *
 * From Eq. (6) we have
 *
 * \f{align}
 * \frac{\partial \bar{x}^i}{\partial \sigma} &= (0,0,2).
 * \f}
 *
 */
class Endcap {
 public:
  Endcap(const std::array<double, 3>& center, double radius,
         double z_plane) noexcept;

  Endcap() = default;
  ~Endcap() = default;
  Endcap(Endcap&&) = default;
  Endcap(const Endcap&) = default;
  Endcap& operator=(const Endcap&) = default;
  Endcap& operator=(Endcap&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double sigma_in) const
      noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tt::remove_cvref_wrap_t<T> sigma(const std::array<T, 3>& source_coords) const
      noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> deriv_sigma(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> dxbar_dsigma(
      const std::array<T, 3>& source_coords) const noexcept;

  boost::optional<double> lambda_tilde(
      const std::array<double, 3>& parent_mapped_target_coords,
      const std::array<double, 3>& projection_point) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> deriv_lambda_tilde(
      const std::array<T, 3>& target_coords, const T& lambda_tilde,
      const std::array<double, 3>& projection_point) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

  static bool projection_source_is_between_focus_and_target() noexcept {
    return true;
  }

 private:
  friend bool operator==(const Endcap& lhs, const Endcap& rhs) noexcept;
  std::array<double, 3> center_{};
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double theta_{std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Endcap& lhs, const Endcap& rhs) noexcept;
}  // namespace domain::CoordinateMaps::FocallyLiftedInnerMaps
