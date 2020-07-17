// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// Holds helper functions for use with
/// domain::CoordinateMaps::FocallyLiftedMap.
namespace domain::CoordinateMaps::FocallyLiftedMapHelpers {

/*!
 * \brief Finds how long to extend a line segment to have it intersect
 * a point on a 2-sphere.
 *
 * \details Consider a 2-sphere with center \f$C\f$ and radius \f$R\f$, and
 * and let \f$P\f$ and \f$x_0\f$ be two arbitrary 3D points.
 *
 * Consider the line passing through \f$P\f$ and \f$x_0\f$.
 * If this line intersects the sphere at a point \f$x\f$, then we can write
 *
 * \f[
 *  x = P + (x_0-P) \lambda,
 * \f]
 *
 * where \f$\lambda\f$ is a scale factor.
 *
 * `scale_factor` computes and returns \f$\lambda\f$.
 *
 * ### Even more detail:
 *
 * To solve for \f$\lambda\f$, we note that \f$x\f$ is on the surface of
 * the sphere, so
 *
 * \f[
 *  |x-C|^2 = R^2,
 * \f]
 *
 *  or equivalently
 *
 * \f[
 *  | P-C + (x_0-P)\lambda |^2 = R^2.
 * \f]
 *
 * This is a quadratic equation for \f$\lambda\f$
 * and it generally has more than one real root.
 *
 * So how do we choose between multiple roots?  Some of the maps that
 * use `scale_factor` assume that *for all points*, \f$x_0\f$ is
 * between \f$P\f$ and \f$x\f$.  Those maps should set the parameter
 * `src_is_between_proj_and_target` to true. Other maps assume that
 * *for all points*, \f$x\f$ is always between src_point
 * and proj_center. Those maps should set the parameter
 * `src_is_between_proj_and_target` to false.
 *
 * Note: if we ever add maps where
 * `src_is_between_proj_and_target` can change from point to point,
 * the logic of `scale_factor` needs to be changed.
 *
 */
template <typename T>
tt::remove_cvref_wrap_t<T> scale_factor(
    const std::array<T, 3>& src_point, const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    bool src_is_between_proj_and_target) noexcept;

/*!
 *  Same as `scale_factor` but has additional options as to which root
 *  to choose. `try_scale_factor` returns boost::none if the roots
 *  are not as expected (i.e. if the inverse map was called for a
 *  point not in the range of the map).
 */
boost::optional<double> try_scale_factor(
    const std::array<double, 3>& src_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    bool pick_larger_root, bool pick_root_greater_than_one) noexcept;

/*!
 * Computes \f$\partial \lambda/\partial x^i\f$, where \f$\lambda\f$
 * is the quantity returned by `scale_factor`.
 *
 * Note that it takes `intersection_point` and not `src_point` as a
 * parameter.
 */
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> d_scale_factor_d_src_point(
    const std::array<T, 3>& intersection_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, const T& lambda) noexcept;

}  // namespace domain::CoordinateMaps::FocallyLiftedMapHelpers
