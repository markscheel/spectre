// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdint>

#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ForceInline.hpp"

/// \ingroup SurfacesGroup
/// \brief A star-shaped surface expanded in spherical harmonics.
template <typename Frame>
class Strahlkorper {
 public:
  using ThetaPhiVector = tnsr::i<DataVector, 2, ::Frame::Spherical<Frame>>;
  using OneForm = tnsr::i<DataVector, 3, Frame>;
  using ThreeVector = tnsr::I<DataVector, 3, Frame>;
  using Jacobian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Up, Frame>,
                        SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<Frame>>>>;
  using InvJacobian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                        SpatialIndex<3, UpLo::Lo, Frame>>>;
  using InvHessian =
      Tensor<DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
             index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Frame>>,
                        SpatialIndex<3, UpLo::Lo, Frame>,
                        SpatialIndex<3, UpLo::Lo, Frame>>>;
  using SecondDeriv = tnsr::ii<DataVector, 3, Frame>;
  using InverseMetric = tnsr::II<DataVector, 3, Frame>;

  // Pup needs default constructor
  Strahlkorper() = default;

  /// Construct a sphere of radius r with a given center.
  Strahlkorper(size_t l_max, size_t m_max, double radius,
               std::array<double, 3> center) noexcept;
  /// Prolong or restrict another surface to the given l_max and m_max.
  Strahlkorper(size_t l_max, size_t m_max,
               const Strahlkorper& another_strahlkorper) noexcept;
  /// Construct a Strahlkorper from another Strahlkorper,
  /// but explicitly specifying the coefficients.
  /// Here coefficients are in the same storage scheme
  /// as the `coefficients` member function returns.
  Strahlkorper(const Strahlkorper& another_strahlkorper,
               DataVector coefs) noexcept;
  /// Construct a Strahlkorper from a DataVector containing the radius
  /// at the collocation points.
  Strahlkorper(const DataVector& radius_at_collocation_points, size_t l_max,
               size_t m_max, std::array<double, 3> center) noexcept;

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

  /*!
   *  These coefficients are stored as Spherepack coefficients.
   *  Suppose you represent a set of coefficients \f$F^{lm}\f$ in the expansion
   *  \f[
   *  f(\theta,\phi) =
   *  \sum_{l=0}^{l_max} \sum_{m=-l}^{l} F^{lm} Y^{lm}(\theta\phi)
   *  \f]
   *  Here the \f$Y^{lm}(\theta\phi)\f$ are the usual complex-valued scalar
   *  spherical harmonics, so \f$F^{lm}\f$ are also complex-valued.
   *  But here we assume that \f$f(\theta,\phi)\f$ is real, so therefore
   *  the \f$F^{lm}\f$ obey \f$F^{l-m} = (-1)^m (F^{lm})^\star\f$, so one
   *  does not need to store both real and imaginary parts.
   *  So:
   *  strahlkorper_coefs_(l,m) = \f$(-1)^m Q Re(F^{lm})\f$ for \f$m\ge 0\f$
   *  strahlkorper_coefs_(l,m) = \f$(-1)^m Q Im(F^{lm})\f$ for \f$m<   0\f$
   *  Where Q = sqrt(2.0/M_PI)
   */
  SPECTRE_ALWAYS_INLINE const DataVector& coefficients() const noexcept {
    return strahlkorper_coefs_;
  }

  /// Point about which the spectral basis of the Strahlkorper is expanded.
  /// The center is given in the frame in which the Strahlkorper is defined.
  /// This center must be somewhere inside the Strahlkorper, but in principle
  /// it can be anywhere.  See physical_center for a different measure.
  SPECTRE_ALWAYS_INLINE const std::array<double, 3>& center() const noexcept {
    return center_;
  }

  /// Approximate physical center (determined by l=1 coefficients)
  /// Implementation of Eqs. (38)-(40) in Hemberger et al, arXiv:1211.6079
  std::array<double, 3> physical_center() const noexcept;

  /// Average radius of the surface (determined by l=0 Ylm coefficient)
  double average_radius() const noexcept;

  /// Maximum \f$l\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t l_max() const noexcept { return l_max_; }

  /// Maximum \f$m\f$ in \f$Y_{lm}\f$ decomposition.
  SPECTRE_ALWAYS_INLINE size_t m_max() const noexcept { return m_max_; }

  /// Radius at a particular angle \f$(\theta,\phi)\f$.
  /// This is inefficient if done at multiple points many times.
  /// See YlmSpherepack for alternative ways of computing this.
  double radius(double theta, double phi) const noexcept;

  /// Determine if a point `x` is contained inside the surface.
  /// The point must be given in Cartesian coordinates in the frame in
  /// which the Strahlkorper is defined and relative to the origin `center`.
  /// This is inefficient if done at multiple points many times.
  bool point_is_contained(const std::array<double, 3>& x) const noexcept;

  SPECTRE_ALWAYS_INLINE const YlmSpherepack& ylm_spherepack() const noexcept {
    return ylm_;
  }

 private:
  size_t l_max_{2}, m_max_{2};
  YlmSpherepack ylm_{2, 2};
  std::array<double, 3> center_{{0.0, 0.0, 0.0}};
  DataVector strahlkorper_coefs_ = DataVector(ylm_.physical_size(), 0.0);
};

template <typename Frame>
bool operator==(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max() and
         lhs.center() == rhs.center() and
         lhs.coefficients() == rhs.coefficients();
}

template <typename Frame>
bool operator!=(const Strahlkorper<Frame>& lhs,
                const Strahlkorper<Frame>& rhs) {
  return not(lhs == rhs);
}
