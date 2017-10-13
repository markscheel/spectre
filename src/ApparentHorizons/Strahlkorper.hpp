// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "YlmSpherepack.hpp"

/// \brief A star-shaped surface expanded in spherical harmonics.
template <class Fr>
class Strahlkorper {
 public:
  using ThetaPhiVector = tnsr::i<DataVector, 2, Fr>;
  using OneForm = tnsr::i<DataVector, 3, Fr>;
  using ThreeVector = tnsr::I<DataVector, 3, Fr>;
  using Jacobian = Tensor<
      DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Fr>, SpatialIndex<2, UpLo::Lo, Fr>>>;
  using InvJacobian = Tensor<
      DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Fr>, SpatialIndex<3, UpLo::Lo, Fr>>>;
  using InvHessian = Tensor<
      DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Fr>, SpatialIndex<3, UpLo::Lo, Fr>,
                 SpatialIndex<3, UpLo::Lo, Fr>>>;
  using SecondDeriv = tnsr::ii<DataVector, 3, Fr>;
  using InverseMetric = tnsr::II<DataVector, 3, Fr>;

  /// Pup needs default constructor
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
  /// as the Coefficients() member function returns.
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
   *  mStrahlkorperCoefs(l,m) = \f$(-1)^m Q Re(F^{lm})\f$ for \f$m\ge 0\f$
   *  mStrahlkorperCoefs(l,m) = \f$(-1)^m Q Im(F^{lm})\f$ for \f$m<   0\f$
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

  /// Spherical Harmonic decomposition max size l_max
  SPECTRE_ALWAYS_INLINE size_t l_max() const noexcept { return l_max_; }

  /// Spherical Harmonic decomposition max size m_max
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

  // The following 5 functions of coords don't depend on the actual shape
  // of the surface, but only on l_max and m_max.

  /// \f$(\theta,\phi)\f$ on the grid.
  const ThetaPhiVector& theta_phi() const noexcept;
  /// \f$x_i/r\f$
  const OneForm& r_hat() const noexcept;
  /// jacobian(i,j) is \f$\frac{1,r}\partial x^i/\partial\theta\f$,
  /// \f$\frac{1,r\sin\theta}\partial x^i/\partial\phi\f$
  const Jacobian& jacobian() const noexcept;
  /// inv_jacobian(j,i) is \f$r\partial\theta/\partial x^i\f$,
  ///    \f$r\sin\theta\partial\phi/\partial x^i\f$
  const InvJacobian& inv_jacobian() const noexcept;
  /// inv_hessian(k,i,j) is \f$\partial\f$ijac(k,j)\f$/partial x^i\f$,
  /// where \f$ijac\f$ is the inverse Jacobian.
  /// It is not symmetric because the Jacobians are Pfaffian.
  const InvHessian& inv_hessian() const noexcept;

  // The following 7 functions do depend on the shape of the surface.

  /// (Euclidean) distance r of each grid point from center.
  const DataVector& radius() const noexcept;
  /// dx_radius(i) is \f$\partial r/\partial x^i\f$
  const OneForm& dx_radius() const noexcept;
  /// d2x_radius(i,j) is \f$\partial^2 r/\partial x^i\partial x^j\f$
  const SecondDeriv& d2x_radius() const noexcept;
  /// \f$\nabla^2 radius\f$
  const DataVector& nabla_squared_radius() const noexcept;
  /// \f$(x,y,z)\f$ of each point on the surface.
  const ThreeVector& surface_cartesian_coords() const noexcept;
  /// Cartesian components of (unnormalized) one-form defining the surface.
  /// This is computed by \f$x_i/r-\partial r/\partial x^i\f$,
  /// where \f$x_i/r\f$ is `r_hat` and
  /// \f$\partial r/\partial x^i\f$ is `dx_radius`.
  const OneForm& surface_normal_one_form() const noexcept;
  /// surface_tangents[j](i) is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
  /// where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the surface
  /// (i.e. `surface_cartesian_coords`)
  /// and are considered functions of \f$(\theta,\phi)\f$,
  /// \f$\partial/\partial q^0\f$ means \f$\partial/\partial\theta\f$,
  /// and \f$\partial/\partial q^1\f$ means
  /// \f$\csc\theta\partial/\partial\phi\f$.
  const std::array<ThreeVector, 2>& surface_tangents() const noexcept;

  /// Magnitude of surface_normal_one_form.
  /// This routine needs an upper 3-metric in cartesian coordinates.
  /// The normalized one-form to the surface is
  /// surface_normal_one_form(i)/surface_normal_magnitude
  DataVector surface_normal_magnitude(const InverseMetric& inv_g) const
      noexcept;

 private:
  void initialize_mesh_quantities() const noexcept;
  void initialize_jac_and_hess() const noexcept;

  size_t l_max_{2}, m_max_{2};
  YlmSpherepack ylm_{2, 2};
  std::array<double, 3> center_{{0.0, 0.0, 0.0}};
  DataVector strahlkorper_coefs_ = DataVector(ylm_.physical_size(), 0.0);

  // The following 5 variables are coordinate quantities on the 2D mesh,
  // centered at the origin.  They don't depend on the actual shape of
  // the surface, just on l_max_ and m_max_.
  // They are mutable because sometimes they don't need to be computed at
  // all, so they are computed once only when needed and then saved.

  mutable ThetaPhiVector theta_phi_;
  mutable OneForm r_hat_;
  mutable Jacobian jac_;
  mutable InvJacobian inv_jac_;
  mutable InvHessian inv_hess_;

  // The following variables do depend on the shape of the surface.
  // They are computed only when needed, which is why they are mutable.

  mutable DataVector radius_;
  mutable ThreeVector global_coords_;
  mutable OneForm dx_radius_;
  mutable SecondDeriv d2x_radius_;
  mutable DataVector nabla_squared_radius_;
  mutable OneForm surface_normal_one_form_;
  mutable std::array<ThreeVector, 2> surface_tangents_;
};

template <typename Fr>
bool operator==(const Strahlkorper<Fr>& lhs, const Strahlkorper<Fr>& rhs) {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max() and
         lhs.center() == rhs.center() and
         lhs.coefficients() == rhs.coefficients();
  // We don't care about any other member variables because they
  // are completely determined by the ones we check above.
}

template <typename Fr>
bool operator!=(const Strahlkorper<Fr>& lhs, const Strahlkorper<Fr>& rhs) {
  return not(lhs == rhs);
}
