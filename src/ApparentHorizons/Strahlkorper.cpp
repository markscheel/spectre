// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Strahlkorper.hpp"

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(const size_t l_max, const size_t m_max,
                                  const double radius,
                                  std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(center)),  // NOLINT
      strahlkorper_coefs_(ylm_.spectral_size(), 0.0) {
  ylm_.add_constant(&strahlkorper_coefs_, radius);
}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(
    const size_t l_max, const size_t m_max,
    const Strahlkorper& another_strahlkorper) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(another_strahlkorper.ylm_.prolong_or_restrict(
          another_strahlkorper.strahlkorper_coefs_, ylm_)) {}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(const Strahlkorper& another_strahlkorper,
                                  DataVector coefs) noexcept
    : l_max_(another_strahlkorper.l_max_),
      m_max_(another_strahlkorper.m_max_),
      ylm_(another_strahlkorper.ylm_),
      center_(another_strahlkorper.center_),
      strahlkorper_coefs_(std::move(coefs)) {
  ASSERT(
      strahlkorper_coefs_.size() == another_strahlkorper.ylm_.spectral_size(),
      "Bad size " << strahlkorper_coefs_.size() << ", expected "
                  << another_strahlkorper.ylm_.spectral_size());
}

template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(
    const DataVector& radius_at_collocation_points, const size_t l_max,
    const size_t m_max, std::array<double, 3> center) noexcept
    : l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      // clang-tidy: do not std::move trivially constructable types
      center_(std::move(center)),  // NOLINT
      strahlkorper_coefs_(ylm_.phys_to_spec(radius_at_collocation_points)) {
  ASSERT(radius_at_collocation_points.size() == ylm_.physical_size(),
         "Bad size " << radius_at_collocation_points.size() << ", expected "
                     << ylm_.physical_size());
}

template <typename Frame>
void Strahlkorper<Frame>::pup(PUP::er& p) {
  p | l_max_;
  p | m_max_;
  p | center_;
  p | strahlkorper_coefs_;

  if (p.isUnpacking()) {
    ylm_ = YlmSpherepack(l_max_, m_max_);
  }
}

template <typename Frame>
std::array<double, 3> Strahlkorper<Frame>::physical_center() const noexcept {
  // Uses Eqs. (38)-(40) in Hemberger et al, arXiv:1211.6079.  This is
  // an approximation of Eq. (37) in the same paper, which gives the
  // exact result.
  std::array<double, 3> result = center_;
  SpherepackIterator it(l_max_, m_max_);
  result[0] += strahlkorper_coefs_[it.set(1, 1)()] * sqrt(0.75);
  result[1] -= strahlkorper_coefs_[it.set(1, -1)()] * sqrt(0.75);
  result[2] += strahlkorper_coefs_[it.set(1, 0)()] * sqrt(0.375);
  return result;
}

template <typename Frame>
double Strahlkorper<Frame>::average_radius() const noexcept {
  return ylm_.average(coefficients());
}

template <typename Frame>
double Strahlkorper<Frame>::radius(const double theta, const double phi) const
    noexcept {
  return ylm_.interpolate_from_coefs(strahlkorper_coefs_, {{{theta, phi}}})[0];
}

template <typename Frame>
bool Strahlkorper<Frame>::point_is_contained(
    const std::array<double, 3>& x) const noexcept {
  // The point `x` is assumed to be in Cartesian coords in the
  // Strahlkorper frame.

  // Make the point relative to the center of the Strahlkorper.
  auto xmc = x;
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(xmc, d) -= gsl::at(center_, d);
  }

  // Convert the point from Cartesian to spherical coordinates.
  const double r = magnitude(xmc);
  const double theta = atan2(sqrt(square(xmc[0]) + square(xmc[1])), xmc[2]);
  double phi = atan2(xmc[1], xmc[0]);
  // Range of atan2 is [-pi,pi],
  // So adjust to get range [0,2pi]
  if (phi < 0.0) {
    phi += 2.0 * M_PI;
  }

  // Is the point inside the surface?
  return r < radius(theta, phi);
}

template class Strahlkorper<Frame::Inertial>;
