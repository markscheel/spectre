// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"

#include <boost/none.hpp>
#include <cmath>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

template <typename InnerMap>
FocallyLiftedMap<InnerMap>::FocallyLiftedMap(
    const std::array<double, 3>& center,
    const std::array<double, 3>& proj_center, double radius,
    InnerMap inner_map) noexcept
    : center_(center),
      proj_center_(proj_center),
      radius_(radius),
      inner_map_(std::move(inner_map)) {}

template <typename InnerMap>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> FocallyLiftedMap<InnerMap>::
operator()(const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // lower_coords are the mapped coords on the surface.
  const std::array<ReturnType, 3> lower_coords = inner_map_(source_coords);

  // upper_coords are the mapped coords on the surface of the sphere.
  const auto lambda = FocallyLiftedMapHelpers::scale_factor<ReturnType>(
      lower_coords, proj_center_, center_, radius_,
      InnerMap::projection_source_is_between_focus_and_target());

  std::array<ReturnType, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // mapped_coords goes linearly from lower_coords to upper_coords
  // as sigma goes from 0 to 1.
  const ReturnType sigma = inner_map_.sigma(source_coords);
  auto mapped_coords = make_with_value<std::array<ReturnType, 3>>(sigma, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(mapped_coords, i) =
        gsl::at(lower_coords, i) +
        (gsl::at(upper_coords, i) - gsl::at(lower_coords, i)) * sigma;
  }
  return mapped_coords;
}

template <typename InnerMap>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FocallyLiftedMap<InnerMap>::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // lower_coords are the mapped coords on the surface.
  const std::array<ReturnType, 3> lower_coords = inner_map_(source_coords);
  const auto lambda = FocallyLiftedMapHelpers::scale_factor<ReturnType>(
      lower_coords, proj_center_, center_, radius_,
      InnerMap::projection_source_is_between_focus_and_target());

  // upper_coords are the mapped coords on the surface of the sphere.
  std::array<ReturnType, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // Do the easiest of the terms involving the inner map.
  const ReturnType sigma = inner_map_.sigma(source_coords);
  const auto d_inner = inner_map_.jacobian(source_coords);
  const ReturnType lambda_factor = 1.0 - sigma + lambda * sigma;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian_matrix.get(i, j) += lambda_factor * d_inner.get(i, j);
    }
  }

  // Do the deriv sigma term
  const auto d_sigma = inner_map_.deriv_sigma(source_coords);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian_matrix.get(i, j) +=
          gsl::at(d_sigma, j) *
          (gsl::at(upper_coords, i) - gsl::at(lower_coords, i));
    }
  }

  // Do lambda term, which is the most complicated one.
  const auto d_lambda_d_lower_coords =
      FocallyLiftedMapHelpers::d_scale_factor_d_src_point<ReturnType>(
          upper_coords, proj_center_, center_, lambda);
  for (size_t j = 0; j < 3; ++j) {
    auto temp = make_with_value<ReturnType>(sigma, 0.0);
    for (size_t k = 0; k < 3; ++k) {
      temp += gsl::at(d_lambda_d_lower_coords, k) * d_inner.get(k, j);
    }
    temp *= sigma;
    for (size_t i = 0; i < 3; ++i) {
      jacobian_matrix.get(i, j) +=
          temp * (gsl::at(lower_coords, i) - gsl::at(proj_center_, i));
    }
  }

  return jacobian_matrix;
}

template <typename InnerMap>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
FocallyLiftedMap<InnerMap>::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  // lower_coords are the mapped coords on the surface.
  const std::array<ReturnType, 3> lower_coords = inner_map_(source_coords);
  const auto lambda = FocallyLiftedMapHelpers::scale_factor<ReturnType>(
      lower_coords, proj_center_, center_, radius_,
      InnerMap::projection_source_is_between_focus_and_target());

  // upper_coords are the mapped coords on the surface of the sphere.
  std::array<ReturnType, 3> upper_coords = lower_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(upper_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) * lambda;
  }

  // Derivative of lambda
  const auto d_lambda_d_lower_coords =
      FocallyLiftedMapHelpers::d_scale_factor_d_src_point<ReturnType>(
          upper_coords, proj_center_, center_, lambda);

  // Lambda_tilde is the scale factor between mapped coords and lower coords.
  // We can compute it with a shortcut because there is a relationship
  // between lambda, lambda_tilde, and sigma.
  const ReturnType sigma = inner_map_.sigma(source_coords);
  const ReturnType lambda_tilde = 1.0 / (1.0 - sigma * (1.0 - lambda));

  // Derivative of lambda_tilde
  const auto d_lambda_tilde_d_mapped_coords =
      inner_map_.deriv_lambda_tilde(lower_coords, lambda_tilde, proj_center_);

  // Deriv of x_0 with respect to x
  auto dx0_dx =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dx0_dx.get(i, j) = gsl::at(d_lambda_tilde_d_mapped_coords, j) *
                         (gsl::at(lower_coords, i) - gsl::at(proj_center_, i)) /
                         lambda_tilde;
    }
    dx0_dx.get(i, i) += lambda_tilde;
  }

  // Deriv of sigma with respect to x,y,z
  auto d_sigma_d_mapped_coords =
      make_with_value<tnsr::i<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          sigma, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    auto tmp = make_with_value<ReturnType>(lambda, 0.0);
    for (size_t j = 0; j < 3; ++j) {
      tmp += gsl::at(d_lambda_d_lower_coords, j) * dx0_dx.get(j, i);
    }
    d_sigma_d_mapped_coords.get(i) =
        (sigma * tmp +
         gsl::at(d_lambda_tilde_d_mapped_coords, i) / square(lambda_tilde)) /
        (1.0 - lambda);
  }

  const auto dxbar_dx_inner = inner_map_.inv_jacobian(source_coords);
  const auto dxbar_dsigma = inner_map_.dxbar_dsigma(source_coords);

  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        inv_jacobian_matrix.get(i, j) +=
            dxbar_dx_inner.get(i, k) * dx0_dx.get(k, j);
      }
      inv_jacobian_matrix.get(i, j) +=
          gsl::at(dxbar_dsigma, i) * d_sigma_d_mapped_coords.get(j);
    }
  }

  return inv_jacobian_matrix;
}

template <typename InnerMap>
boost::optional<std::array<double, 3>> FocallyLiftedMap<InnerMap>::inverse(
    const std::array<double, 3>& target_coords) const noexcept {

  // Scale factor taking target_coords to lower_coords.
  const auto lambda_tilde =
      inner_map_.lambda_tilde(target_coords, proj_center_);

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_tilde) {
    return boost::none;
  }

  // Try to find lambda_bar going from target_coords to sphere.
  const auto lambda_bar = FocallyLiftedMapHelpers::try_scale_factor(
      target_coords, proj_center_, center_, radius_,
      not InnerMap::projection_source_is_between_focus_and_target(),
      InnerMap::projection_source_is_between_focus_and_target());

  // Cannot find scale factor, so we are out of range of the map.
  if (not lambda_bar) {
    return boost::none;
  }

  // compute sigma in a roundoff-friendly way.
  double sigma = 0.0;
  if (equal_within_roundoff(lambda_tilde.get(), 1.0, 1.e-5)) {
    // Get sigma correct for sigma near 0
    sigma =
        (lambda_tilde.get() - 1.0) / (lambda_tilde.get() - lambda_bar.get());
  } else {
    // Get sigma correct for sigma near 1
    sigma = (lambda_bar.get() - 1.0) / (lambda_tilde.get() - lambda_bar.get()) +
            1.0;
  }

  std::array<double, 3> lower_coords = target_coords;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(lower_coords, i) =
        gsl::at(proj_center_, i) +
        (gsl::at(target_coords, i) - gsl::at(proj_center_, i)) *
            lambda_tilde.get();
  }

  boost::optional<std::array<double, 3>> orig_coords =
      inner_map_.inverse(lower_coords, sigma);

  // Root polishing.
  // Here we do a single Newton iteration to get the
  // inverse to agree with the forward map to the level of machine
  // roundoff that is required by the unit tests.
  // Without the root polishing, the unit tests occasionally fail
  // the 'inverse(map(x))=x' test at a level slightly above roundoff.
  if (orig_coords) {
    const auto inv_jac = inv_jacobian(orig_coords.get());
    const auto mapped_coords = operator()(orig_coords.get());
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(orig_coords.get(), i) +=
            (gsl::at(target_coords, j) - gsl::at(mapped_coords, j)) *
            inv_jac.get(i, j);
      }
    }
  }

  return orig_coords;
}

template <typename InnerMap>
void FocallyLiftedMap<InnerMap>::pup(PUP::er& p) noexcept {
  p | center_;
  p | proj_center_;
  p | radius_;
  p | inner_map_;
}

template <typename InnerMap>
bool operator!=(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename InnerMap>
bool operator==(const FocallyLiftedMap<InnerMap>& lhs,
                const FocallyLiftedMap<InnerMap>& rhs) noexcept {
  return lhs.center_ == rhs.center_ and lhs.proj_center_ == rhs.proj_center_ and
         lhs.radius_ == rhs.radius_ and lhs.inner_map_ == rhs.inner_map_;
}

// Explicit instantiations
/// \cond
#define IMAP(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template class FocallyLiftedMap<IMAP(data)>;                                \
  template bool operator==(const FocallyLiftedMap<IMAP(data)>& lhs,           \
                           const FocallyLiftedMap<IMAP(data)>& rhs) noexcept; \
  template bool operator!=(const FocallyLiftedMap<IMAP(data)>& lhs,           \
                           const FocallyLiftedMap<IMAP(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (FocallyLiftedInnerMaps::Endcap))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  FocallyLiftedMap<IMAP(data)>::operator()(                                  \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  FocallyLiftedMap<IMAP(data)>::jacobian(                                    \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  FocallyLiftedMap<IMAP(data)>::inv_jacobian(                                \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (FocallyLiftedInnerMaps::Endcap),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE
#undef DTYPE
#undef IMAP
/// \endcond

}  // namespace domain::CoordinateMaps
