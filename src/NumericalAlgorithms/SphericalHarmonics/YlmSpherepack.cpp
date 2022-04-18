// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <tuple>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Spherepack.hpp"

//============================================================================
// Note that SPHEREPACK (which is wrapped by YlmSpherepack) takes
// n_theta and n_phi as input, and is ok with arbitrary values of
// n_theta and n_phi.  SPHEREPACK computes the maximum Ylm l and m
// using the formulas l_max=n_theta-1 and m_max=min(l_max,n_phi/2).
//
// However, some combinations of n_theta and n_phi have strange properties:
// - The maximum m that is AT LEAST PARTIALLY represented by (n_theta,n_phi)
//   is std::min(n_theta-1,n_phi/2). This is called m_max here. But an arbitrary
//   (n_theta,n_phi) does not necessarily fully represent all m's up to m_max,
//   because sin(m_max phi) might be zero at all collocation points, and
//   therefore sin(m_max phi) might not be representable on the grid.
// - The largest m that is fully represented by (n_theta,n_phi) is
//   m_max_represented = std::min(n_theta-1,(n_phi-1)/2).
// - Therefore, if n_phi is odd,  m_max = m_max_represented,
//              if n_phi is even, m_max = m_max_represented+1.
// - To remedy this situation, we choose YlmSpherepack to take as arguments
//   l_max and m_max, instead of n_theta and n_phi.
//   We then choose
//      n_theta = l_max+1
//      n_phi   = 2*m_max+1
//   This ensures that m_max = m_max_represented
//   (as opposed to m_max = m_max_represented+1)
//============================================================================

YlmSpherepack::YlmSpherepack(const size_t l_max, const size_t m_max)
    : l_max_{l_max},
      m_max_{m_max},
      n_theta_{l_max_ + 1},
      n_phi_{2 * m_max_ + 1},
      spectral_size_{2 * (l_max_ + 1) * (m_max_ + 1)},
      storage_(l_max_, m_max_) {
  if (l_max_ < 2) {
    ERROR("Must use l_max>=2, not l_max=" << l_max_);
  }
  if (m_max_ < 2) {
    ERROR("Must use m_max>=2, not m_max=" << m_max_);
  }
  if (m_max_ > l_max_) {
    ERROR("Must use m_max<=l_max, not l_max=" << l_max_
                                              << ", m_max=" << m_max_);
  }
  calculate_collocation_points();
  fill_scalar_work_arrays();
  fill_vector_work_arrays();
  calculate_interpolation_data();
}

size_t YlmSpherepack::phys_to_spec_buffer_size() const {
  return 2 * n_theta_ * n_phi_;
}

size_t YlmSpherepack::spec_to_phys_buffer_size() const {
  return 2 * n_theta_ * n_phi_;
}

void YlmSpherepack::phys_to_spec_impl(
    const gsl::not_null<gsl::span<double>*> buffer,
    const gsl::not_null<double*> spectral_coefs,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset,
    const size_t spectral_stride, const size_t spectral_offset,
    const bool loop_over_offset) const {
  size_t work_size = 2 * n_theta_ * n_phi_;
  if (loop_over_offset) {
    ASSERT(physical_stride == spectral_stride, "invalid call");
    work_size *= spectral_stride;
  }
  if (UNLIKELY(buffer->size()<work_size)) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << work_size << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }

  double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  double* const b =
      a + (m_max_ + 1) * (l_max_ + 1) * spectral_stride;  // NOLINT
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);
  auto& work_phys_to_spec = storage_.work_phys_to_spec;
  shags_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
         effective_physical_offset, effective_spectral_offset,
         static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
         collocation_values, static_cast<int>(n_theta_),
         static_cast<int>(n_phi_), a, b, static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), work_phys_to_spec.data(),
         static_cast<int>(work_phys_to_spec.size()), buffer->data(),
         static_cast<int>(buffer->size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shags error " << err << " in YlmSpherepack");
  }
}

void YlmSpherepack::spec_to_phys_impl(
    const gsl::not_null<gsl::span<double>*> buffer,
    const gsl::not_null<double*> collocation_values,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset,
    const bool loop_over_offset) const {
  size_t work_size = 2 * n_theta_ * n_phi_;
  if (loop_over_offset) {
    ASSERT(physical_stride == spectral_stride, "invalid call");
    work_size *= spectral_stride;
  }
  if (UNLIKELY(buffer->size()<work_size)) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << work_size << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }

  // 'a' and 'b' are Spherepack's coefficient arrays.
  const double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b =
      a + (m_max_ + 1) * (l_max_ + 1) * spectral_stride;  // NOLINT
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);

  auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
  shsgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
         effective_physical_offset, effective_spectral_offset,
         static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
         collocation_values, static_cast<int>(n_theta_),
         static_cast<int>(n_phi_), a, b, static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), work_scalar_spec_to_phys.data(),
         static_cast<int>(work_scalar_spec_to_phys.size()), buffer->data(),
         static_cast<int>(buffer->size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shsgs error " << err << " in YlmSpherepack");
  }
}

DataVector YlmSpherepack::phys_to_spec(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values, const size_t physical_stride,
    const size_t physical_offset) const {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  DataVector result(spectral_size());
  phys_to_spec_impl(buffer,
                    result.data(), collocation_values.data(), physical_stride,
                    physical_offset, 1, 0, false);
  return result;
}

DataVector YlmSpherepack::spec_to_phys(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& spectral_coefs, const size_t spectral_stride,
    const size_t spectral_offset) const {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  DataVector result(physical_size());
  spec_to_phys_impl(buffer, result.data(), spectral_coefs.data(),
                    spectral_stride, spectral_offset, 1, 0, false);
  return result;
}

size_t YlmSpherepack::phys_to_spec_all_offsets_buffer_size(
    size_t stride) const {
  return 2 * n_theta_ * n_phi_ * stride;
}

size_t YlmSpherepack::spec_to_phys_all_offsets_buffer_size(
    size_t stride) const {
  return 2 * n_theta_ * n_phi_ * stride;
}

DataVector YlmSpherepack::phys_to_spec_all_offsets(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values, const size_t stride) const {
  ASSERT(collocation_values.size() == physical_size() * stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * stride);
  DataVector result(spectral_size() * stride);
  phys_to_spec_impl(buffer, result.data(), collocation_values.data(), stride, 0,
                    stride, 0, true);
  return result;
}

DataVector YlmSpherepack::spec_to_phys_all_offsets(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& spectral_coefs, const size_t stride) const {
  ASSERT(spectral_coefs.size() == spectral_size() * stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * stride);
  DataVector result(physical_size() * stride);
  spec_to_phys_impl(buffer, result.data(), spectral_coefs.data(), stride, 0,
                    stride, 0, true);
  return result;
}

size_t YlmSpherepack::gradient_buffer_size() const {
  return spectral_size() +
         std::max(phys_to_spec_buffer_size(),
                  gradient_from_coefs_impl_buffer_size(false, 1));
}

size_t YlmSpherepack::gradient_from_coefs_buffer_size() const {
  return gradient_from_coefs_impl_buffer_size(false, 1);
}

void YlmSpherepack::gradient(
    const gsl::not_null<gsl::span<double>*> buffer,
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const {
  if (UNLIKELY(buffer->size() < gradient_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << gradient_buffer_size() << ". Here n_theta = "
                            << n_theta_ << " and n_phi = " << n_phi_);
  }
  gsl::span<double> phys_to_spec_buffer = gsl::make_span(
      buffer->data() + spectral_size(), phys_to_spec_buffer_size());
  phys_to_spec_impl(make_not_null(&phys_to_spec_buffer), buffer->data(),
                    collocation_values, physical_stride, physical_offset, 1, 0,
                    false);
  gsl::span<double> gradient_from_coefs_impl_buffer =
      gsl::make_span(buffer->data() + spectral_size(),
                     gradient_from_coefs_impl_buffer_size(false, 1));
  gradient_from_coefs_impl(make_not_null(&gradient_from_coefs_impl_buffer), df,
                           buffer->data(), 1, 0, physical_stride,
                           physical_offset, false);
}

size_t YlmSpherepack::gradient_all_offsets_buffer_size(
    const size_t stride) const {
  return stride * spectral_size() +
         std::max(phys_to_spec_all_offsets_buffer_size(stride),
                  gradient_from_coefs_impl_buffer_size(true, stride));
}

void YlmSpherepack::gradient_all_offsets(
    const gsl::not_null<gsl::span<double>*> buffer,
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> collocation_values,
    const size_t stride) const {
  const size_t spectral_stride = stride;
  if (UNLIKELY(buffer->size() < gradient_all_offsets_buffer_size(stride))) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << gradient_all_offsets_buffer_size(stride)
                            << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }
  gsl::span<double> phys_to_spec_buffer =
      gsl::make_span(buffer->data() + stride * spectral_size(),
                     phys_to_spec_all_offsets_buffer_size(stride));
  phys_to_spec_impl(make_not_null(&phys_to_spec_buffer), buffer->data(),
                    collocation_values, stride, 0, spectral_stride, 0, true);
  gsl::span<double> gradient_from_coefs_impl_buffer =
      gsl::make_span(buffer->data() + stride * spectral_size(),
                     gradient_from_coefs_impl_buffer_size(true, stride));
  gradient_from_coefs_impl(make_not_null(&gradient_from_coefs_impl_buffer), df,
                           buffer->data(), spectral_stride, 0, stride, 0, true);
}

size_t YlmSpherepack::gradient_from_coefs_impl_buffer_size(
    const bool loop_over_offset, const size_t stride) const {
  const size_t l1 = m_max_ + 1;
  size_t work_size = n_theta_ * (3 * n_phi_ + 2 * l1 + 1);
  if (loop_over_offset) {
    work_size *= stride;
  }
  return work_size;
}

void YlmSpherepack::gradient_from_coefs_impl(
    const gsl::not_null<gsl::span<double>*> buffer,
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset,
    bool loop_over_offset) const {
  ASSERT((not loop_over_offset) or spectral_stride == physical_stride,
         "physical and spectral strides must be equal "
         "for loop_over_offset=true");
  const size_t expected_buffer_size =
      gradient_from_coefs_impl_buffer_size(loop_over_offset, spectral_stride);
  if (UNLIKELY(buffer->size() < expected_buffer_size)) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << expected_buffer_size << ". Here n_theta = "
                            << n_theta_ << " and n_phi = " << n_phi_);
  }

  const size_t l1 = m_max_ + 1;
  const double* const f_k = spectral_coefs;
  const double* const a = f_k;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b = f_k + l1 * n_theta_ * spectral_stride;  // NOLINT

  size_t work_size = n_theta_ * (3 * n_phi_ + 2 * l1 + 1);
  if (loop_over_offset) {
    work_size *= spectral_stride;
  }
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);
  auto& work_vector_spec_to_phys = storage_.work_vector_spec_to_phys;
  gradgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
          effective_physical_offset, effective_spectral_offset,
          static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1, df[0],
          df[1], static_cast<int>(n_theta_), static_cast<int>(n_phi_), a, b,
          static_cast<int>(l1), static_cast<int>(n_theta_),
          work_vector_spec_to_phys.data(),
          static_cast<int>(work_vector_spec_to_phys.size()), buffer->data(),
          static_cast<int>(buffer->size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("gradgs error " << err << " in YlmSpherepack");
  }
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values, const size_t physical_stride,
    const size_t physical_offset) const {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  if (UNLIKELY(buffer->size() < gradient_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << gradient_buffer_size() << ". Here n_theta = "
                            << n_theta_ << " and n_phi = " << n_phi_);
  }
  gsl::span<double> phys_to_spec_buffer = gsl::make_span(
      buffer->data() + spectral_size(), phys_to_spec_buffer_size());
  phys_to_spec_impl(make_not_null(&phys_to_spec_buffer), buffer->data(),
                    collocation_values.data(), physical_stride, physical_offset,
                    1, 0, false);
  FirstDeriv result(physical_size());
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gsl::span<double> gradient_from_coefs_impl_buffer =
      gsl::make_span(buffer->data() + spectral_size(),
                     gradient_from_coefs_impl_buffer_size(false, 1));
  gradient_from_coefs_impl(make_not_null(&gradient_from_coefs_impl_buffer),
                           temp, buffer->data(), 1, 0, 1, 0, false);
  return result;
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_all_offsets(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values, const size_t stride) const {
  ASSERT(collocation_values.size() == physical_size() * stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * stride);
  FirstDeriv result(physical_size() * stride);
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_all_offsets(buffer, temp, collocation_values.data(), stride);
  return result;
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_from_coefs(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& spectral_coefs, const size_t spectral_stride,
    const size_t spectral_offset) const {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  FirstDeriv result(physical_size());
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_from_coefs_impl(buffer, temp, spectral_coefs.data(), spectral_stride,
                           spectral_offset, 1, 0, false);
  return result;
}

size_t YlmSpherepack::gradient_from_coefs_all_offsets_buffer_size(
    const size_t stride) const {
  return gradient_from_coefs_impl_buffer_size(true, stride);
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_from_coefs_all_offsets(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& spectral_coefs, const size_t stride) const {
  ASSERT(spectral_coefs.size() == spectral_size() * stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * stride);
  const size_t expected_buffer_size =
      gradient_from_coefs_all_offsets_buffer_size(stride);
  if (UNLIKELY(buffer->size() < expected_buffer_size)) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << expected_buffer_size << ". Here n_theta = "
                            << n_theta_ << " and n_phi = " << n_phi_);
  }
  FirstDeriv result(physical_size() * stride);
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_from_coefs_impl(buffer, temp, spectral_coefs.data(), stride, 0,
                           stride, 0, true);
  return result;
}

size_t YlmSpherepack::scalar_laplacian_buffer_size() const {
  return spectral_size() + std::max(phys_to_spec_buffer_size(),
                                    scalar_laplacian_from_coefs_buffer_size());
}

void YlmSpherepack::scalar_laplacian(
    const gsl::not_null<gsl::span<double>*> buffer,
    const gsl::not_null<double*> scalar_laplacian,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const {
  if (UNLIKELY(buffer->size() < scalar_laplacian_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << scalar_laplacian_buffer_size()
                            << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }
  gsl::span<double> phys_to_spec_buffer = gsl::make_span(
      buffer->data() + spectral_size(), phys_to_spec_buffer_size());
  phys_to_spec(make_not_null(&phys_to_spec_buffer), buffer->data(),
               collocation_values, physical_stride, physical_offset, 1, 0);

  gsl::span<double> scalar_laplacian_from_coefs_buffer =
      gsl::make_span(buffer->data() + spectral_size(),
                     scalar_laplacian_from_coefs_buffer_size());
  scalar_laplacian_from_coefs(
      make_not_null(&scalar_laplacian_from_coefs_buffer), scalar_laplacian,
      buffer->data(), 1, 0, physical_stride, physical_offset);
}

size_t YlmSpherepack::scalar_laplacian_from_coefs_buffer_size() const {
  const size_t l1 = m_max_ + 1;
  const size_t work_size = n_theta_ * (3 * n_phi_ + 2 * l1 + 1);
  return work_size;
}

void YlmSpherepack::scalar_laplacian_from_coefs(
    const gsl::not_null<gsl::span<double>*> buffer,
    const gsl::not_null<double*> scalar_laplacian,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset) const {
  if (UNLIKELY(buffer->size() < scalar_laplacian_from_coefs_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << scalar_laplacian_from_coefs_buffer_size()
                            << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }
  const size_t l1 = m_max_ + 1;
  const double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b = a + l1 * n_theta_ * spectral_stride;  // NOLINT
  int err = 0;
  auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
  slapgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
          static_cast<int>(physical_offset), static_cast<int>(spectral_offset),
          static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
          scalar_laplacian, static_cast<int>(n_theta_),
          static_cast<int>(n_phi_), a, b, static_cast<int>(l1),
          static_cast<int>(n_theta_), work_scalar_spec_to_phys.data(),
          static_cast<int>(work_scalar_spec_to_phys.size()), buffer->data(),
          static_cast<int>(buffer->size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("slapgs error " << err << " in YlmSpherepack");
  }
}

DataVector YlmSpherepack::scalar_laplacian(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values, const size_t physical_stride,
    const size_t physical_offset) const {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  DataVector result(physical_size());
  scalar_laplacian(buffer, result.data(), collocation_values.data(),
                   physical_stride, physical_offset);
  return result;
}

DataVector YlmSpherepack::scalar_laplacian_from_coefs(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& spectral_coefs, const size_t spectral_stride,
    const size_t spectral_offset) const {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  DataVector result(physical_size());
  scalar_laplacian_from_coefs(buffer, result.data(), spectral_coefs.data(),
                              spectral_stride, spectral_offset);
  return result;
}

std::array<DataVector, 2> YlmSpherepack::theta_phi_points() const {
  std::array<DataVector, 2> result = make_array<2>(DataVector(physical_size()));
  const auto& theta = theta_points();
  const auto& phi = phi_points();
  // Storage in SPHEREPACK: theta varies fastest (i.e. fortran ordering).
  for (size_t i_phi = 0, s = 0; i_phi < n_phi_; ++i_phi) {
    for (size_t i_theta = 0; i_theta < n_theta_; ++i_theta, ++s) {
      result[0][s] = theta[i_theta];
      result[1][s] = phi[i_phi];
    }
  }
  return result;
}

size_t YlmSpherepack::second_derivative_buffer_size() const {
  return 7 * physical_size() + gradient_buffer_size();
}

void YlmSpherepack::second_derivative(
    const gsl::not_null<gsl::span<double>*> buffer,
    const std::array<double*, 2>& df, const gsl::not_null<SecondDeriv*> ddf,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const {
  if (UNLIKELY(buffer->size() < second_derivative_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << second_derivative_buffer_size()
                            << ". Here n_theta = " << n_theta_
                            << " and n_phi = " << n_phi_);
  }
  // Initialize trig functions at collocation points
  auto& cos_theta = storage_.cos_theta;
  auto& sin_theta = storage_.sin_theta;
  auto& sin_phi = storage_.sin_phi;
  auto& cos_phi = storage_.cos_phi;
  auto& cot_theta = storage_.cot_theta;
  auto& cosec_theta = storage_.cosec_theta;

  // Get first derivatives
  gsl::span<double> gradient_buffer = gsl::make_span(
      buffer->data() + 7 * physical_size(), gradient_buffer_size());
  gradient(make_not_null(&gradient_buffer), df, collocation_values,
           physical_stride, physical_offset);

  // Now get Cartesian derivatives.

  // First derivative
  std::vector<double*> dfc(3, nullptr);
  for (size_t i = 0; i < 3; ++i) {
    dfc[i] = buffer->data() + i * physical_size();
  }
  for (size_t j = 0, s = 0; j < n_phi_; ++j) {
    for (size_t i = 0; i < n_theta_; ++i, ++s) {
      dfc[0][s] = cos_theta[i] * cos_phi[j] *
                      df[0][s * physical_stride + physical_offset] -
                  sin_phi[j] * df[1][s * physical_stride + physical_offset];
      dfc[1][s] = cos_theta[i] * sin_phi[j] *
                      df[0][s * physical_stride + physical_offset] +
                  cos_phi[j] * df[1][s * physical_stride + physical_offset];
      dfc[2][s] = -sin_theta[i] * df[0][s * physical_stride + physical_offset];
    }
  }

  // Take derivatives of Cartesian derivatives to get second derivatives.
  std::vector<std::array<double*, 2>> ddfc(3, {{nullptr, nullptr}});
  // next_memory are the next memory locations that will be available
  // in the buffer for storing ddfc.  They are chosen carefully so as
  // to not overwrite any memory too early.  For example, ddfc[1][0]
  // (the third entry in the list, so the third time the inner j loop
  // is accessed) corresponds to buffer location 0 because at that
  // point the previous contents of buffer 0 (which is dfc[0]) is no
  // longer needed, having been used in the gradient call.  We can
  // eliminate this tricky memory logic by allocating slightly more memory in
  // the buffer (9 * physical_size instead of 7 * physical_size).
  const std::vector<size_t> next_memory = {{3,4,0,5,1,6}};
  size_t n = 0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j, ++n) {
      gsl::at(ddfc[i], j) = buffer->data() + next_memory[n] * physical_size();
    }
    gradient(make_not_null(&gradient_buffer), ddfc[i], dfc[i], 1, 0);
  }

  // Combine into Pfaffian second derivatives
  for (size_t j = 0, s = 0; j < n_phi_; ++j) {
    for (size_t i = 0; i < n_theta_; ++i, ++s) {
      ddf->get(1, 0)[s * physical_stride + physical_offset] =
          -ddfc[2][1][s] * cosec_theta[i];
      ddf->get(0, 1)[s * physical_stride + physical_offset] =
          ddf->get(1, 0)[s * physical_stride + physical_offset] -
          cot_theta[i] * df[1][s * physical_stride + physical_offset];
      ddf->get(1, 1)[s * physical_stride + physical_offset] =
          cos_phi[j] * ddfc[1][1][s] - sin_phi[j] * ddfc[0][1][s] -
          cot_theta[i] * df[0][s * physical_stride + physical_offset];
      ddf->get(0, 0)[s * physical_stride + physical_offset] =
          cos_theta[i] *
              (cos_phi[j] * ddfc[0][0][s] + sin_phi[j] * ddfc[1][0][s]) -
          sin_theta[i] * ddfc[2][0][s];
    }
  }
}

std::pair<YlmSpherepack::FirstDeriv, YlmSpherepack::SecondDeriv>
YlmSpherepack::first_and_second_derivative(
    const gsl::not_null<gsl::span<double>*> buffer,
    const DataVector& collocation_values) const {
  ASSERT(collocation_values.size() == physical_size(),
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size());
  std::pair<FirstDeriv, SecondDeriv> result(
      std::piecewise_construct, std::forward_as_tuple(physical_size()),
      std::forward_as_tuple(physical_size()));
  std::array<double*, 2> temp = {
      {std::get<0>(result).get(0).data(), std::get<0>(result).get(1).data()}};
  second_derivative(buffer, temp, &std::get<1>(result),
                    collocation_values.data());
  return result;
}

template <typename T>
YlmSpherepack::InterpolationInfo<T>::InterpolationInfo(
    const size_t l_max, const size_t m_max, const gsl::span<double> pmm,
    const std::array<T, 2>& target_points)
    : cos_theta(cos(target_points[0])),
      cos_m_phi(DynamicBuffer<T>(m_max + 1, get_size(target_points[0]))),
      sin_m_phi(DynamicBuffer<T>(m_max + 1, get_size(target_points[0]))),
      pbar_factor(DynamicBuffer<T>(m_max + 1, get_size(target_points[0]))),
      l_max_(l_max),
      m_max_(m_max),
      num_points_(get_size(target_points[0])) {
  const auto& theta = target_points[0];
  const auto& phi = target_points[1];

  ASSERT(get_size(theta) == get_size(phi),
         "Size mismatch for thetas and phis: " << get_size(theta) << " and "
                                               << get_size(phi));
  if (m_max < 2) {
    ERROR("Must use m_max>=2, not m_max=" << m_max);
  }

  // `DataVectors` for working. `pbar_factor` is guaranteed to be at least size
  // 3 as demanded by the `YlmSpherepack` constructor
  auto& alpha = pbar_factor.at(0);
  auto& beta = pbar_factor.at(1);
  auto& deltasinmphi = pbar_factor.at(2);

  // If T is DataVector, `sinmphi` and `cosmphi` will be created with
  // size `num_points`, if T is double they will be initialized to value
  // `num_points` but this value is not used.
  T sinmphi(num_points_);
  T cosmphi(num_points_);

  // Evaluate cos(m*phi) and sin(m*phi) by numerical recipes eq. 5.5.6
  {
    alpha = 2.0 * square(sin(0.5 * phi));
    beta = sin(phi);
    sinmphi = 0.;
    cosmphi = 1.;
    cos_m_phi[0] = 1.;
    sin_m_phi[0] = 0.;
    for (size_t m = 1; m < m_max + 1; ++m) {
      deltasinmphi = alpha * sinmphi - beta * cosmphi;
      cosmphi -= alpha * cosmphi + beta * sinmphi;
      sinmphi -= deltasinmphi;
      cos_m_phi[m] = cosmphi;
      sin_m_phi[m] = sinmphi;
    }
  }

  T& sin_theta = sinmphi;
  T& sinmtheta = cosmphi;
  sin_theta = sin(theta);
  sinmtheta = 1.;

  // Fill pbar_factor[m] = Pbar(m,m)*sin(theta)^m  (m<l1)
  for (size_t m = 0; m < m_max + 1; ++m) {
    pbar_factor[m] = pmm[m] * sinmtheta;
    sinmtheta *= sin_theta;
  }
}

template <typename T>
YlmSpherepack::InterpolationInfo<T> YlmSpherepack::set_up_interpolation_info(
    const std::array<T, 2>& target_points) const {
  return InterpolationInfo(l_max_, m_max_, storage_.work_interp_pmm,
                           target_points);
}

size_t YlmSpherepack::interpolate_buffer_size() const {
  return spectral_size() + phys_to_spec_buffer_size();
}

template <typename T>
void YlmSpherepack::interpolate(
    const gsl::not_null<gsl::span<double>*> buffer,
    const gsl::not_null<T*> result,
    const gsl::not_null<const double*> collocation_values,
    const InterpolationInfo<T>& interpolation_info,
    const size_t physical_stride, const size_t physical_offset) const {
  ASSERT(get_size(*result) == interpolation_info.size(),
         "Size mismatch: " << get_size(*result) << ","
                           << interpolation_info.size());
  if (UNLIKELY(buffer->size() < interpolate_buffer_size())) {
    ERROR("Buffer size is " << buffer->size() << " but it must be at least "
                            << interpolate_buffer_size() << ". Here n_theta = "
                            << n_theta_ << " and n_phi = " << n_phi_);
  }
  gsl::span<double> phys_to_spec_buffer = gsl::make_span(
      buffer->data() + spectral_size(), phys_to_spec_buffer_size());
  phys_to_spec(make_not_null(&phys_to_spec_buffer), buffer->data(),
               collocation_values, physical_stride, physical_offset, 1, 0);
  interpolate_from_coefs(result, *buffer, interpolation_info);
}

template <typename T, typename R>
void YlmSpherepack::interpolate_from_coefs(
    const gsl::not_null<T*> result, const R& spectral_coefs,
    const InterpolationInfo<T>& interpolation_info,
    const size_t spectral_stride, const size_t spectral_offset) const {
  if (UNLIKELY(m_max_ != interpolation_info.m_max())) {
    ERROR("Different m_max for InterpolationInfo ("
          << interpolation_info.m_max() << ") and YlmSpherepack instance ("
          << m_max_ << ")");
  };
  if (UNLIKELY(l_max_ != interpolation_info.l_max())) {
    ERROR("Different l_max for InterpolationInfo ("
          << interpolation_info.l_max() << ") and YlmSpherepack instance ("
          << l_max_ << ")");
  };
  const auto& alpha = storage_.work_interp_alpha;
  const auto& beta = storage_.work_interp_beta;
  const auto& index = storage_.work_interp_index;
  // alpha holds alpha(n,m,x)/x, beta holds beta(n+1,m).
  // index holds the index into the coefficient array.
  // All are indexed together.

  const size_t num_points = get_size(*result);
  ASSERT(num_points == interpolation_info.size(),
         "Size mismatch: " << num_points << "," << interpolation_info.size());

  // initialize work `DataVectors` in `TempBuffer` to reduce allocations
  TempBuffer<tmpl::list<::Tags::TempScalar<0, T>, ::Tags::TempScalar<1, T>,
                        ::Tags::TempScalar<2, T>, ::Tags::TempScalar<3, T>,
                        ::Tags::TempScalar<4, T>, ::Tags::TempScalar<5, T>>>
      buffer(num_points);

  auto& ycn = get(get<::Tags::TempScalar<0, T>>(buffer));
  auto& ycnp1 = get(get<::Tags::TempScalar<1, T>>(buffer));
  auto& ycnp2 = get(get<::Tags::TempScalar<2, T>>(buffer));
  auto& ysn = get(get<::Tags::TempScalar<3, T>>(buffer));
  auto& ysnp1 = get(get<::Tags::TempScalar<4, T>>(buffer));
  auto& ysnp2 = get(get<::Tags::TempScalar<5, T>>(buffer));

  const size_t l1 = m_max_ + 1;

  // Offsets of 'a' and 'b' in spectral_coefs.
  const size_t a_offset = spectral_offset;
  const size_t b_offset = spectral_offset + (l1 * n_theta_) * spectral_stride;

  const auto& cos_theta = interpolation_info.cos_theta;

  // Clenshaw recurrence for m=0.  Separate because there is no phi
  // dependence, and there is a factor of 1/2.
  size_t idx = 0;
  {
    ycn = 0.;
    ycnp1 = 0.;
    for (size_t n = n_theta_ - 1; n > 0;
         --n, ++idx) {  // Loops from n_theta_-1 to 1.
      ycnp2 = ycnp1;
      ycnp1 = ycn;
      ycn = cos_theta * alpha[idx] * ycnp1 + beta[idx] * ycnp2 +
            spectral_coefs[a_offset + spectral_stride * index[idx]];
    }
    *result = 0.5 * interpolation_info.pbar_factor[0] *
              (beta[idx] * ycnp1 + cos_theta * alpha[idx] * ycn +
               spectral_coefs[a_offset + spectral_stride * index[idx]]);
    ++idx;
  }
  // Now do recurrence for other m.
  for (size_t m = 1; m < l1; ++m) {
    ycn = 0.;
    ycnp1 = 0.;
    ysn = 0.;
    ysnp1 = 0.;
    for (size_t n = n_theta_ - 1; n > m; --n, ++idx) {
      ycnp2 = ycnp1;
      ysnp2 = ysnp1;
      ycnp1 = ycn;
      ysnp1 = ysn;
      ycn = cos_theta * alpha[idx] * ycnp1 + beta[idx] * ycnp2 +
            spectral_coefs[a_offset + spectral_stride * index[idx]];
      ysn = cos_theta * alpha[idx] * ysnp1 + beta[idx] * ysnp2 +
            spectral_coefs[b_offset + spectral_stride * index[idx]];
    }

    auto& fc = ycnp2;
    auto& fs = ysnp2;
    fc = interpolation_info.pbar_factor[m] *
         (beta[idx] * ycnp1 + cos_theta * alpha[idx] * ycn +
          spectral_coefs[a_offset + spectral_stride * index[idx]]);
    fs = interpolation_info.pbar_factor[m] *
         (beta[idx] * ysnp1 + cos_theta * alpha[idx] * ysn +
          spectral_coefs[b_offset + spectral_stride * index[idx]]);
    *result += fc * interpolation_info.cos_m_phi[m] -
               fs * interpolation_info.sin_m_phi[m];
    ++idx;
  }
  ASSERT(idx == index.size(),
         "Wrong size " << idx << ", expected " << index.size());
}

template <typename T>
T YlmSpherepack::interpolate(const gsl::not_null<gsl::span<double>*> buffer,
                             const DataVector& collocation_values,
                             const std::array<T, 2>& target_points) const {
  ASSERT(collocation_values.size() == physical_size(),
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size());
  auto result = make_with_value<T>(target_points[0], 0.);
  interpolate<T>(buffer, make_not_null(&result), collocation_values.data(),
                 set_up_interpolation_info<T>(target_points));
  return result;
}

template <typename T>
T YlmSpherepack::interpolate_from_coefs(
    const DataVector& spectral_coefs,
    const std::array<T, 2>& target_points) const {
  ASSERT(spectral_coefs.size() == spectral_size(),
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size());
  auto result = make_with_value<T>(target_points[0], 0.);
  interpolate_from_coefs<T>(&result, spectral_coefs,
                            set_up_interpolation_info<T>(target_points));
  return result;
}

void YlmSpherepack::calculate_collocation_points() {
  // Theta
  auto& theta = storage_.theta;
  DataVector temp(2 * n_theta_ + 1);
  auto work = gsl::make_span(temp.data(), n_theta_);
  auto unused_weights = gsl::make_span(temp.data() + n_theta_, n_theta_ + 1);

  int err = 0;
  gaqd_(static_cast<int>(n_theta_), theta.data(), unused_weights.data(),
        work.data(), static_cast<int>(unused_weights.size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("gaqd error " << err << " in YlmSpherepack");
  }

  // Phi
  auto& phi = storage_.phi;
  const double two_pi_over_n_phi = 2.0 * M_PI / n_phi_;
  for (size_t i = 0; i < n_phi_; ++i) {
    phi[i] = two_pi_over_n_phi * i;
  }

  // Other trig functions at collocation points
  auto& cos_theta = storage_.cos_theta;
  auto& sin_theta = storage_.sin_theta;
  auto& cot_theta = storage_.cot_theta;
  auto& cosec_theta = storage_.cosec_theta;
  for (size_t i = 0; i < n_theta_; ++i) {
    cos_theta[i] = cos(theta[i]);
    sin_theta[i] = sin(theta[i]);
    cosec_theta[i] = 1.0 / sin_theta[i];
    cot_theta[i] = cos_theta[i] * cosec_theta[i];
  }

  auto& sin_phi = storage_.sin_phi;
  auto& cos_phi = storage_.cos_phi;
  for (size_t i = 0; i < n_phi_; ++i) {
    cos_phi[i] = cos(phi[i]);
    sin_phi[i] = sin(phi[i]);
  }
}

void YlmSpherepack::calculate_interpolation_data() {
  // SPHEREPACK expands f(theta,phi) as
  //
  // f(theta,phi) =
  // 1/2 Sum_{l=0}^{l_max} Pbar(l,0) a(0,l)
  //   + Sum_{m=1}^{m_max} Sum_{l=m}^{l_max} Pbar(l,m)(  a(m,l) cos(m phi)
  //                                                   - b(m,l) sin(m phi))
  //
  // where Pbar(l,m) are unit-orthonormal Associated Legendre
  // polynomials (which are functions of x = cos(theta)), and a(m,l)
  // and b(m,l) are the SPHEREPACK spectral coefficients.
  //
  // Note that Pbar(l,m) = sqrt((2l+1)(l-m)!/(2(l+m)!)) P_l^m.
  // and that Integral_{-1}^{+1} Pbar(k,m)(x) Pbar(l,m)(x) = delta_kl
  //
  // We will interpolate via Clenshaw's recurrence formula in l, for fixed m.
  //
  // The recursion relation between Associated Legendre polynomials is
  // Pbar(l+1)(m) = alpha(l,x) Pbar(l)(m) + beta(l,x) Pbar(l-1)(m)
  // where alpha(l,x) = x sqrt((2l+3)(2l+1)/((l+1-m)(l+1+m)))
  // where beta(l,x)  = - sqrt((2l+3)(l-m)(l+m)/((l+1-m)(l+1+m)(2l-1)))
  //
  // The Clenshaw recurrence formula for
  // f(x) = Sum_{l=m}^{l_max} c_(l)(m) Pbar(l)(m) is
  // y_{l_max+1} = y_{l_max+2} = 0
  // y_k  = alpha(k,x) y_{k+1} + beta(k+1,x) y_{k+2} + c_(k)(m)  (m<k<=l_max)
  // f(x) = beta(m+1,x) Pbar(m,m) y_{m+2} + Pbar(m+1,m) y_{m+1}
  //      + Pbar(m,m) c_(m)(m).
  //
  // So we will compute and store alpha(l,x)/x in 'alpha' and
  // beta(l+1,x) [NOT beta(l,x)] in 'beta'.  We will also compute and
  // store the x-independent piece of Pbar(m)(m) in 'pmm' and we
  // will compute and store the x-independent piece of Pbar(m+1)(m)/Pbar(m)(m)
  // in a component of 'alpha'. See below for storage.
  // Note Pbar(m)(m)   is (2m-1)!! (1-x^2)^(n/2) sqrt((2m+1)/(2 (2m)!))
  // and  Pbar(m+1)(m) is (2m+1)!! x(1-x^2)^(n/2)sqrt((2m+3)/(2(2m+1)!))
  //  Ratio Pbar(m+1)(m)/Pbar(m)(m)   = x sqrt(2m+3)
  //  Ratio Pbar(m+1)(m+1)/Pbar(m)(m) = sqrt(1-x^2) sqrt((2m+3)/(2m+2))
  auto& alpha = storage_.work_interp_alpha;
  auto& beta = storage_.work_interp_beta;
  auto& pmm = storage_.work_interp_pmm;
  auto& index = storage_.work_interp_index;

  const size_t l1 = m_max_ + 1;

  // Fill alpha,beta,index arrays in the same order as the Clenshaw
  // recurrence, so that we can index them easier during the recurrence.
  // First do m=0.
  size_t idx = 0;
  for (size_t n = n_theta_ - 1; n > 0; --n, ++idx) {
    const auto n_dbl = static_cast<double>(n);
    const double tnp1 = 2.0 * n_dbl + 1;
    const double np1sq = n_dbl * n_dbl + 2.0 * n_dbl + 1.0;
    alpha[idx] = sqrt(tnp1 * (tnp1 + 2.0) / np1sq);
    beta[idx] = -sqrt((tnp1 + 4.0) / tnp1 * np1sq / (np1sq + 2 * n_dbl + 3));
    index[idx] = n_dbl * l1;
  }
  // The next value of beta stores beta(n=1,m=0).
  // The next value of alpha stores Pbar(n=1,m=0)/(x*Pbar(n=0,m=0)).
  // These two values are needed for the final Clenshaw recurrence formula.
  beta[idx] = -0.5 * sqrt(5.0);
  alpha[idx] = sqrt(3.0);
  index[idx] = 0;  // Index of coef in the final recurrence formula
  ++idx;

  // Now do other m.
  for (size_t m = 1; m < l1; ++m) {
    for (size_t n = n_theta_ - 1; n > m; --n, ++idx) {
      const double tnp1 = 2.0 * n + 1;
      const double np1sqmmsq = (n + 1.0 + m) * (n + 1.0 - m);
      alpha[idx] = sqrt(tnp1 * (tnp1 + 2.0) / np1sqmmsq);
      beta[idx] =
          -sqrt((tnp1 + 4.0) / tnp1 * np1sqmmsq / (np1sqmmsq + 2. * n + 3.));
      index[idx] = m + n * l1;
    }
    // The next value of beta stores beta(n=m+1,m).
    // The next value of alpha stores Pbar(n=m+1,m)/(x*Pbar(n=m,m)).
    // These two values are needed for the final Clenshaw recurrence formula.
    beta[idx] = -0.5 * sqrt((2.0 * m + 5) / (m + 1.0));
    alpha[idx] = sqrt(2.0 * m + 3);
    index[idx] = m + m * l1;  // Index of coef in the final recurrence formula.
    ++idx;
  }
  ASSERT(idx == index.size(),
         "Wrong size " << idx << ", expected " << index.size());

  // Now do pmm, which stores Pbar(m,m).
  pmm[0] = M_SQRT1_2;  // 1/sqrt(2) = Pbar(0)(0)
  for (size_t m = 1; m < l1; ++m) {
    pmm[m] = pmm[m - 1] * sqrt((2.0 * m + 1.0) / (2.0 * m));
  }
}

void YlmSpherepack::fill_vector_work_arrays() {
  DataVector work((3 * n_theta_ * (n_theta_ + 3) + 2) / 2);

  auto& work_vector_spec_to_phys = storage_.work_vector_spec_to_phys;
  int err = 0;
  vhsgsi_(static_cast<int>(n_theta_), static_cast<int>(n_phi_),
          work_vector_spec_to_phys.data(),
          static_cast<int>(work_vector_spec_to_phys.size()), work.data(),
          static_cast<int>(work.size()), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("vhsgsi error " << err << " in YlmSpherepack");
  }
}

void YlmSpherepack::fill_scalar_work_arrays() {
  // Quadrature weights
  {
    DataVector temp(3 * n_theta_ + 1);
    auto weights = gsl::make_span(temp.data(), n_theta_);
    auto work0 = gsl::make_span(temp.data() + n_theta_, n_theta_);
    auto work1 = gsl::make_span(temp.data() + 2 * n_theta_, n_theta_ + 1);
    int err = 0;
    gaqd_(static_cast<int>(n_theta_), work0.data(), weights.data(),
          work1.data(), static_cast<int>(work1.size()), &err);
    if (UNLIKELY(err != 0)) {
      ERROR("gaqd error " << err << " in YlmSpherepack");
    }
    auto& quadrature_weights = storage_.quadrature_weights;
    for (size_t i = 0; i < n_theta_; ++i) {
      for (size_t j = 0; j < n_phi_; ++j) {
        quadrature_weights[i + j * n_theta_] = (2 * M_PI / n_phi_) * weights[i];
      }
    }
  }

  // Scalar arrays
  {
    const size_t work0_size = 4 * n_theta_ * (n_theta_ + 2) + 2;
    const size_t work1_size = n_theta_ * (n_theta_ + 4);
    DataVector temp(work0_size + work1_size);
    auto work0 = gsl::make_span(temp.data(), work0_size);
    auto work1 = gsl::make_span(temp.data() + work0_size, work1_size);
    auto& work_phys_to_spec = storage_.work_phys_to_spec;
    int err = 0;
    shagsi_(static_cast<int>(n_theta_), static_cast<int>(n_phi_),
            work_phys_to_spec.data(),
            static_cast<int>(work_phys_to_spec.size()), work0.data(),
            static_cast<int>(work0.size()), work1.data(),
            static_cast<int>(work1.size()), &err);
    if (UNLIKELY(err != 0)) {
      ERROR("shagsi error " << err << " in YlmSpherepack");
    }
    auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
    shsgsi_(static_cast<int>(n_theta_), static_cast<int>(n_phi_),
            work_scalar_spec_to_phys.data(),
            static_cast<int>(work_scalar_spec_to_phys.size()), work0.data(),
            static_cast<int>(work0.size()), work1.data(),
            static_cast<int>(work1.size()), &err);
    if (UNLIKELY(err != 0)) {
      ERROR("shsgsi error " << err << " in YlmSpherepack");
    }
  }
}

DataVector YlmSpherepack::prolong_or_restrict(
    const DataVector& spectral_coefs, const YlmSpherepack& target) const {
  ASSERT(spectral_coefs.size() == spectral_size(),
         "Expecting " << spectral_size() << ", got " << spectral_coefs.size());
  DataVector result(target.spectral_size(), 0.0);
  SpherepackIterator src_it(l_max_, m_max_);
  SpherepackIterator dest_it(target.l_max_, target.m_max_);
  for (; dest_it; ++dest_it) {
    if (dest_it.l() <= src_it.l_max() and dest_it.m() <= src_it.m_max()) {
      src_it.set(dest_it.l(), dest_it.m(), dest_it.coefficient_array());
      result[dest_it()] = spectral_coefs[src_it()];
    }
  }
  return result;
}

bool operator==(const YlmSpherepack& lhs, const YlmSpherepack& rhs) {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max();
}

bool operator!=(const YlmSpherepack& lhs, const YlmSpherepack& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template YlmSpherepack::InterpolationInfo<DTYPE(data)>::InterpolationInfo( \
      size_t, size_t, const gsl::span<double>,                               \
      const std::array<DTYPE(data), 2>&);                                    \
  template YlmSpherepack::InterpolationInfo<DTYPE(data)>                     \
  YlmSpherepack::set_up_interpolation_info(                                  \
      const std::array<DTYPE(data), 2>& target_points) const;                \
  template void YlmSpherepack::interpolate(                                  \
      const gsl::not_null<gsl::span<double>*> buffer,                        \
      gsl::not_null<DTYPE(data)*> result, gsl::not_null<const double*>,      \
      const YlmSpherepack::InterpolationInfo<DTYPE(data)>&, size_t, size_t)  \
      const;                                                                 \
  template void YlmSpherepack::interpolate_from_coefs(                       \
      gsl::not_null<DTYPE(data)*> result, const DataVector&,                 \
      const YlmSpherepack::InterpolationInfo<DTYPE(data)>&, size_t, size_t)  \
      const;                                                                 \
  template DTYPE(data) YlmSpherepack::interpolate(                           \
      const gsl::not_null<gsl::span<double>*> buffer, const DataVector&,     \
      const std::array<DTYPE(data), 2>&) const;                              \
  template DTYPE(data) YlmSpherepack::interpolate_from_coefs(                \
      const DataVector&, const std::array<DTYPE(data), 2>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
