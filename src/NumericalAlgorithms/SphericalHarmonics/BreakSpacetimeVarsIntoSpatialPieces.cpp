// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/BreakSpacetimeVarsIntoSpatialPieces.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

// Here we don't define the primary template function at all.
// We define only explicit specializations.
// If anyone wants to take the time to write a generic function that
// works for generic Variables, go for it!  For now, we specialize only
// for the generalized harmonic variables.

namespace {
using gh_spacetime_vars_list =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
               gh::Tags::Pi<DataVector, 3, Frame::Inertial>,
               gh::Tags::Phi<DataVector, 3, Frame::Inertial>>;

using gh_spatial_vars_list =
    tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempi<0, 3>, ::Tags::Tempii<0, 3>,
               ::Tags::TempScalar<1>, ::Tags::Tempi<1, 3>, ::Tags::Tempii<1, 3>,
               ::Tags::Tempi<2, 3>, ::Tags::Tempij<2, 3>,
               ::Tags::Tempijj<2, 3>>;

using metric_tag = gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>;
using pi_tag = gh::Tags::Pi<DataVector, 3, Frame::Inertial>;
using phi_tag = gh::Tags::Phi<DataVector, 3, Frame::Inertial>;
}  // namespace

template <>
void break_spacetime_vars_into_spatial_pieces<gh_spacetime_vars_list,
                                              gh_spatial_vars_list>(
    const gsl::not_null<Variables<gh_spatial_vars_list>*> spatial_vars,
    const Variables<gh_spacetime_vars_list>& spacetime_vars) {
  const auto& metric = get<metric_tag>(spacetime_vars);
  const auto& pi = get<pi_tag>(spacetime_vars);
  const auto& phi = get<phi_tag>(spacetime_vars);
  auto& g_00 = get<::Tags::TempScalar<0>>(*spatial_vars);
  auto& g_0i = get<::Tags::Tempi<0, 3>>(*spatial_vars);
  auto& g_ij = get<::Tags::Tempii<0, 3>>(*spatial_vars);
  auto& pi_00 = get<::Tags::TempScalar<1>>(*spatial_vars);
  auto& pi_0i = get<::Tags::Tempi<1, 3>>(*spatial_vars);
  auto& pi_ij = get<::Tags::Tempii<1, 3>>(*spatial_vars);
  auto& phi_k00 = get<::Tags::Tempi<2, 3>>(*spatial_vars);
  auto& phi_ki0 = get<::Tags::Tempij<2, 3>>(*spatial_vars);
  auto& phi_kij = get<::Tags::Tempijj<2, 3>>(*spatial_vars);
  get<>(g_00) = get<0, 0>(metric);
  get<>(pi_00) = get<0, 0>(pi);
  for (size_t i = 0; i < 3; ++i) {
    g_0i.get(i) = metric.get(i + 1, 0);
    pi_0i.get(i) = pi.get(i + 1, 0);
    for (size_t j = i; j < 3; ++j) {
      g_ij.get(i, j) = metric.get(i + 1, j + 1);
      pi_ij.get(i, j) = pi.get(i + 1, j + 1);
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    phi_k00.get(k) = phi.get(k, 0, 0);
    for (size_t i = 0; i < 3; ++i) {
      phi_ki0.get(k, i) = phi.get(k, i + 1, 0);
      for (size_t j = i; j < 3; ++j) {
        phi_kij.get(k, i, j) = phi.get(k, i + 1, j + 1);
      }
    }
  }
}

template <>
void assemble_spacetime_vars_from_spatial_pieces<gh_spacetime_vars_list,
                                                 gh_spatial_vars_list>(
    const gsl::not_null<Variables<gh_spacetime_vars_list>*> spacetime_vars,
    const Variables<gh_spatial_vars_list>& spatial_vars) {
  auto& metric = get<metric_tag>(*spacetime_vars);
  auto& pi = get<pi_tag>(*spacetime_vars);
  auto& phi = get<phi_tag>(*spacetime_vars);
  const auto& g_00 = get<::Tags::TempScalar<0>>(spatial_vars);
  const auto& g_0i = get<::Tags::Tempi<0, 3>>(spatial_vars);
  const auto& g_ij = get<::Tags::Tempii<0, 3>>(spatial_vars);
  const auto& pi_00 = get<::Tags::TempScalar<1>>(spatial_vars);
  const auto& pi_0i = get<::Tags::Tempi<1, 3>>(spatial_vars);
  const auto& pi_ij = get<::Tags::Tempii<1, 3>>(spatial_vars);
  const auto& phi_k00 = get<::Tags::Tempi<2, 3>>(spatial_vars);
  const auto& phi_ki0 = get<::Tags::Tempij<2, 3>>(spatial_vars);
  const auto& phi_kij = get<::Tags::Tempijj<2, 3>>(spatial_vars);
  get<0, 0>(metric) = get<>(g_00);
  get<0, 0>(pi) = get<>(pi_00);
  for (size_t i = 0; i < 3; ++i) {
    metric.get(i + 1, 0) = g_0i.get(i);
    pi.get(i + 1, 0) = pi_0i.get(i);
    for (size_t j = i; j < 3; ++j) {
      metric.get(i + 1, j + 1) = g_ij.get(i, j);
      pi.get(i + 1, j + 1) = pi_ij.get(i, j);
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    phi.get(k, 0, 0) = phi_k00.get(k);
    for (size_t i = 0; i < 3; ++i) {
      phi.get(k, i + 1, 0) = phi_ki0.get(k, i);
      for (size_t j = i; j < 3; ++j) {
        phi.get(k, i + 1, j + 1) = phi_kij.get(k, i, j);
      }
    }
  }
}
