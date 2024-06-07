// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/ComovingCharSpeedDerivative.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {
void comoving_char_speed_derivative(
    const gsl::not_null<Scalar<DataVector>*> result, const double lambda_00,
    const double dt_lambda_00, const double horizon_00,
    const double dt_horizon_00, const double grid_frame_excision_sphere_radius,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_rhat,
    const tnsr::i<DataVector, 3, Frame::Distorted>& excision_normal_one_form,
    const Scalar<DataVector>& excision_normal_one_form_norm,
    const tnsr::I<DataVector, 3, Frame::Distorted>&
        distorted_components_of_grid_shift,
    const tnsr::II<DataVector, 3, Frame::Distorted>&
        inverse_spatial_metric_on_excision_boundary,
    const tnsr::Ijj<DataVector, 3, Frame::Distorted>&
        spatial_christoffel_second_kind,
    const tnsr::i<DataVector, 3, Frame::Distorted>& deriv_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Distorted>& deriv_of_distorted_shift,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Distorted>&
        inverse_jacobian_grid_to_distorted) {
  const double Y00 = 0.25 * M_2_SQRTPI;

  // Define temporary storage.
  using excision_normal_vector_tag =
      ::Tags::TempI<1, 3, Frame::Distorted, DataVector>;
  using deriv_normal_one_form_tag =
      ::Tags::Tempi<2, 3, Frame::Distorted, DataVector>;
  TempBuffer<tmpl::list<excision_normal_vector_tag, deriv_normal_one_form_tag>>
      buffer(get<0>(excision_rhat).size());
  auto& excision_normal_vector = get<excision_normal_vector_tag>(buffer);
  auto& deriv_normal_one_form = get<deriv_normal_one_form_tag>(buffer);

  // excision_rhat is a tnsr:i when it is returned from a Strahlkorper.
  // But excision_rhat is a coordinate quantity, not a physical tensor, so
  // it can also be used as a tnsr::I.  Here we create a tnsr::I called
  // excision_rhat_vector that points into excision_rhat.
  const tnsr::I<DataVector, 3, Frame::Distorted> excision_rhat_vector;
  for (size_t i = 0; i < 3; ++i) {
    excision_rhat_vector.get(i).set_data_ref(
        make_not_null(&excision_rhat.get(i)));
  }

  tenex::evaluate<ti::I>(
      make_not_null(&excision_normal_vector),
      excision_normal_one_form(ti::j) *
          inverse_spatial_metric_on_excision_boundary(ti::J, ti::I));

  // Fill result temporarily with xi_k n_j gamma^jk/rEB
  get(*result) = get<0>(excision_normal_vector) * get<0>(excision_rhat) /
                 grid_frame_excision_sphere_radius;
  for (size_t k = 1; k < 3; ++k) {
    get(*result) += excision_normal_vector.get(k) * excision_rhat.get(k) /
                    grid_frame_excision_sphere_radius;
  }
  // Add n_p n_j gamma^{pk} xi^m Gamma^j_{km} to result
  for (size_t k = 0; k < 3; ++k) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t m = 0; m < 3; ++m) {
        get(*result) += excision_normal_vector.get(k) *
                        excision_normal_one_form.get(j) * excision_rhat.get(m) *
                        spatial_christoffel_second_kind.get(j, k, m);
      }
    }
  }
  // scale so result contains most of the terms in
  // d/dlambda00 (n_hati)
  get(*result) /= cube(get(excision_normal_one_form_norm));

  // Set deriv_normal_one_form to d/dlambda00 (n_hati).
  // Possible memory optimization: excision_normal_vector isn't used anymore,
  // so that storage could be used for deriv_normal_one_form.
  for (size_t i = 0; i < 3; ++i) {
    deriv_normal_one_form.get(i) =
        Y00 * (excision_rhat.get(i) / grid_frame_excision_sphere_radius -
               get(*result) * excision_normal_one_form.get(i));
  }

  // Overwrite result term by term. First do the n_i xi^i term
  // (without the Y00 or norm factor)
  get(*result) = get<0>(excision_normal_one_form) * get<0>(excision_rhat);
  for (size_t i = 1; i < 3; ++i) {
    get(*result) += excision_normal_one_form.get(i) * excision_rhat.get(i);
  }
  get(*result) *= -dt_horizon_00 / horizon_00;

  // Add the n_i xi^j partial_j \beta^i term (without the Y00 or norm
  // factor)
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += excision_rhat.get(j) * excision_normal_one_form.get(i) *
                      deriv_of_distorted_shift.get(j, i);
    }
  }

  // Put in the norm factor.
  get(*result) /= get(excision_normal_one_form_norm);

  // Add the dlapse term (without the Y00 factor).
  for (size_t i = 0; i < 3; ++i) {
    get(*result) += deriv_lapse.get(i) * excision_rhat.get(i);
  }

  // Put in the Y00 factor.
  get(*result) *= Y00;

  // Add the final factor to result
  for (size_t i = 0; i < 3; ++i) {
    get(*result) += deriv_normal_one_form.get(i) *
                    (Y00 * dt_lambda_00 * excision_rhat.get(i) +
                     distorted_components_of_grid_shift.get(i) -
                     excision_rhat.get(i) * (dt_horizon_00 / horizon_00) *
                         (Y00 * lambda_00 - grid_frame_excision_sphere_radius));
  }
}
}  // namespace control_system::size
