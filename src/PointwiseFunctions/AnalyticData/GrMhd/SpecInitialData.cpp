// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/SpecInitialData.hpp"

#include <Exporter.hpp>  // The SpEC Exporter
#include <pup.h>

namespace grmhd::AnalyticData {

// Define destructor where the spec::Exporter type is complete so
// std::unique_ptr is happy
SpecInitialData::~SpecInitialData() = default;

SpecInitialData::SpecInitialData(
    std::string data_directory,
    std::unique_ptr<equation_of_state_type> equation_of_state)
    : data_directory_(std::move(data_directory)),
      equation_of_state_(std::move(equation_of_state)),
      spec_exporter_(std::make_unique<spec::Exporter>(1, data_directory_,
                                                      vars_to_interpolate_)) {}

std::unique_ptr<evolution::initial_data::InitialData>
SpecInitialData::get_clone() const {
  return std::make_unique<SpecInitialData>(data_directory_,
                                           equation_of_state_->get_clone());
}

SpecInitialData::SpecInitialData(CkMigrateMessage* msg) : InitialData(msg) {}

void SpecInitialData::pup(PUP::er& p) {
  InitialData::pup(p);
  p | data_directory_;
  p | equation_of_state_;
  if (p.isUnpacking()) {
    spec_exporter_ = std::make_unique<spec::Exporter>(1, data_directory_,
                                                      vars_to_interpolate_);
  }
}

PUP::able::PUP_ID SpecInitialData::my_PUP_ID = 0;

template <typename DataType>
tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<DataType>>
SpecInitialData::interpolate_from_spec(const tnsr::I<DataType, 3>& x) const {
  // Transform coordinates into SpEC's expected format
  const size_t num_points = get_size(get<0>(x));
  std::vector<std::vector<double>> spec_grid_coords{num_points};
  for (size_t i = 0; i < num_points; ++i) {
    spec_grid_coords[i] = std::vector<double>{get_element(get<0>(x), i),
                                              get_element(get<1>(x), i),
                                              get_element(get<2>(x), i)};
  }
  // Allocate memory and point into it
  tuples::tagged_tuple_from_typelist<interpolated_tags<DataType>>
      interpolation_buffer{};
  std::vector<std::vector<double*>> buffer_pointers{
      vars_to_interpolate_.size()};
  ASSERT(tmpl::size<interpolated_tags<DataType>>::value ==
             vars_to_interpolate_.size(),
         "Mismatch between interpolation buffer size and number of variables.");
  size_t var_i = 0;
  tmpl::for_each<interpolated_tags<DataType>>([&interpolation_buffer,
                                               &buffer_pointers, &num_points,
                                               &var_i](const auto tag_v) {
    using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
    auto& tensor = get<tag>(interpolation_buffer);
    destructive_resize_components(make_not_null(&tensor), num_points);
    const size_t num_components = tensor.size();
    buffer_pointers[var_i].resize(num_components);
    // The SpEC exporter supports tensors up to symmetric rank 2, which are
    // ordered xx, xy, xz, yy, yz, zz. Because this is also the order in
    // which we store components in the Tensor class, we don't have to do
    // anything special here.
    // WARNING: If the Tensor storage order changes for some reason, this
    // code needs to be updated.
    for (size_t component_i = 0; component_i < num_components; ++component_i) {
      auto& component = tensor[component_i];
      auto& component_pointer = buffer_pointers[var_i][component_i];
      if constexpr (std::is_same_v<DataType, double>) {
        component_pointer = &component;
      } else {
        component_pointer = component.data();
      }
    }
    ++var_i;
  });
  // Interpolate!
  spec_exporter_->interpolate(buffer_pointers, spec_grid_coords, 0);
  return interpolation_buffer;
}

template tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<double>>
SpecInitialData::interpolate_from_spec(const tnsr::I<double, 3>& x) const;
template tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<DataVector>>
SpecInitialData::interpolate_from_spec(const tnsr::I<DataVector, 3>& x) const;

}  // namespace grmhd::AnalyticData
