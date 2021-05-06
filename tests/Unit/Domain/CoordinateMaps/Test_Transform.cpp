// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Domain/CoordinateMaps/Transform.hpp"

namespace {
template <size_t Dim, typename SrcFrame, typename DestFrame,
          typename DataType>
void test_transform_to_different_frame(const DataType& used_for_size) {
  tnsr::ii<DataType, Dim, DestFrame> (*f)(
      const tnsr::ii<DataType, Dim, SrcFrame>&,
      const ::Jacobian<DataType, Dim, DestFrame, SrcFrame>&) =
      transform::to_different_frame<Dim, SrcFrame, DestFrame>;
  pypp::check_with_random_values<1>(f, "Transform", "to_different_frame",
                                    {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, typename SrcFrame, typename DestFrame, typename DataType>
void test_transform_first_index_to_different_frame(
    const DataType& used_for_size) {
  tnsr::ijj<DataType, Dim, DestFrame> (*f)(
      const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                   index_list<SpatialIndex<Dim, UpLo::Lo, SrcFrame>,
                              SpatialIndex<Dim, UpLo::Lo, DestFrame>,
                              SpatialIndex<Dim, UpLo::Lo, DestFrame>>>&,
      const ::Jacobian<DataType, Dim, DestFrame, SrcFrame>&) =
      transform::first_index_to_different_frame<Dim, SrcFrame, DestFrame>;
  pypp::check_with_random_values<1>(f, "Transform",
                                    "first_index_to_different_frame",
                                    {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Transform",
                  "[Domain][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Domain/CoordinateMaps/");
  const DataVector dv(5);
  test_transform_to_different_frame<1, Frame::Grid, Frame::Inertial>(dv);
  test_transform_to_different_frame<2, Frame::Grid, Frame::Inertial>(dv);
  test_transform_to_different_frame<3, Frame::Grid, Frame::Inertial>(dv);
  test_transform_first_index_to_different_frame<1, Frame::Logical,
                                                Frame::Grid>(dv);
  test_transform_first_index_to_different_frame<2, Frame::Logical,
                                                Frame::Grid>(dv);
  test_transform_first_index_to_different_frame<3, Frame::Logical,
                                                Frame::Grid>(dv);
}
