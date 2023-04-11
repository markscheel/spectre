// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <string>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SpecInitialData.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::AnalyticData {
namespace {

template <typename DataType>
void test(const DataType& used_for_size) {
  const std::string data_directory =
      "/panfs/ds09/sxs/ffoucart/InitialData/NsNs-SpectreTest/ZeroRhoOutside/"
      "EvID";
  const SpecInitialData spec_initial_data{data_directory, nullptr};

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 0.1);
  const auto x = make_with_random_values<tnsr::I<DataType, 3>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);

  spec_initial_data.variables(
      x, tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>{});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.SlabJet",
                  "[Unit][PointwiseFunctions]") {
  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5));
}
}  // namespace grmhd::AnalyticData
