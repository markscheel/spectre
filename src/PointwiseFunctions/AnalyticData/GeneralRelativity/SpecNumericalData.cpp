// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GeneralRelativity/SpecNumericalData.hpp"

#include <Exporter.hpp>  // The SpEC Exporter

#include "Utilities/System/ParallelInfo.hpp"

namespace gr::AnalyticData {
SpecNumericalData::SpecNumericalData(std::string source_directory)
    : source_directory_(std::move(source_directory)),
      spec_exporter_(
          new spec::Exporter(static_cast<size_t>(sys::number_of_procs()),
                             source_directory_, vars_to_interpolate_)) {}

void SpecNumericalData::pup(PUP::er& p) {
  p | source_directory_;
  p | vars_to_interpolate_;
  if (p.isUnpacking()) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    spec_exporter_ =
        new spec::Exporter(static_cast<size_t>(sys::number_of_procs()),
                           source_directory_, vars_to_interpolate_);
  }
}

template <typename DataType>
SpecNumericalData::IntermediateComputer<DataType>::IntermediateComputer(
    const spec::Exporter* spec_exporter,
    const tnsr::I<DataType, 3, Frame::Inertial>& coords)
    : spec_exporter_(spec_exporter), coords_(coords) {}

SpecNumericalData::~SpecNumericalData() { delete spec_exporter_; }
}  // namespace gr::AnalyticData
