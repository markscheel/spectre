// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/AnalyticData.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

/// \cond
namespace spec {
class Exporter;
}  // namespace spec
/// \endcond

namespace gr::AnalyticData {
class SpecNumericalData : public AnalyticDataBase<3>,
                          public MarkAsAnalyticData {
 public:
  SpecNumericalData() = default;
  explicit SpecNumericalData(std::string source_directory);
  ~SpecNumericalData();
  SpecNumericalData(const SpecNumericalData& rhs);
  SpecNumericalData& operator=(const SpecNumericalData& rhs);
  SpecNumericalData(SpecNumericalData&& /*rhs*/) = default;
  SpecNumericalData& operator=(SpecNumericalData&& /*rhs*/) = default;
  explicit SpecNumericalData(CkMigrateMessage* msg);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /*!
   * \brief Computes and returns spacetime quantities of numerical data loaded
   * from a SpEC elliptic solve of the XCTS equations.
   *
   * \param x Cartesian coordinates of the position at which to compute
   * spacetime quantities
   */
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const {
    // static_assert(
    //     tmpl2::flat_all_v<
    //         tmpl::list_contains_v<tags<DataType, Frame::Inertial>, Tags>...>,
    //     "At least one of the requested tags is not supported. The requested "
    //     "tags are listed as template parameters of the `variables`
    //     function.");
    // IntermediateVars<DataType, Frame::Inertial> cache(get_size(*x.begin()));
    // IntermediateComputer<DataType, Frame::Inertial> computer(*this, x);
    // return {cache.get_var(computer, Tags{})...};
  }

  /*!
   * \brief Buffer for caching computed intermediates and quantities that we do
   * not want to recompute across the solution's implementation
   *
   * \details See `internal_tags` documentation for details on what quantities
   * the internal tags represent
   */
  template <typename DataType>
  using CachedBuffer = CachedTempBuffer<
      gr::Tags::Lapse<DataType>, gr::Tags::Shift<3, Frame::Inertial, DataType>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>,
      DerivSpatialMetric<DataType, Frame::Inertial>>;

  template <typename DataType>
  class IntermediateComputer {
   public:
    using CachedBuffer = SpecNumericalData::CachedBuffer<DataType>;

    /*!
     * \brief Constructs a computer for spacetime quantities of a given
     * `gr::AnalyticData::BrillLindquist` solution at at a specific
     * Cartesian position
     *
     * \param analytic_data the given `gr::AnalyticData::BrillLindquist` data
     * \param x Cartesian coordinates of the position at which to compute
     * spacetime quantities
     */
    IntermediateComputer(const spec::Exporter* spec_exporter,
                         const tnsr::I<DataType, 3, Frame::Inertial>& coords);

   private:
    const spec::Exporter& spec_exporter_;
    const tnsr::I<DataType, 3, Frame::Inertial>& coords_;
  };

 private:
  std::string source_directory_{};
  std::vector<std::string> vars_to_interpolate_{
      "Nid_g",     // (lower 3-metric)
      "Nid_K",     // (lower extrinsic curvature)
      "Nid_N",     // (lapse function)
      "Nid_Shift"  // (upper shift vector)
  };
  // Use a raw pointer because libstdc++ has a weird bug where a
  // std::unique_ptr reqires the type to be complete.
  spec::Exporter* spec_exporter_{nullptr};
};
}  // namespace gr::AnalyticData
