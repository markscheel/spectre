// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperCoordsInDifferentFrame.hpp"

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename SrcFrame, typename DestFrame>
void strahlkorper_coords_in_different_frame(
    gsl::not_null<tnsr::I<DataVector, 3, DestFrame>*> cartesian_coords,
    const Strahlkorper<SrcFrame>& strahlkorper,
    const domain::CoordinateMapBase<SrcFrame, DestFrame, 3>& map_src_to_dest,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) noexcept {
  Variables<tmpl::list<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>,
                       ::Tags::Tempi<1, 3, SrcFrame>,
                       ::Tags::TempI<2, 3, SrcFrame>, ::Tags::TempScalar<3>>>
      temp_buffer(strahlkorper.ylm_spherepack().physical_size());
  auto& theta_phi =
      get<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>>(temp_buffer);
  auto& r_hat = get<::Tags::Tempi<1, 3, SrcFrame>>(temp_buffer);
  auto& src_cartesian_coords = Tags::TempI<2, 3, SrcFrame>(temp_buffer);
  auto& radius = get<::Tags::TempScalar<3>>(temp_buffer);
  StrahlkorperTags::ThetaPhiCompute<SrcFrame>::function(
      make_not_null(&theta_phi), strahlkorper);
  StrahlkorperTags::RhatCompute<SrcFrame>::function(make_not_null(&r_hat),
                                                    theta_phi);
  StrahlkorperTags::RadiusCompute<SrcFrame>::function(make_not_null(&radius),
                                                      strahlkorper);
  StrahlkorperTags::CartesianCoordsCompute<SrcFrame>::function(
      make_not_null(&src_cartesian_coords), strahlkorper, radius, r_hat);
  *cartesian_coords =
      map_src_to_dest(src_cartesian_coords, time, functions_of_time);
}
