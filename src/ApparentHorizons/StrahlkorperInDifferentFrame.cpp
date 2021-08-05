// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"

#include <cmath>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "NumericalAlgorithms/RootFinding/RootBracketing.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    const gsl::not_null<Strahlkorper<DestFrame>*> dest_strahlkorper,
    const Strahlkorper<SrcFrame>& src_strahlkorper,
    const domain::CoordinateMapBase<SrcFrame, DestFrame, 3>& map_src_to_dest,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) noexcept {
  // Temporary storage; reduce the number of allocations.
  Variables<
      tmpl::list<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>,
                 ::Tags::Tempi<1, 3, SrcFrame>, ::Tags::TempI<2, 3, SrcFrame>,
                 ::Tags::TempI<3, 3, DestFrame>, ::Tags::TempScalar<4>,
                 ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
                 ::Tags::TempScalar<7>, ::Tags::TempScalar<8>>>
      temp_buffer(src_strahlkorper.ylm_spherepack().physical_size());
  auto& src_theta_phi =
      get<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>>(temp_buffer);
  auto& r_hat = get<::Tags::Tempi<1, 3, SrcFrame>>(temp_buffer);
  auto& src_cartesian_coords = Tags::TempI<2, 3, SrcFrame>(temp_buffer);
  auto& dest_cartesian_coords = Tags::TempI<3, 3, DestFrame>(temp_buffer);
  auto& src_radius = get<::Tags::TempScalar<4>>(temp_buffer);
  auto& bracket_r_min = get<::Tags::TempScalar<5>>(temp_buffer);
  auto& bracket_r_max = get<::Tags::TempScalar<6>>(temp_buffer);
  auto& f_bracket_r_min = get<::Tags::TempScalar<7>>(temp_buffer);
  auto& f_bracket_r_max = get<::Tags::TempScalar<8>>(temp_buffer);

  StrahlkorperTags::ThetaPhiCompute<SrcFrame>::function(
      make_not_null(&src_theta_phi), src_strahlkorper);
  // r_hat doesn't depend on the actual surface (that is, it is
  // identical for the src and dest surfaces), so we use
  // src_strahlkorper to compute it because it has a sensible max Ylm l.
  StrahlkorperTags::RhatCompute<SrcFrame>::function(make_not_null(&r_hat),
                                                    src_theta_phi);
  StrahlkorperTags::RadiusCompute<SrcFrame>::function(
      make_not_null(&src_radius), src_strahlkorper);
  StrahlkorperTags::CartesianCoordsCompute<SrcFrame>::function(
      make_not_null(&src_cartesian_coords), src_strahlkorper, src_radius,
      r_hat);
  dest_cartesian_coords =
      map_src_to_dest(src_cartesian_coords, time, functions_of_time);

  // To find the expansion center of the destination surface, take a
  // simple average of the Cartesian coordinates of the surface in the
  // destination frame.  An average should be good enough, since this
  // is only the expansion center (not the physical center), so its
  // only requirement is that it is centered enough so that the
  // surface is star-shaped.  If we want to re-center the strahlkorper
  // later, we can call `change_expansion_center_of_strahlkorper_to_physical`.
  const auto center_dest = [&dest_cartesian_coords]() noexcept {
    std::array<double, 3> center{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(center, d) =
          std::accumulate(dest_cartesian_coords.get(d).begin(),
                          dest_cartesian_coords.get(d).end(), 0.0) /
          dest_cartesian_coords.get(d).size();
    }
    return center;
  }();

  // Find the coordinate radius of the destination surface at each of
  // the angular collocation points of the destination surface. To do
  // so, for each index 's' (corresponding to an angular collocation
  // point), find the root 'r_dest' that zeroes this lambda function.
  //
  // This version of the function returns a std::optional<double>
  // because it might fail if r_dest is out of range of the map. Below
  // there is another version of the function that returns a double.
  const auto radius_function_for_bracketing =
      [&r_hat, &center_dest, &src_strahlkorper, &map_src_to_dest,
       &functions_of_time,
       &time](const double r_dest,
              const size_t s) noexcept -> std::optional<double> {
    // Get destination Cartesian coordinates of the point.
    const tnsr::I<double, 3, DestFrame> x_dest{
        {r_dest * get<0>(r_hat)[s] + center_dest[0],
         r_dest * get<1>(r_hat)[s] + center_dest[1],
         r_dest * get<2>(r_hat)[s] + center_dest[2]}};

    // Transform to source Cartesian coordinates of the point.
    const auto x_src = map_src_to_dest.inverse(x_dest, time, functions_of_time);
    if (not x_src) {
      // The inverse might fail, particularly if the point is outside
      // the domain (e.g. inside an excision boundary or outside the outer
      // boundary).
      return {};
    }

    // Find (r_src,theta_src,phi_src) in source coordinates.
    const double r_src =
        sqrt(square(get<0>(x_src.get())) + square(get<1>(x_src.get)()) +
             square(get<2>(x_src.get())));
    const double theta_src = acos(get<2>(x_src.get()) / r_src);
    const double phi_src = atan2(get<1>(x_src.get()), get<0>(x_src.get()));

    // Evaluate the radius of the surface at (theta_src,phi_src).
    const double r_surf_src = src_strahlkorper.radius(theta_src, phi_src);

    // If r_surf_src = r_src, then r_dest is on the surface.
    return r_surf_src - r_src;
  };

  // This version of the function will be used once the root is bracketed.
  const auto radius_function = [&r_hat, &center_dest, &src_strahlkorper,
                                &map_src_to_dest, &functions_of_time,
                                &time](const double r_dest,
                                       const size_t s) noexcept {
    return radius_function_for_bracketing(r_dest, s).get();
  };

  // We try to bracket the root between r_min and r_max.
  const auto [r_min, r_max] = [&dest_cartesian_coords,
                               &center_dest]() noexcept {
    const DataVector dest_radius =
        sqrt(square(get<0>(dest_cartesian_coords) - center_dest[0]) +
             square(get<1>(dest_cartesian_coords) - center_dest[1]) +
             square(get<2>(dest_cartesian_coords) - center_dest[2]));
    const auto minmax =
        std::minmax_element(dest_radius.begin(), dest_radius.end());
    return std::make_pair(*(minmax.first), *(minmax.second));
  }();

  // But r_min and r_max are only approximate, and there may be grid points
  // with radii outside that range. So we pad by 10% to be safe.
  const double padding = 0.10;
  bracket_r_min = r_min * (1.0 - padding);
  bracket_r_max = r_max * (1.0 + padding);
  RootFinder::bracket_possibly_undefined_function_in_interval(
      make_not_null(&bracket_r_min), make_not_null(&bracket_r_max),
      make_not_null(&f_bracket_r_min), make_not_null(&f_bracket_r_max),
      radius_function_for_bracketing);

  // Find the radius at each angular point.
  const auto radius_at_each_angle = RootFinder::toms748(
      radius_function, bracket_r_min, bracket_r_max, f_bracket_r_min,
      f_bracket_r_max, std::numeric_limits<double>::epsilon() * (r_min + r_max),
      2.0 * std::numeric_limits<double>::epsilon());

  // Reset the radius and center of the destination strahlkorper.
  // Keep the same l_max() and m_max() as the source strahlkorper.
  *dest_strahlkorper = Strahlkorper<DestFrame>(
      src_strahlkorper->l_max(), src_strahlkorper->m_max(),
      radius_at_each_angle, center_dest);
}
