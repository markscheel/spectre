// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"

#include <cmath>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatSide.hpp"
#include "Domain/CoordinateMaps/CylindricalSide.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace {
std::array<double, 3> rotate_to_z_axis(std::array<double, 3> input) noexcept {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::lower_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::upper_xi()}},
      input);
}
std::array<double, 3> flip_about_xy_plane(
    std::array<double, 3> input) noexcept {
  return std::array<double, 3>{input[0], input[1], -input[2]};
}
}  // namespace

namespace domain::creators {
CylindricalBinaryCompactObject::CylindricalBinaryCompactObject(
    typename CenterA::type center_A, typename CenterB::type center_B,
    typename RadiusA::type radius_A, typename RadiusB::type radius_B,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_grid_points_per_dim,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(rotate_to_z_axis(center_A)),
      center_B_(rotate_to_z_axis(center_B)),
      radius_A_(radius_A),
      radius_B_(radius_B),
      outer_radius_(outer_radius),
      initial_refinement_(initial_refinement),
      initial_grid_points_per_dim_(initial_grid_points_per_dim),
      time_dependence_(std::move(time_dependence)),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
  if (center_A_[2] <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input center_A is expected to be positive");
  }
  if (center_B_[2] >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input center_B is expected to be negative");
  }
  if (radius_A_ <= 0.0 or radius_B_ <= 0.0) {
    PARSE_ERROR(context, "Radii are expected to be positive");
  }
  if (radius_A_ < radius_B_) {
    PARSE_ERROR(context, "radius_A should not be smaller than radius_B");
  }
  if (std::abs(center_A_[2]) > std::abs(center_B_[2])) {
    PARSE_ERROR(context,
                "We expect |x_A| <= |x_B|, for x the x-coordinate of the "
                "center of object A/B.  We should roughly have "
                "radius_A x_A + radius_B x_B = 0 (i.e. for BBHs the "
                "center of mass should be about at the origin).");
  }
  // The value 3.0 * (center_A_[2] - center_B_[2]) is what is
  // chosen in SpEC as the inner radius of the innermost outer sphere.
  if (outer_radius_ < 3.0 * (center_A_[2] - center_B_[2])) {
    PARSE_ERROR(context,
                "Outer boundary radius is too small. Please increase it "
                "beyond "
                    << 3.0 * (center_A_[2] - center_B_[2]));
  }

  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }

  if ((outer_boundary_condition_ == nullptr) xor
      (inner_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // The choices made below for the quantities xi, z_cutting_plane_,
  // and xi_min_sphere_e are the ones made in SpEC, and in the
  // Appendix of https://arxiv.org/abs/1206.3015.  Other choices could
  // be made that would still result in a reasonable Domain. In
  // particular, during a SpEC BBH evolution the excision boundaries
  // can sometimes get too close to z_cutting_plane_, and the
  // simulation must be halted and regridded with a different choice
  // of z_cutting_plane_, so it may be possible to choose a different
  // initial value of z_cutting_plane_ that reduces the number of such
  // regrids or eliminates them.

  // xi is the quantity in Eq. (A10) of
  // https://arxiv.org/abs/1206.3015 that represents how close the
  // cutting plane is to either center.  Unfortunately, there is a
  // discrepancy between what xi means in the paper and what it is in
  // the code.  I (Mark) think that this is a typo in the paper,
  // because otherwise the domain doesn't make sense.  To fix this,
  // either Eq. (A9) in the paper should have xi -> 1-xi, or Eq. (A10)
  // should have x_A and x_B swapped.
  // Here we will use the same definition of xi in Eq. (A10), but we
  // will swap xi -> 1-xi in Eq. (A9).
  // Therefore, xi = 0 means that the cutting plane passes through the center of
  // object B, and xi = 1 means that the cutting plane passes through
  // the center of object A.  Note that for |x_A| <= |x_B| (as assumed
  // above), xi is always <= 1/2.
  constexpr double xi_min = 0.25;
  // Same as Eq. (A10)
  const double xi =
      std::max(xi_min, std::abs(center_A_[2]) /
                           (std::abs(center_A_[2]) + std::abs(center_B_[2])));

  // Compute cutting plane
  // This is Eq. (A9) with xi -> 1-xi.
  z_cutting_plane_ = cut_spheres_offset_factor_ *
                     ((1.0 - xi) * center_B_[2] + xi * center_A_[2]);

  // Number of blocks without the SphereEs
  // Note: support for SphereEs will be added in the next PR,
  // and then the ERROR below will be removed and replaced
  // with code that changes the number of blocks.
  number_of_blocks_ = 46;

  // Add SphereE blocks if necessary.  Note that
  // https://arxiv.org/abs/1206.3015 has a mistake just above
  // Eq. (A.11) and the same mistake above Eq. (A.20), where it lists
  // the wrong mass ratio (for BBHs). The correct statement is that if
  // xi <= 1/3, this means that the mass ratio (for BBH) is large (>=2)
  // and we should add SphereE blocks.
  constexpr double xi_min_sphere_e = 1.0 / 3.0;
  if (xi <= xi_min_sphere_e) {
    // The following ERROR will be removed in an upcoming PR that
    // will support higher mass ratios.
    ERROR(
        "We currently only support domains where objects A and B are "
        "approximately the same size, and approximately the same distance from "
        "the origin.  More technically, we support xi > "
        << xi_min_sphere_e << ", but the value of xi is " << xi
        << ". Support for more general domains will be added in the near "
           "future");
  }

}

Domain<3> CylindricalBinaryCompactObject::create_domain() const noexcept {
  using BcMap = DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>;

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coordinate_maps{};

  std::vector<BcMap> boundary_conditions_all_blocks{};

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const std::array<double, 3> center_cutting_plane = {0.0, 0.0,
                                                      z_cutting_plane_};

  // The labels EA, EB, EE, etc are from Figure 20 of
  // https://arxiv.org/abs/1206.3015
  //
  // center_EA and radius_EA are the center and outer-radius of the
  // cylindered-sphere EA in Figure 20.
  //
  // center_EB and radius_EB are the center and outer-radius of the
  // cylindered-sphere EB in Figure 20.
  //
  // radius_MB is eq. A16 or A23 in the paper (depending on whether
  // the EE spheres exist), and is the radius of the circle where the EB sphere
  // intersects the cutting plane.
  const auto [center_EA, radius_EA, center_EB, radius_MB,
              radius_EB] = [this]() noexcept {
    const std::array<double, 3> center_EB_l = {
        0.0, 0.0, center_B_[2] * cut_spheres_offset_factor_};
    std::array<double, 3> center_EA_l{};
    double radius_EA_l = std::numeric_limits<double>::signaling_NaN();
    double radius_EB_l = std::numeric_limits<double>::signaling_NaN();
    double radius_MB_l = std::numeric_limits<double>::signaling_NaN();
    const double radius_MB_factor =
        std::abs(cut_spheres_offset_factor_ * center_B_[2] - z_cutting_plane_);
    radius_MB_l = radius_MB_factor;
    center_EA_l = {0.0, 0.0, cut_spheres_offset_factor_ * center_A_[2]};
    radius_EA_l = sqrt(square(center_EA_l[2] - z_cutting_plane_) +
                       square(radius_MB_factor));
    radius_EB_l = sqrt(2.0) * std::abs(center_EB_l[2] - z_cutting_plane_);
    return std::make_tuple(center_EA_l, radius_EA_l, center_EB_l, radius_MB_l,
                           radius_EB_l);
  }();

  // Construct vector<CoordMap>s that go from logical coordinates to
  // various blocks making up a unit right cylinder.  These blocks are
  // either the central square blocks, or the surrounding wedge
  // blocks. The radii and bounds are what are expected by the
  // CylindricalEndcap maps, (except cylinder_inner_radius, which
  // determines the internal block boundaries inside the cylinder, and
  // which the CylindricalEndcap maps don't care about).
  const double cylinder_inner_radius = 0.5;
  const double cylinder_outer_radius = 1.0;
  const double cylinder_lower_bound_z = -1.0;
  const double cylinder_upper_bound_z = 1.0;
  const auto logical_to_cylinder_center_maps =
      cyl_wedge_coord_map_center_blocks(cylinder_inner_radius,
                                        cylinder_lower_bound_z,
                                        cylinder_upper_bound_z);
  const auto logical_to_cylinder_surrounding_maps =
      cyl_wedge_coord_map_surrounding_blocks(
          cylinder_inner_radius, cylinder_outer_radius, cylinder_lower_bound_z,
          cylinder_upper_bound_z, 0.0);

  // Lambda that takes a CylindricalEndcap map and a DiscreteRotation
  // map, composes it with the logical-to-cylinder maps, and adds it
  // to the list of coordinate maps. Also adds boundary conditions if
  // requested.
  auto add_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_center_maps,
       &logical_to_cylinder_surrounding_maps, &boundary_conditions_all_blocks,
       this](const CoordinateMaps::CylindricalEndcap& endcap_map,
             const CoordinateMaps::DiscreteRotation<3>& rotation_map,
             const bool add_inner_boundary_condition = false,
             const bool add_outer_boundary_condition = false) noexcept {
        auto new_logical_to_cylinder_center_maps =
            domain::make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_center_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_center_maps.begin()),
            std::make_move_iterator(new_logical_to_cylinder_center_maps.end()));
        auto new_logical_to_cylinder_surrounding_maps =
            domain::make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_surrounding_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.end()));

        // outer_boundary_condition_ == nullptr means do not add
        // any boundary conditions at all.
        if (outer_boundary_condition_ != nullptr) {
          for (size_t i = 0; i < 5; ++i) {
            BcMap bcs{};
            if (add_outer_boundary_condition) {
              bcs[Direction<3>::upper_zeta()] =
                  outer_boundary_condition_->get_clone();
            }
            if (add_inner_boundary_condition) {
              // The upper_zeta() below is correct because all the
              // CylindricalEndcaps with inner boundaries have upper_zeta
              // on the inner boundary.
              bcs[Direction<3>::upper_zeta()] =
                  inner_boundary_condition_->get_clone();
            }
            boundary_conditions_all_blocks.push_back(std::move(bcs));
          }
        }
      };

  // Lambda that takes a CylindricalFlatEndcap map and a
  // DiscreteRotation map, composes it with the logical-to-cylinder
  // maps, and adds it to the list of coordinate maps. Also adds
  // boundary conditions if requested.
  auto add_flat_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_center_maps,
       &logical_to_cylinder_surrounding_maps, &boundary_conditions_all_blocks,
       this](const CoordinateMaps::CylindricalFlatEndcap& endcap_map,
             const CoordinateMaps::DiscreteRotation<3>& rotation_map) noexcept {
        auto new_logical_to_cylinder_center_maps =
            domain::make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_center_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_center_maps.begin()),
            std::make_move_iterator(new_logical_to_cylinder_center_maps.end()));
        auto new_logical_to_cylinder_surrounding_maps =
            domain::make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_surrounding_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.end()));

        // inner_boundary_condition_ == nullptr means do not add
        // any boundary conditions at all.
        if (inner_boundary_condition_ != nullptr) {
          for (size_t i = 0; i < 5; ++i) {
            BcMap bcs{};
            // Note that upper_zeta below is correct: inner boundary
            // condition for FlatEndcaps are on upper_zeta faces.
            bcs[Direction<3>::upper_zeta()] =
                inner_boundary_condition_->get_clone();
            boundary_conditions_all_blocks.push_back(std::move(bcs));
          }
        }
      };

  // Construct vector<CoordMap>s that go from logical coordinates to
  // various blocks making up a right cylindrical shell of inner radius 1,
  // outer radius 2, and z-extents from -1 to +1.  These blocks are
  // either the central square blocks, or the surrounding wedge
  // blocks. The radii and bounds are what are expected by the
  // CylindricalEndcap maps, (except cylinder_inner_radius, which
  // determines the internal block boundaries inside the cylinder, and
  // which the CylindricalEndcap maps don't care about).
  const double cylindrical_shell_inner_radius = 1.0;
  const double cylindrical_shell_outer_radius = 2.0;
  const double cylindrical_shell_lower_bound_z = -1.0;
  const double cylindrical_shell_upper_bound_z = 1.0;
  const auto logical_to_cylindrical_shell_maps =
      cyl_wedge_coord_map_surrounding_blocks(
          cylindrical_shell_inner_radius, cylindrical_shell_outer_radius,
          cylindrical_shell_lower_bound_z, cylindrical_shell_upper_bound_z,
          1.0);

  // Lambda that takes a CylindricalSide map and a DiscreteRotation
  // map, composes it with the logical-to-cylinder maps, and adds it
  // to the list of coordinate maps.  Also adds boundary conditions if
  // requested.
  auto add_side_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylindrical_shell_maps,
       &boundary_conditions_all_blocks,
       this](const CoordinateMaps::CylindricalSide& side_map,
             const CoordinateMaps::DiscreteRotation<3>& rotation_map,
             const bool add_inner_boundary_condition = false,
             const bool add_outer_boundary_condition = false) noexcept {
        auto new_logical_to_cylindrical_shell_maps =
            domain::make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial, 3>(
                logical_to_cylindrical_shell_maps, side_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylindrical_shell_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylindrical_shell_maps.end()));

        // outer_boundary_condition_ == nullptr means do not add
        // any boundary conditions at all.
        if (outer_boundary_condition_ != nullptr) {
          for (size_t i = 0; i < 4; ++i) {
            BcMap bcs{};
            if (add_outer_boundary_condition) {
              bcs[Direction<3>::upper_xi()] =
                  outer_boundary_condition_->get_clone();
            }
            if (add_inner_boundary_condition) {
              // upper_xi() below is correct, since all
              // CylindricalSide maps with inner boundaries
              // have the inner boundary at upper_xi.
              bcs[Direction<3>::upper_xi()] =
                  inner_boundary_condition_->get_clone();
            }
            boundary_conditions_all_blocks.push_back(std::move(bcs));
          }
        }
      };

  // z_cut_EA is the z_plane position for CA and EA endcaps.
  const double z_cut_EA =
      z_cutting_plane_ + 1.5 * (center_EA[2] - z_cutting_plane_);

  // CA Filled Cylinder
  // 5 blocks: 0 thru 4
  add_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalEndcap(center_EA, make_array<3>(0.0),
                                        center_cutting_plane, radius_EA,
                                        outer_radius_, z_cut_EA),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), false, true);

  // CA Cylinder
  // 4 blocks: 5 thru 8
  add_side_to_list_of_maps(
      CoordinateMaps::CylindricalSide(
          center_EA, make_array<3>(0.0), center_cutting_plane, radius_EA,
          outer_radius_, z_cutting_plane_, z_cut_EA),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), false, true);

  // EA Filled Cylinder
  // 5 blocks: 9 thru 13
  add_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalEndcap(center_EA, center_A_, center_A_,
                                        radius_EA, radius_A_, z_cut_EA),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), true, false);

  // EA Cylinder
  // 4 blocks: 14 thru 17
  add_side_to_list_of_maps(
      CoordinateMaps::CylindricalSide(center_EA, center_A_, center_A_,
                                      radius_EA, radius_A_, z_cutting_plane_,
                                      z_cut_EA),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), true, false);

  const double z_cut_EB =
      z_cutting_plane_ + 1.5 * (center_EB[2] - z_cutting_plane_);

  // EB Filled Cylinder
  // 5 blocks: 18 thru 22
  add_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalEndcap(
          flip_about_xy_plane(center_EB), flip_about_xy_plane(center_B_),
          flip_about_xy_plane(center_B_), radius_EB, radius_B_, -z_cut_EB),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), true, false);

  // EB Cylinder
  // 4 blocks: 23 thru 26
  add_side_to_list_of_maps(
      CoordinateMaps::CylindricalSide(center_EB, center_B_, center_B_,
                                      radius_EB, radius_B_, z_cut_EB,
                                      z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), true, false);

  // MA Filled Cylinder
  // 5 blocks: 27 thru 31
  add_flat_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalFlatEndcap(center_cutting_plane, center_A_,
                                            center_A_, radius_MB, radius_A_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // MB Filled Cylinder
  // 5 blocks: 32 thru 36
  add_flat_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalFlatEndcap(
          flip_about_xy_plane(center_cutting_plane),
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_B_),
          radius_MB, radius_B_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // CB Filled Cylinder
  // 5 blocks: 37 thru 41
  add_endcap_to_list_of_maps(
      CoordinateMaps::CylindricalEndcap(
          flip_about_xy_plane(center_EB), make_array<3>(0.0),
          flip_about_xy_plane(center_cutting_plane), radius_EB, outer_radius_,
          -z_cut_EB),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis), false, true);

  // CB Cylinder
  // 4 blocks: 42 thru 45
  add_side_to_list_of_maps(
      CoordinateMaps::CylindricalSide(
          center_EB, make_array<3>(0.0), center_cutting_plane, radius_EB,
          outer_radius_, z_cut_EB, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis), false, true);

  Domain<3> domain{std::move(coordinate_maps),
                   std::move(boundary_conditions_all_blocks)};

  if (not time_dependence_->is_none()) {
    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block,
          std::move(time_dependence_->block_maps(number_of_blocks_)[block]));
    }
  }
  return domain;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_extents() const noexcept {
  return {number_of_blocks_, make_array<3>(initial_grid_points_per_dim_)};
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_refinement_levels() const noexcept {
  return {number_of_blocks_, make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CylindricalBinaryCompactObject::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}

}  // namespace domain::creators
