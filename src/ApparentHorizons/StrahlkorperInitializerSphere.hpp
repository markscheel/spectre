// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "ApparentHorizons/StrahlkorperInitializer.hpp"
#include "Options/Options.hpp"

namespace StrahlkorperInitializers {

/*!
 * \ingroup ApparentHorizons
 * \brief Makes a spherical Strahlkorper.
 *
 * \details Input file options are Lmax, Radius, and Center.
 */
template <typename Frame>
class Sphere : public StrahlkorperInitializer<Frame> {
 public:
  struct Lmax {
    using type = size_t;
    static size_t lower_bound() { return 3; }
    static constexpr OptionString help = {
        "Maximum spherical-harmonic L of the surface."};
  };

  struct Radius {
    using type = double;
    static double lower_bound() { return 0.0; }
    static constexpr OptionString help = {"Radius of the surface."};
  };

  struct Center {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"Center of the surface."};
    static size_t size() { return 3; }
  };

  using options = tmpl::list<Lmax, Radius, Center>;
  static constexpr OptionString help = {"Creates a spherical Strahlkorper."};

  Sphere(size_t l_max, double radius, std::array<double, 3> center,
         const OptionContext& context = {}) noexcept;

  Sphere() = default;
  Sphere(const Sphere& /*rhs*/) = delete;
  Sphere& operator=(const Sphere& /*rhs*/) = delete;
  Sphere(Sphere&& /*rhs*/) noexcept = default;
  Sphere& operator=(Sphere&& /*rhs*/) noexcept = default;
  ~Sphere() override = default;

  Strahlkorper<Frame> create_strahlkorper() const noexcept override;

 private:
  size_t l_max_{
      0};  // Cannot disable default constructor, so put bad value here.
  double radius_{2.0};
  std::array<double, 3> center_{{0., 0., 0.}};
};
}  // namespace StrahlkorperInitializers
