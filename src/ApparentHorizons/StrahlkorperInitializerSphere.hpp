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
class Sphere : public StrahlkorperInitializer {
 public:
  struct Lmax {
    using type = size_t;
    static constexpr OptionString_t help = {
        "Maximum spherical-harmonic L of the surface."};
  };

  struct Radius {
    using type = double;
    static constexpr OptionString_t help = {"Radius of the surface."};
  };

  struct Center {
    using type = std::array<double, 3>;
    static constexpr OptionString_t help = {"Center of the surface."};
    static size_t size() { return 3; }
  };

  using options = tmpl::list<Lmax, Radius, Center>;
  static constexpr OptionString_t help = {"Creates a spherical Strahlkorper."};

  Sphere(size_t l_max, double radius, std::array<double, 3> center,
         const OptionContext& context = {}) noexcept;

  Sphere() = delete;
  Sphere(const Sphere& /*rhs*/) = delete;
  Sphere& operator=(const Sphere& /*rhs*/) = delete;
  Sphere(Sphere&& /*rhs*/) noexcept = default;
  Sphere& operator=(Sphere&& /*rhs*/) noexcept = default;
  ~Sphere() override = default;

  Strahlkorper create_strahlkorper() const noexcept override;

 private:
  size_t l_max_;
  double radius_;
  std::array<double, 3> center_;
};
}  // namespace StrahlkorperInitializers
