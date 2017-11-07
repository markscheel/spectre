// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/Strahlkorper.hpp"
#include "Options/Factory.hpp"

/*!
 * \ingroup ApparentHorizons
 * Holds StrahlkorperInitializers.
 */
namespace StrahlkorperInitializers {
class Sphere;
}  // namespace StrahlkorperInitializers

/*!
 *  \ingroup ApparentHorizons
 *  Encodes a function that creates a Strahlkorper.
 */
class StrahlkorperInitializer : public Factory<StrahlkorperInitializer> {
 public:
  using creatable_classes = typelist<StrahlkorperInitializers::Sphere>;

  StrahlkorperInitializer() = default;
  StrahlkorperInitializer(const StrahlkorperInitializer& /*rhs*/) = delete;
  StrahlkorperInitializer& operator=(const StrahlkorperInitializer& /*rhs*/) =
      delete;
  StrahlkorperInitializer(StrahlkorperInitializer&& /*rhs*/) noexcept = default;
  StrahlkorperInitializer& operator=(
      StrahlkorperInitializer&& /*rhs*/) noexcept = default;
  virtual ~StrahlkorperInitializer() = default;

  virtual Strahlkorper create_strahlkorper() const noexcept = 0;
};

#include "ApparentHorizons/StrahlkorperInitializerSphere.hpp"
