// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/Strahlkorper.hpp"
#include "Options/Options.hpp"

/*!
 * \ingroup ApparentHorizons
 * Holds StrahlkorperInitializers.
 */
namespace StrahlkorperInitializers {
template <typename Frame>
class Sphere;
}  // namespace StrahlkorperInitializers

/*!
 *  \ingroup ApparentHorizons
 *  Encodes a function that creates a Strahlkorper.
 */
template <typename Frame>
class StrahlkorperInitializer {
 public:
  using creatable_classes = typelist<StrahlkorperInitializers::Sphere<Frame>>;

  StrahlkorperInitializer() = default;
  StrahlkorperInitializer(const StrahlkorperInitializer& /*rhs*/) = delete;
  StrahlkorperInitializer& operator=(const StrahlkorperInitializer& /*rhs*/) =
      delete;
  StrahlkorperInitializer(StrahlkorperInitializer&& /*rhs*/) noexcept = default;
  StrahlkorperInitializer& operator=(
      StrahlkorperInitializer&& /*rhs*/) noexcept = default;
  virtual ~StrahlkorperInitializer() = default;

  virtual Strahlkorper<Frame> create_strahlkorper() const noexcept = 0;
};

#include "ApparentHorizons/StrahlkorperInitializerSphere.hpp"
