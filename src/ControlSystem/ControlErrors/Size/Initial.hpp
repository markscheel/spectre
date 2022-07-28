// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"

namespace control_system::size::States {
class Initial : public State {
 public:
  void update(const gsl::not_null<Info*> info,
              const StateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  double control_signal(
      const Info& info,
      const ControlSignalArgs& control_signal_args) const override;
};
}  // namespace control_system::size::States
