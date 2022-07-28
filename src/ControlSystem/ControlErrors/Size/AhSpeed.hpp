// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"

namespace control_system::size::States {
class AhSpeed : public State {
 public:
  void update(const gsl::not_null<Info*> info,
              const StateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 92 of \cite Hemberger2012jz.
  double control_signal(
      const Info& info,
      const ControlSignalArgs& control_signal_args) const override;
};
}  // namespace control_system::size::States
