// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/ControlErrors/SizeControlInfo.hpp"
#include "ControlSystem/ControlErrors/SizeControlState.hpp"

namespace SizeControlStates {
class Initial : public SizeControlState {
 public:
  void update(const gsl::not_null<SizeControlInfo*> info,
              const SizeControlStateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  double control_signal(const SizeControlInfo& info,
                        const SizeControlStateControlSignalArgs&
                            control_signal_args) const override;
};
}  // namespace SizeControlStates
