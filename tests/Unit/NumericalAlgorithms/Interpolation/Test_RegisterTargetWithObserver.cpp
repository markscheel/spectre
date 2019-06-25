// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "IO/Observer/Actions.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/RegisterTargetWithObserver.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
class DataVector;
namespace Tags {
template <size_t Dim, typename Frame>
struct Domain;
}  // namespace Tags
/// \endcond

namespace {

struct MockRegisterSingletonWithObserverWriter {
  struct Results {
    observers::ObservationId observation_id{};
  };
  static Results results;

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id) noexcept {
    results.observation_id = observation_id;
  }
};

MockRegisterSingletonWithObserverWriter::Results
    MockRegisterSingletonWithObserverWriter::results{};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              InterpolationTargetTag>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Registration,
          tmpl::list<intrp::Actions::RegisterTargetWithObserver<
              InterpolationTargetTag>>>>;
  using add_options_to_databox =
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template AddOptionsToDataBox<Metavariables>;

  using component_being_mocked =
      ::intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter>;
  using with_these_simple_actions =
      tmpl::list<MockRegisterSingletonWithObserverWriter>;
};

struct Metavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    struct mock_callback {
      using observation_types = tmpl::list<InterpolationTargetA>;
    };
    using post_interpolation_callback = mock_callback;
  };
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using observed_reduction_data_tags = tmpl::list<>;

  using component_list = tmpl::list<
      mock_interpolation_target<Metavariables, InterpolationTargetA>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Registration, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Register",
                  "[Unit]") {
  using metavars = Metavariables;
  using component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;
  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  runner.set_phase(Metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(&runner, 0,
                                              domain_creator.create_domain());
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(Metavariables::Phase::Registration);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(Metavariables::Phase::Testing);

  // Invoke all actions
  runner.invoke_queued_simple_action<component>(0);

  const auto& results = MockRegisterSingletonWithObserverWriter::results;
  CHECK(results.observation_id ==
        observers::ObservationId{0., Metavariables::InterpolationTargetA{}});

  // No more queued simple actions.
  CHECK(runner.is_simple_action_queue_empty<component>(0));
}

}  // namespace
