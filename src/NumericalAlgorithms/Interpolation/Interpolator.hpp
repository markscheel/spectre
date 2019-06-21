// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmGroup.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PhaseDependentActionList.hpp"

namespace intrp {

/// \brief ParallelComponent responsible for collecting data from
/// `Element`s and interpolating it onto `InterpolationTarget`s.
///
/// For requirements on Metavariables, see InterpolationTarget
template <class Metavariables>
struct Interpolator {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<Actions::InitializeInterpolator>>>;
  //-:  using initial_databox =
  //-:      db::compute_databox_type<typename Actions::InitializeInterpolator::
  //-:                                   template
  //return_tag_list<Metavariables>>;
  using options = tmpl::list<>;
  static void initialize(Parallel::CProxy_ConstGlobalCache<Metavariables>&
                         /*global_cache*/) noexcept {};
  static void execute_next_phase(
      typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {};
};

//-:template <class Metavariables>
//-:void Interpolator<Metavariables>::initialize(
//-:    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
//-:  auto& my_proxy = Parallel::get_parallel_component<Interpolator>(
//-:      *(global_cache.ckLocalBranch()));
//-:  Parallel::simple_action<Actions::InitializeInterpolator>(my_proxy);
//-:}

}  // namespace intrp
