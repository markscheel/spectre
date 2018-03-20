// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "ApparentHorizons/HorizonManager.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"

struct TestMetavariables;

template <class Metavariables>
struct HorizonManagerComponent {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<db::get_databox_list<
      tmpl::list<typename Metavariables::number_of_elements_tag>>>;
  using options = tmpl::list<>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache);
  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    if (next_phase == Metavariables::Phase::CheckAnswer) {
      auto& my_proxy = Parallel::get_parallel_component<
          HorizonManagerComponent<Metavariables>>(
          *(global_cache.ckLocalBranch()));
      my_proxy
          .template simple_action<Actions::HorizonManager::PrintNumElements>();
    }
  }
};

template <class Metavariables>
void HorizonManagerComponent<Metavariables>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
  auto& my_proxy = Parallel::get_parallel_component<HorizonManagerComponent>(
      *(global_cache.ckLocalBranch()));
  my_proxy.template simple_action<Actions::HorizonManager::InitNumElements>();
}

template <class Metavariables>
struct DgElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using array_index = ElementIndex<3>;
  using initial_databox = db::DataBox<db::get_databox_list<tmpl::list<>>>;
  using options = tmpl::list<typename Metavariables::domain_creator_tag>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<3, Frame::Inertial>>
          domain_creator) noexcept;
  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    if (next_phase == Metavariables::Phase::CountElements) {
      auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
          *(global_cache.ckLocalBranch()));

      dg_element_array
          .template simple_action<Actions::HorizonManager::SendNumElements<
              HorizonManagerComponent<Metavariables>>>();
    }
  }
};

template <class Metavariables>
void DgElementArray<Metavariables>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
    std::unique_ptr<DomainCreator<3, Frame::Inertial>>
        domain_creator) noexcept {
  auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
      *(global_cache.ckLocalBranch()));

  auto domain = domain_creator->create_domain();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    const std::vector<ElementId<3>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (size_t i = 0, which_proc = 0,
                number_of_procs =
                    static_cast<size_t>(Parallel::number_of_procs());
         i < element_ids.size(); ++i) {
      dg_element_array(ElementIndex<3>(element_ids[i]))
          .insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();
}

struct TestMetavariables {
  using component_list = tmpl::list<HorizonManagerComponent<TestMetavariables>,
                                    DgElementArray<TestMetavariables>>;
  static constexpr const char* const help{"Test HorizonManager in parallel"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  struct number_of_elements_tag : db::DataBoxTag {
    static constexpr db::DataBoxString label = "number_of_elements";
    using type = size_t;
  };

  using domain_creator_tag = OptionTags::DomainCreator<3, Frame::Inertial>;

  enum class Phase { Initialization, CountElements, CheckAnswer, Exit };
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    if (current_phase == Phase::Initialization) {
      return Phase::CountElements;
    }
    if (current_phase == Phase::CountElements) {
      return Phase::CheckAnswer;
    }
    return Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.cpp"
