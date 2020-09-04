// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GlobalCache.

#pragma once

#include <boost/optional.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#include "Parallel/GlobalCache.decl.h"

namespace Parallel {

namespace GlobalCache_detail {

// Note: Returned list does not need to be size 1
template <class GlobalCacheTag, class Metavariables>
using get_list_of_matching_tags = typename list_of_matching_tags_helper<
    GlobalCacheTag,
    tmpl::append<get_const_global_cache_tags<Metavariables>,
                 get_mutable_global_cache_tags<Metavariables>>>::type;

template <class GlobalCacheTag, class Metavariables>
using type_for_get = typename type_for_get_helper<typename tmpl::front<
    get_list_of_matching_tags<GlobalCacheTag, Metavariables>>::type>::type;

template <class T, class = std::void_t<>>
struct has_component_being_mocked_alias : std::false_type {};

template <class T>
struct has_component_being_mocked_alias<
    T, std::void_t<typename T::component_being_mocked>> : std::true_type {};

template <class T>
constexpr bool has_component_being_mocked_alias_v =
    has_component_being_mocked_alias<T>::value;

template <typename ComponentToFind, typename ComponentFromList>
struct get_component_if_mocked_helper {
  static_assert(
      has_component_being_mocked_alias_v<ComponentFromList>,
      "The parallel component was not found, and it looks like it is not being "
      "mocked. Did you forget to add it to the "
      "'Metavariables::component_list'? See the first template parameter for "
      "the component that we are looking for and the second template parameter "
      "for the component that is being checked for mocking it.");
  using type = std::is_same<typename ComponentFromList::component_being_mocked,
                            ComponentToFind>;
};

template <typename... Tags>
auto make_mutable_cache_tag_storage(
    tuples::TaggedTuple<Tags...>&& input) noexcept {
  return tuples::TaggedTuple<MutableCacheTag<Tags>...>(
      std::make_tuple<typename Tags::type, std::vector<CkCallback>>(
          std::move(tuples::get<Tags>(input)), std::vector<CkCallback>{})...);
}

/// In order to be able to use a mock action testing framework we need to be
/// able to get the correct parallel component from the global cache even when
/// the correct component is a mock. We do this by having the mocked
/// components have a member type alias `component_being_mocked`, and having
/// `Parallel::get_component` check if the component to be retrieved is in the
/// `metavariables::component_list`. If it is not in the `component_list` then
/// we search for a mock component that is mocking the component we are trying
/// to retrieve.
template <typename ComponentList, typename ParallelComponent>
using get_component_if_mocked = tmpl::front<tmpl::type_from<tmpl::conditional_t<
    tmpl::list_contains_v<ComponentList, ParallelComponent>,
    tmpl::type_<tmpl::list<ParallelComponent>>,
    tmpl::lazy::find<ComponentList,
                     tmpl::type_<get_component_if_mocked_helper<
                         tmpl::pin<ParallelComponent>, tmpl::_1>>>>>>;

}  // namespace GlobalCache_detail

template <typename Metavariables>
class MutableGlobalCache : public CBase_MutableGlobalCache<Metavariables> {
 public:
  explicit MutableGlobalCache(tuples::tagged_tuple_from_typelist<
                              get_mutable_global_cache_tags<Metavariables>>
                                  mutable_global_cache) noexcept
      : mutable_global_cache_(
            GlobalCache_detail::make_mutable_cache_tag_storage(
                std::move(mutable_global_cache))) {}
  explicit MutableGlobalCache(CkMigrateMessage* /*msg*/) {}
  ~MutableGlobalCache() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        MutableGlobalCache<Metavariables>,
        CkIndex_MutableGlobalCache<Metavariables>>::registrar;
  }
  /// \cond
  MutableGlobalCache(const MutableGlobalCache&) = default;
  MutableGlobalCache& operator=(const MutableGlobalCache&) = default;
  MutableGlobalCache(MutableGlobalCache&&) = default;
  MutableGlobalCache& operator=(MutableGlobalCache&&) = default;
  /// \endcond

  template <typename GlobalCacheTag>
  auto get() const noexcept
      -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>&;

  // Entry method to mutate an object.  Internally calls
  // Function::apply(), where Function is a struct, and
  // Function::apply is a user-defined static function that mutates
  // the object.  Function::apply() takes as its first argument a
  // gsl::not_null pointer to the object named by the GlobalCacheTag,
  // and then the contents of 'args' as subsequent arguments.
  template <typename GlobalCacheTag, typename Function, typename... Args>
  void mutate(const std::tuple<Args...>& args) noexcept;

  // Not an entry method.
  template <typename GlobalCacheTag, typename Function>
  bool mutable_cache_item_is_ready(const Function& function) noexcept;

 private:
  tuples::tagged_tuple_from_typelist<
      get_mutable_global_cache_tag_storage<Metavariables>>
      mutable_global_cache_{};
};

template <typename Metavariables>
template <typename GlobalCacheTag>
auto MutableGlobalCache<Metavariables>::get() const noexcept
    -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>& {
  using tag =
      MutableCacheTag<tmpl::front<GlobalCache_detail::get_list_of_matching_tags<
          GlobalCacheTag, Metavariables>>>;
  if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
    return *(std::get<0>(tuples::get<tag>(mutable_global_cache_)).get());
  } else {
    return std::get<0>(tuples::get<tag>(mutable_global_cache_));
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function>
bool MutableGlobalCache<Metavariables>::mutable_cache_item_is_ready(
    const Function& function) noexcept {
  using tag = MutableCacheTag<
      tmpl::front<GlobalCache_detail::get_list_of_matching_mutable_tags<
          GlobalCacheTag, Metavariables>>>;
  boost::optional<CkCallback> optional_callback{};
  if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
    optional_callback =
        function(*(std::get<0>(tuples::get<tag>(mutable_global_cache_)).get()));
  } else {
    optional_callback =
        function(std::get<0>(tuples::get<tag>(mutable_global_cache_)));
  }
  if (optional_callback) {
    std::get<1>(tuples::get<tag>(mutable_global_cache_))
        .push_back(optional_callback.get());
    return false;
  } else {
    return true;
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function, typename... Args>
void MutableGlobalCache<Metavariables>::mutate(
    const std::tuple<Args...>& args) noexcept {
  (void)Parallel::charmxx::RegisterMutableGlobalCacheMutate<
      Metavariables, GlobalCacheTag, Function, Args...>::registrar;
  using tag = MutableCacheTag<
      tmpl::front<GlobalCache_detail::get_list_of_matching_mutable_tags<
          GlobalCacheTag, Metavariables>>>;

  // Get the callbacks to call, and at the same time
  // reset the list of stored callbacks to an empty vector.
  std::vector<CkCallback> callbacks_to_call{};
  std::swap(callbacks_to_call,
            std::get<1>(tuples::get<tag>(mutable_global_cache_)));

  // Do the mutate.
  std::apply(Function::apply,
             std::tuple_cat(std::forward_as_tuple(make_not_null(&std::get<0>(
                                tuples::get<tag>(mutable_global_cache_)))),
                            args));

  // Call the callbacks
  for (auto& callback : callbacks_to_call) {
    callback.send(nullptr);
  }
}

/// \ingroup ParallelGroup
/// A Charm++ chare that caches constant data once per Charm++ node.
///
/// `Metavariables` must define the following metavariables:
///   - `component_list`   typelist of ParallelComponents
///   - `const_global_cache_tags`   (possibly empty) typelist of tags of
///     constant data
///
/// The tag list for the items added to the GlobalCache is created by
/// combining the following tag lists:
///   - `Metavariables::const_global_cache_tags` which should contain only those
///     tags that cannot be added from the other tag lists below.
///   - `Component::const_global_cache_tags` for each `Component` in
///     `Metavariables::component_list` which should contain the tags needed by
///     any simple actions called on the Component, as well as tags need by the
///     `allocate_array` function of an array component.  The type alias may be
///     omitted for an empty list.
///   - `Action::const_global_cache_tags` for each `Action` in the
///     `phase_dependent_action_list` of each `Component` of
///     `Metavariables::component_list` which should contain the tags needed by
///     that  Action.  The type alias may be omitted for an empty list.
///
/// The tags in the `const_global_cache_tags` type lists are db::SimpleTag%s
/// that have a `using option_tags` type alias and a static function
/// `create_from_options` that are used to create the constant data from input
/// file options.
///
/// References to items in the GlobalCache are also added to the
/// db::DataBox of each `Component` in the `Metavariables::component_list` with
/// the same tag with which they were inserted into the GlobalCache.
template <typename Metavariables>
class GlobalCache : public CBase_GlobalCache<Metavariables> {
  using parallel_component_tag_list = tmpl::transform<
      typename Metavariables::component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;

 public:
  /// Access to the Metavariables template parameter
  using metavariables = Metavariables;
  /// Typelist of the ParallelComponents stored in the GlobalCache
  using component_list = typename Metavariables::component_list;

  // Constructor used only by the ActionTesting framework and other
  // non-charm++ tests that don't know about proxies.
  GlobalCache(tuples::tagged_tuple_from_typelist<
                  get_const_global_cache_tags<Metavariables>>
                  const_global_cache,
              MutableGlobalCache<Metavariables>* mutable_global_cache) noexcept
      : const_global_cache_(std::move(const_global_cache)),
        mutable_global_cache_(mutable_global_cache) {
    ASSERT(mutable_global_cache_ != nullptr,
           "GlobalCache: Do not construct with a nullptr!");
  }
  // Constructor used by Main and anything else that is charm++ aware.
  GlobalCache(tuples::tagged_tuple_from_typelist<
                  get_const_global_cache_tags<Metavariables>>
                  const_global_cache,
              CProxy_MutableGlobalCache<Metavariables>
                  mutable_global_cache_proxy) noexcept
      : const_global_cache_(std::move(const_global_cache)),
        mutable_global_cache_(nullptr),
        mutable_global_cache_proxy_(std::move(mutable_global_cache_proxy)) {}

  explicit GlobalCache(CkMigrateMessage* /*msg*/) {}
  ~GlobalCache() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        GlobalCache<Metavariables>,
        CkIndex_GlobalCache<Metavariables>>::registrar;
  }
  /// \cond
  GlobalCache(const GlobalCache&) = default;
  GlobalCache& operator=(const GlobalCache&) = default;
  GlobalCache(GlobalCache&&) = default;
  GlobalCache& operator=(GlobalCache&&) = default;
  /// \endcond

  /// Entry method to set the ParallelComponents (should only be called once)
  void set_parallel_components(
      tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&&
          parallel_components,
      const CkCallback& callback) noexcept;

  /// Returns whether the object referred to by `GlobalCacheTag`
  /// is ready to be accessed by a `get` call.
  ///
  /// `function` is an invokable that takes one argument: a const
  /// reference to the object referred to by the `GlobalCacheTag`.
  /// `function` returns a `boost::optional<CkCallBack>`.  If the
  /// `boost::optional` is valid, then `mutable_cache_item_is_ready` appends the
  /// `CkCallback` to the internal list of callbacks to be called on
  /// `mutate`, and then `mutable_cache_item_is_ready` returns false.  If the
  /// `boost::optional` is not valid, then `mutable_cache_item_is_ready` returns
  /// true.
  template <typename GlobalCacheTag, typename Function>
  bool mutable_cache_item_is_ready(const Function& function) noexcept;

  /// Mutates a non-const object.  Internally calls
  /// `Function::apply()`, where `Function` is a struct and
  /// `Function::apply()` is a user-defined static function that
  /// mutates the object.  `Function::apply()` takes as its first
  /// argument a gsl::not_null pointer to the object named by the
  /// GlobalCacheTag, and takes the contents of `args` as subsequent
  /// arguments.
  template <typename GlobalCacheTag, typename Function, typename... Args>
  void mutate(const std::tuple<Args...>& args) noexcept;

 private:
  // clang-tidy: false positive, redundant declaration
  template <typename GlobalCacheTag, typename MV>
  friend auto get(const GlobalCache<MV>& cache) noexcept  // NOLINT
      -> const GlobalCache_detail::type_for_get<GlobalCacheTag, MV>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      GlobalCache<MV>& cache) noexcept
      -> Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list, ParallelComponentTag>>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      const GlobalCache<MV>& cache) noexcept
      -> const Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list,
              ParallelComponentTag>>&;  // NOLINT

  tuples::tagged_tuple_from_typelist<get_const_global_cache_tags<Metavariables>>
      const_global_cache_{};
  tuples::tagged_tuple_from_typelist<parallel_component_tag_list>
      parallel_components_{};
  // We store both a pointer and a proxy to the MutableGlobalCache.
  // If the charm-aware constructor is used, then the pointer is set
  // to nullptr and the proxy is set.
  // If the non-charm-aware constructor is used, the the pointer is set
  // and the proxy is ignored.
  // The member functions that need the MutableGlobalCache should
  // use the pointer if it is not nullptr, otherwise use the proxy.
  MutableGlobalCache<Metavariables>* mutable_global_cache_{nullptr};
  CProxy_MutableGlobalCache<Metavariables> mutable_global_cache_proxy_{};
  bool parallel_components_have_been_set_{false};
};

template <typename Metavariables>
void GlobalCache<Metavariables>::set_parallel_components(
    tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&&
        parallel_components,
    const CkCallback& callback) noexcept {
  ASSERT(!parallel_components_have_been_set_,
         "Can only set the parallel_components once");
  parallel_components_ = std::move(parallel_components);
  parallel_components_have_been_set_ = true;
  this->contribute(callback);
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function>
bool GlobalCache<Metavariables>::mutable_cache_item_is_ready(
    const Function& function) noexcept {
  if (mutable_global_cache_ == nullptr) {
    return mutable_global_cache_proxy_.ckLocalBranch()
        ->template mutable_cache_item_is_ready<GlobalCacheTag>(function);
  } else {
    return mutable_global_cache_
        ->template mutable_cache_item_is_ready<GlobalCacheTag>(function);
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function, typename... Args>
void GlobalCache<Metavariables>::mutate(
    const std::tuple<Args...>& args) noexcept {
  (void)Parallel::charmxx::RegisterGlobalCacheMutate<
      Metavariables, GlobalCacheTag, Function, Args...>::registrar;
  if (mutable_global_cache_ == nullptr) {
    // charm-aware version: Mutate the variable on all PEs on this node.
    for (auto pe = CkNodeFirst(CkMyNode());
         pe < CkNodeFirst(CkMyNode()) + CkNodeSize(CkMyNode()); ++pe) {
      mutable_global_cache_proxy_[pe].template mutate<GlobalCacheTag, Function>(
          args);
    }
  } else {
    // version that bypasses proxies.  Just call the function.
    mutable_global_cache_->template mutate<GlobalCacheTag, Function>(args);
  }
}

// @{
/// \ingroup ParallelGroup
/// \brief Access the Charm++ proxy associated with a ParallelComponent
///
/// \requires ParallelComponentTag is a tag in component_list
///
/// \returns a Charm++ proxy that can be used to call an entry method on the
/// chare(s)
template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(GlobalCache<Metavariables>& cache) noexcept
    -> Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}

template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(const GlobalCache<Metavariables>& cache) noexcept
    -> const Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}
// @}

// @{
/// \ingroup ParallelGroup
/// \brief Access data in the cache
///
/// \requires GlobalCacheTag is a tag in tag_list
///
/// \returns a constant reference to an object in the cache
template <typename GlobalCacheTag, typename Metavariables>
auto get(const GlobalCache<Metavariables>& cache) noexcept
    -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>& {
  // We check if the tag is to be retrieved directly or via a base class
  using tag =
      tmpl::front<GlobalCache_detail::get_list_of_matching_tags<GlobalCacheTag,
                                                                Metavariables>>;
  static_assert(tmpl::size<GlobalCache_detail::get_list_of_matching_tags<
                        GlobalCacheTag, Metavariables>>::value == 1,
                "Found more than one tag matching the GlobalCacheTag "
                "requesting to be retrieved.");

  using tag_is_not_in_const_tags = std::is_same<
      tmpl::filter<get_const_global_cache_tags<Metavariables>,
                   std::is_base_of<tmpl::pin<GlobalCacheTag>, tmpl::_1>>,
      tmpl::list<>>;
  if constexpr (tag_is_not_in_const_tags::value) {
    // Tag is not in the const tags, so use MutableGlobalCache
    if (cache.mutable_global_cache_ == nullptr) {
      const auto& local_mutable_cache =
          *cache.mutable_global_cache_proxy_.ckLocalBranch();
      return local_mutable_cache.template get<GlobalCacheTag>();
    } else {
      return cache.mutable_global_cache_->template get<GlobalCacheTag>();
    }
  } else {
    // Tag is in the const tags, so use const_global_cache_
    if constexpr (tt::is_a_v<std::unique_ptr, typename tag::type>) {
      return *(tuples::get<tag>(cache.const_global_cache_).get());
    } else {
      return tuples::get<tag>(cache.const_global_cache_);
    }
  }
}

/// \ingroup ParallelGroup
/// \brief Returns whether an object is ready to be accessed by `get`.
///
/// \requires GlobalCacheTag is a tag in tag_list
/// \requires Function is an invokable that takes one argument: the object
/// referred to by the `GlobalCacheTag`.  `function` returns a
/// `boost::optional<CkCallBack>`.  If the `boost::optional` is valid,
/// then `mutable_cache_item_is_ready` appends the `CkCallback` to the internal
/// list of callbacks to be called on `mutate`, and then
/// `mutable_cache_item_is_ready` returns false. If the `boost::optional` is not
/// valid, then `mutable_cache_item_is_ready` returns true.
template <typename GlobalCacheTag, typename Function, typename Metavariables>
bool mutable_cache_item_is_ready(GlobalCache<Metavariables>& cache,
                                 const Function& function) noexcept {
  return cache.template mutable_cache_item_is_ready<GlobalCacheTag>(function);
}

/// \ingroup ParallelGroup
/// \brief Mutates non-const data in the cache, by calling `Function::apply()`
///
/// \requires `GlobalCacheTag` is a tag in tag_list
/// \requires `Function` is a struct with a static void `apply()`
/// function that mutates the object. `Function::apply()` takes as its
/// first argument a `gsl::not_null` pointer to the object named by
/// the `GlobalCacheTag`, and takes the contents of `args` as
/// subsequent arguments.
///
/// This is the version that takes a GlobalCache<Metavariables>. Used only
/// for tests.
template <typename GlobalCacheTag, typename Function, typename Metavariables,
          typename... Args>
void mutate(GlobalCache<Metavariables>& cache,
            const std::tuple<Args...>& args) noexcept {
  cache.template mutate<GlobalCacheTag, Function>(args);
}

/// \ingroup ParallelGroup
///
/// \brief Mutates non-const data in the cache, by calling `Function::apply()`
///
/// \requires `GlobalCacheTag` is a tag in tag_list.
/// \requires `Function` is a struct with a static void `apply()`
/// function that mutates the object. `Function::apply()` takes as its
/// first argument a `gsl::not_null` pointer to the object named by
/// the `GlobalCacheTag`, and takes the contents of `args` as
/// subsequent arguments.
///
/// This is the version that takes a charm++ proxy to the GlobalCache.
template <typename GlobalCacheTag, typename Function, typename Metavariables,
          typename... Args>
void mutate(CProxy_GlobalCache<Metavariables>& cache_proxy,
            const std::tuple<Args...>& args) noexcept {
  cache_proxy.template mutate<GlobalCacheTag, Function>(args);
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Parallel::GlobalCache` from the DataBox.
struct GlobalCache : db::BaseTag {};

template <class Metavariables>
struct GlobalCacheImpl : GlobalCache, db::SimpleTag {
  using type = const Parallel::GlobalCache<Metavariables>*;
  static std::string name() noexcept { return "GlobalCache"; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag used to retrieve data from the `Parallel::GlobalCache`. This is the
/// recommended way for compute tags to retrieve data out of the global cache.
template <class CacheTag>
struct FromGlobalCache : CacheTag, db::ComputeTag {
  static std::string name() noexcept {
    return "FromGlobalCache(" + pretty_type::short_name<CacheTag>() + ")";
  }
  template <class Metavariables>
  static const GlobalCache_detail::type_for_get<CacheTag, Metavariables>&
  function(const Parallel::GlobalCache<Metavariables>* const& cache) {
    return Parallel::get<CacheTag>(*cache);
  }
  using argument_tags = tmpl::list<GlobalCache>;
};
}  // namespace Tags
}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/GlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
