// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for interfacing with the parallelization framework

#pragma once

#include <charm++.h>

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements.
 */
template<typename DistributedObject>
int number_of_procs(const DistributedObject& distributed_object) {
  return distributed_object.number_of_procs();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my processing element.
 */
template<typename DistributedObject>
int my_proc(const DistributedObject& distributed_object) {
  return distributed_object.my_proc();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of nodes.
 */
template<typename DistributedObject>
int number_of_nodes(const DistributedObject& distributed_object) {
  return distributed_object.number_of_nodes();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my node.
 */
template <typename DistributedObject>
int my_node(const DistributedObject& distributed_object) {
  return distributed_object.my_node();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements on the given node.
 */
template <typename DistributedObject>
int procs_on_node(const int node_index,
                  const DistributedObject& distributed_object) {
  return distributed_object.procs_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node()) - 1.
 */
template <typename DistributedObject>
int my_local_rank(const DistributedObject& distributed_object) {
  return distributed_object.my_local_rank();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of first processing element on the given node.
 */
template <typename DistributedObject>
int first_proc_on_node(const int node_index,
                       const DistributedObject& distributed_object) {
  return distributed_object.first_proc_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of the node for the given processing element.
 */
template <typename DistributedObject>
int node_of(const int proc_index, const DistributedObject& distributed_object) {
  return distributed_object.node_of(proc_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index for the given processing element on its node.
 */
template <typename DistributedObject>
int local_rank_of(const int proc_index,
                  const DistributedObject& distributed_object) {
  return distributed_object.local_rank_of(proc_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The current wall time in seconds
 */
inline double wall_time() { return CmiWallTimer(); }
}  // namespace Parallel
