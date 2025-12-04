// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

class DataVector;

/*!
 * \brief Copies spacetime variables into their spatial pieces.
 *
 * For example, if one of the spacetime variables is the metric
 * $g_{ab}$, then the corresponding spatial variables are a spatial
 * scalar $g_{00}$, a spatial vector $g_{i0}$, and a spatial symmetric
 * 2-tensor $g_{ij}$.
 *
 * The arguments must already be allocated to their correct sizes; no
 * memory allocation is done.
 *
 * \tparam SpacetimeVars A tmpl::list of tags for the spacetime variables.
 * \tparam SpatialVars A tmpl::list of tags for the spatial pieces.
 * \param spatial_vars Points to a Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
template <typename SpacetimeVars, typename SpatialVars>
void break_spacetime_vars_into_spatial_pieces(
    gsl::not_null<Variables<SpatialVars>*> spatial_vars,
    const Variables<SpacetimeVars>& spacetime_vars);

/*!
 * \brief Copies spatial pieces into the corresponding spacetime variables.
 *
 * This is the inverse of break_spacetime_vars_into_spatial_pieces.
 *
 * \tparam SpacetimeVars A tmpl::list of tags for the spacetime variables.
 * \tparam SpatialVars A tmpl::list of tags for the spatial pieces.
 * \param spatial_vars A Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
template <typename SpacetimeVars, typename SpatialVars>
void assemble_spacetime_vars_from_spatial_pieces(
    gsl::not_null<Variables<SpacetimeVars>*> spacetime_vars,
    const Variables<SpatialVars>& spatial_vars);
