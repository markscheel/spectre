// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/*!
 * \brief Converts between scalar-Ylm and tensor-Ylm basis.
 *
 * \details We expand a tensor of arbitrary rank
 * in terms of Cartesian components as follows:
 * \begin{align}
 * {\mathbf T} &= \sum_{\ell,m,\tilde{A}} {}_0 Y_{\ell m}
 *                T^{\tilde A}_{\ell m}\mathbf{e}_{\tilde A},
 * \end{align}
 * where ${}_0 Y_{\ell m}$ are the usual scalar spherical harmonics,
 * $T^{\tilde A}_{\ell m}$ are complex-valued expansion coefficients,
 * $\tilde A$ is a multi-index
 * $\tilde{A}=(\tilde{a}_1,\tilde{a}_2,\tilde{a}_3,\ldots)$, where each
 * of the $\tilde{a}_i$ take on values from 0 to 2, referring to
 * either $\mathbf{e}_x$, $\mathbf{e}_y$, or $\mathbf{e}_z$.
 * The sum over $\tilde A$ goes over all $\tilde A$ of the same rank.
 *
 * Similarly, we can expand the same tensor in terms of a complex tetrad:
 * \begin{align}
 * {\mathbf T} &= \sum_{\ell,m,A} {}_{s(\!A\!)}Y_{\ell m}
 *                T^A_{\ell m} \mathbf{e}_A,
 * \end{align}
 * where ${}_sY_{\ell m}$ are the spin-weighted harmonics,
 * $A$ is a multi-index $A=(a_1,a_2,a_3,\ldots)$, where each of the
 * $a_i$ refer to either $\mathbf{l}$,$\mathbf{m}$, or $\mathbf{\bar{m}}$,
 * and $s(\!A\!)$ is a sum of terms where each
 * $\mathbf{l}$ adds the value zero, each $\mathbf{m}$
 * adds the value -1, and each $\mathbf{\bar{m}}$ adds the value +1.
 * (These values are the spin weights of the complex conjugates
 * of the basis vectors.)
 *
 * We can define the following transformations between the expansion
 * coefficients $T^A_{\ell m}$ and $T^{\tilde A}_{\ell m}$:
 * \begin{align}
 * T^{\tilde B}_{\ell' m'} &=
 * \sum_{\ell,m,A} C_{\ell' m' A}^{\ell m\tilde{B}} T^A_{\ell m},\\
 * T^{B}_{\ell' m'} &= \sum_{\ell m \tilde{A}}
 *                  C_{\ell' m'\tilde{A}}^{\ell mB}
 *                  T^{\tilde A}_{\ell m},
 * \end{align}
 * where the above transformations define the coefficients
 * $C_{\ell' m' A}^{\ell m\tilde{B}}$ and $C_{\ell' m'\tilde{A}}^{\ell mB}$.
 * Analytic expressions for these coefficients are derived in Klinger
 * and Scheel (in prep) and will be used here to compute the coefficients.
 *
 * For real-valued tensors, the coefficients $T^A_{\ell m}$ and
 * $T^{\tilde A}_{\ell m}$ for negative $m$ can be computed from the
 * complex conjugates of the same coefficients with positive $m$.  Therefore
 * we store and loop over only nonnegative $m$. This means we can write
 * \begin{align}
 * T^{\tilde B}_{\ell m} &= \sum_{A, \ell',m'\geq 0}
 *                  \left[
 *              C_{\ell m A}^{\ell' m'\tilde{B}} T^A_{\ell' m'}
 *               +{\hat C}_{\ell m A}^{\ell' m'\tilde{B}} T^A_{\ell' m'}{}^\star
 *                \right]\label{eq:S2C},\\
 * T^{B}_{\ell m} &= \sum_{\tilde{A},\ell',m'\geq 0}
 *                  \left[
 *                  C_{\ell m\tilde{A}}^{\ell' m' B}
 *                  T^{\tilde A}_{\ell' m'}
 *                  +{\hat C}_{\ell m\tilde{A}}^{\ell' m' B}
 *                  T^{\tilde A}_{\ell' m'}{}^\star
 *                  \right]\label{eq:C2S},
 * \end{align}
 * where
 * \begin{align}
 * {\hat C}_{\ell m A}^{\ell' m'\tilde{B}} &=
 *                 (1-\delta_{0m'})C_{\ell m A^\star}^{\ell' -m'\tilde{B}}
 *                 (-1)^{S(A)+m'},\\
 * {\hat C}_{\ell m\tilde{A}}^{\ell' m' B} &=
 *                (1-\delta_{0m'})C_{\ell m \tilde{A}}^{\ell' -m'B}
 *                 (-1)^{m'}.
 * \end{align}
 *
 * The functions FillCartToSphere and FillSphereToCart fill
 * SparseMatrixCollections that encode Eqs.(\ref{eq:C2S}) and
 * (\ref{eq:S2C}).
 *
 * ## Filtering
 *
 * Consider starting with coefficients $T^{\tilde A}_{\ell' m'}$,
 * transforming to spin-weighted harmonics, applying a filter to
 * the spin-weighted harmonic coefficients, and transforming back.
 * We can write this entire operation as
 * \begin{align}
 *   F_{\ell m \tilde{D}}^{\ell'' m''\tilde{A}} &=
 *   \sum_{\ell'=0}^{\ell_{\mathrm{cut}}^--1} C^{\ell' m' \tilde{A}}_{\ell m B}
 *   C^{\ell'' m'' B}_{\ell'  m' \tilde{D}}
 * + \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{cut}}^+}
 *   C^{\ell' m' \tilde{A}}_{\ell m B}
 *   C^{\ell'' m'' B}_{\ell'  m' \tilde{D}} f(\ell'),
 * \end{align}
 * where $f(\ell')$ is a filter function,
 * $\ell_{\mathrm{cut}}^{-}$ is the smallest $\ell$ mode that is filtered,
 * and
 * $\ell_{\mathrm{cut}}^{+}$ is the largest $\ell$ mode that is retained.
 *
 * There are two cases we care about.
 * The first is simple "heaviside filtering", where
 * $\ell_{\mathrm{cut}}^{-} = \ell_{\mathrm{cut}}^{+}$ and $f(\ell')=1$.
 *
 * The second case is
 * \begin{align}
 *  f(\ell') &= \exp{-\alpha
 * \left(\frac{\ell'}{\ell_{\mathrm{cut}}^++1}\right)^{2 \sigma}},\\
 * \ell_{\mathrm{cut}}^- &= \mathrm{ceil}\left[(\ell_{\mathrm{cut}}^++1)
 *    \left(\frac{\epsilon}{\alpha}\right)^{1/(2\sigma)}\right],
 * \end{align}
 * where $\alpha$ is a parameter we choose to be 36, $\sigma$ is an integer
 * parameter typically between 28 and 32, and
 * $\epsilon$ is machine roundoff.
 * Note that the filter remains smooth at $\lpr = \ell_{\mathrm{cut}}^-$ and
 * it reduces to the simple filter as $\sigma \to \infty$.
 *
 * As with the other cases, we sum over only nonnegative $m$, so we can
 * write the action of the filter as
 * \begin{align}
 * T^{\tilde B}_{\ell m} &= \sum_{{\tilde A}, \ell',m'\geq 0}
 *                  \left[
 *              F_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *                T^{\tilde A}_{\ell' m'}
 *               +{\hat F}_{\ell m {\tilde A}}^{\ell' m'\tilde{B}}
 *               T^{\tilde A}_{\ell' m'}{}^\star
 *                \right]\label{eq:Filter},
 * \end{align}
 * where
 * \begin{align}
 * {\hat F}_{\ell m\tilde{A}}^{\ell' m' \tilde{B}} &=
 *                (1-\delta_{0m'})F_{\ell m \tilde{A}}^{\ell' -m' \tilde{B}}
 *                 (-1)^{m'}.
 * \end{align}
 */
namespace ylm::TensorYlm {

/*!
 * \brief Fills a SparseMatrixCollection that transforms from
 * Cartesian to Spherical.
 *
 * Assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * SparseMatrixCollection by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:C2S}).
 */
void FillCartToSphere(gsl::not_null<SparseMatrixCollection*> matrices);

/*!
 * \brief Fills a SparseMatrixCollection that transforms from
 * Spherical to Cartesian.
 *
 * Assumes that $T^A_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * SparseMatrixCollection by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:S2C}).
 */
void FillSphereToCart(gsl::not_null<SparseMatrixCollection*> matrices);

/*!
 * \brief Fills a SparseMatrixCollection that does a TensorYlm filter operation.
 *
 * Assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * SparseMatrixCollection by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:Filter}).
 *
 * If half_power is std::nullopt, implements a Heaviside filter:
 * Given src as a tensor of scalar-Ylm coefficients of Cartesian
 * components, transforms to spin-weighted harmonic coefficients,
 * zeroes the top number_of_ell_modes_to_kill ell modes, and transforms
 * back.  This could be implemented in terms of SpinWeightedToCartesian
 * and CartesianToSpinWeighted, but it is often more efficient to implement
 * this as its own function by simplifying the expressions analytically.
 *
 * If half_power is not std::nullopt, then the filter is the smooth
 * version of the Heaviside function described above, with $\sigma$
 * equal to half_power and $\ell_{\mathrm{cut}}^+$ equal to
 * $\ell_{\rm max}$ minus number_of_ell_modes_to_kill.
 */
void FillFilter(gsl::not_null<SparseMatrixCollection*> matrices,
                const size_t number_of_ell_modes_to_kill,
                const std::optional<size_t> half_power);

};  // namespace ylm::TensorYlm
