// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Fills a blaze::CompressedMatrix that does a TensorYlm filter
 * operation.
 *
 * Assumes that $T^{\tilde A}_{\ell' m'}$ is stored in a
 * Tensor<DataVector>.  Multiplying the resulting
 * blaze::CompressedMatrix by the Tensor<DataVector> is equivalent to
 * evaluating the right-hand side of Eq.~(\ref{eq:Filter}).
 *
 * Assumes that the components of the Tensor<DataVector> are stored
 * contiguous in memory, in order of the storage_index of the Tensor.
 * Also assumes that the stride is unity.  In this way, we are able to
 * construct a blaze::CustomVector pointing to the beginning of the
 * first element of $T^{\tilde A}_{\ell' m'}$ and multiply that by the
 * blaze::CompressedMatrix we compute here, and get the result.
 *
 * This memory layout here is different than in SpEC.  In SpEC, each
 * tensor component is stored in separately-allocated memory, so this
 * equivalent function in SpEC fills $N^2$ sparse matrices, where $N$
 * is the number of independent components of the Tensor. The
 * advantage of the SpEC method is that each sparse matrix is smaller,
 * so sorting elements into the correct order while constructing each
 * sparse matrix is faster (sorting is > linear in the number of
 * matrix elements).  The disadvantage of the SpEC method is that
 * evaluating the filter for a single Tensor involves $N^2$
 * matrix-vector multiplications, whereas here evaluating the filter
 * involves only one matrix-vector multiplication, which should have
 * more efficient memory access.  It is not clear which method is
 * faster overall without profiling.
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
 * version of the Heaviside function described in the TensorYlm
 * namespace documentation, with $\sigma$ equal to half_power and
 * $\ell_{\mathrm{cut}}^+$ equal to $\ell_{\rm max}$ minus
 * number_of_ell_modes_to_kill.
 *
 * For rank-1 tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}} &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 *  \nonumber \\
 *  &- (-1)^{m+m''} (-1)^{\delta(\tilde{D},\mathbf{e}_y)}
 *  \delta(\ell,\ell'')
 *  \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{max}}+1}
 *  \frac{2\ell'+1}{2} g(\ell') \nonumber \\
 *  &\times \sum_{j,p,m'} k_j(\tilde{D})k_p(\tilde{A})
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&1\cr
 *      -m''&m'&m_j(\tilde{D})
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&1\cr
 *      -m&m'&m_p(\tilde{A})
 *    \end{array}\right),
 * \end{align}
 * where the 6-element "matrices"
 * in parentheses are Wigner 3-J symbols, and where
 * \begin{align}
 *  g(\ell') &=
 *  \left\{\begin{array}{lr}
 *      1-f(\ell') & \ell' \leq \ell_{\mathrm{cut}}^+,\\
 *      1 & \ell' > \ell_{\mathrm{cut}}^+.
 *  \end{array}\right.
 * \end{align}
 *
 * For second-rank tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 * \nonumber \\
 * &-
 *  \frac{1}{4}
 * (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 * (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *  \delta_{\ell \ell''}
 * \nonumber \\
 * &\times
 * \sum_{\ell'=\ell_{\mathrm{cut}}^-}^{\ell_{\mathrm{max}}+2}
 * (2\ell'+1) g(\ell')
 * \sum_{u,v,p,q,\bar{\ell},\tilde{m},\bar{m},\mpr}
 * (2 \bar{\ell}+1)
 * k_u(\tilde{D}_1) k_v(\tilde{D}_2)
 * k_p(\tilde{A}_1) k_q(\tilde{A}_2)
 * \nonumber \\
 * &\times
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&\bar{\ell}\cr
 *      -m&m'&\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *      m_p(\tilde{A}_1)&{m_q(\tilde{A}_2)&-\bar{m}
 *    \end{array}\right)
 * \nonumber \\
 * &\times
 *    \left(\begin{array}{rrr}
 *     \ell'&\ell&\bar{\ell}\cr
 *      -m'&m''&\tilde{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *      m_u(\tilde{D}_1)&m_v(\tilde{D}_2)&\tilde{m}
 *    \end{array}\right).
 *  \label{eq:RankTwoTransformWithCut}
 * \end{align}
 *
 * For a symmetric 2nd-rank spatial tensor, we store only half of the
 * off-diagonal components, and we sum over only the components we have
 * stored. In this case, we write
 * \begin{align}
 *   T^{\tilde A}_{\ell m}{}^{\hbox{filtered}}
 *   &= \sum_{\ell'' m'' \tilde{D_1}\geq\tilde{D_2}}
 *                  \breve{F}_{\ell m\tilde{D}}^{\ell'' m'' \tilde{A}}
 *                  T^{\tilde D}_{\ell'' m''},
 * \end{align}
 * where the sum over tensor components goes over only half the off-diagonal
 * components, and where
 * \begin{align}
 *  \breve{F}_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}\nonumber\\
 *  &-\sum_{\tilde{D_1}\geq \tilde{D_2}}
 *  (\hbox{Last term in Eq.~(\ref{eq:RankTwoTransformWithCut})})
 *  (1+(-1)^{\bar{\ell}})
 *  \frac{2-\delta(\tilde{D_1},\tilde{D_2})}{2}.
 * \end{align}
 *
 * For third-rank tensors, the expression for
 * $F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}$ is
 * \begin{align}
 *  F_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}
 *  \nonumber \\
 * &-
 *  \frac{1}{8}
 *  (-1)^{\delta(\tilde{D}_1,\mathbf{e}_y)}
 *  (-1)^{\delta(\tilde{D}_2,\mathbf{e}_y)}
 *  (-1)^{\delta(\tilde{D}_3,\mathbf{e}_y)}
 *  \delta_{\ell \ell''}
 *  \nonumber \\
 *  &\times
 *  \sum_{\ell'=\ell_{\mathrm{cut}}+1}^{\ell_{\mathrm{max}}+3}
 *  (2\ell'+1)
 *  \sum_{u,v,w,p,q,r,\tilde{m},\bar{\ell},\bar{m},
 *  \hat{\ell},\hat{m},\check{m},m'}
 *  (2 \bar{\ell}+1)
 *  (2 \hat{\ell}+1)
 *  \nonumber \\
 *  &\qquad\times
 *  k_u(\tilde{D}_2) k_v(\tilde{D}_3) k_w(\tilde{D}_1)
 *  k_p(\tilde{A}_2) k_q(\tilde{A}_3) k_r(\tilde{A}_1)
 *  (-1)^{\tilde{m}-\bar{m}}
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *     m_p(\tilde{A}_2)&m_q(\tilde{A}_3)&-\bar{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&1&\bar{\ell}\cr
 *     m_u(\tilde{D}_2)&m_v(\tilde{D}_3)&\tilde{m}
 *    \end{array}\right)
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     \ell&\ell'&\hat{\ell}\cr
 *     -m&m'&\check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     \ell'&\ell&\hat{\ell}\cr
 *     -m'&m''&\hat{m}
 *    \end{array}\right)
 *  \nonumber \\
 *  &\qquad\times
 *    \left(\begin{array}{rrr}
 *     1&\bar{\ell}&\hat{\ell}\cr
 *     m_r(\tilde{A}_1)&\bar{m}&-\check{m}
 *    \end{array}\right)
 *    \left(\begin{array}{rrr}
 *     1&\bar{\ell}&\hat{\ell}\cr
 *     m_w(\tilde{D}_1)&-\tilde{m}&\hat{m}.
 *    \end{array}\right)
 *    \label{eq:RankThreeTransformWithCut}
 * \end{align}
 *
 * For rank-3 tensors symmetric on the last two indices, we do
 * the same thing as rank-2 symmetric tensors and write
 * \begin{align}
 *  \breve{F}_{l m \tilde{D}}^{\ell'' m''\tilde{A}}
 *  &=
 *  \delta(\tilde{D},\tilde{A})\delta_{\ell \ell''}\delta_{m m''}\nonumber\\
 *  &-\sum_{\tilde{D_2}\geq \tilde{D_3}}
 *  (\hbox{Last term in Eq.~(\ref{eq:RankThreeTransformWithCut})})
 *  (1+(-1)^{\bar{\ell}})
 *  \frac{2-\delta(\tilde{D_2},\tilde{D_3})}{2}.
 * \end{align}
 *
 * \tparam TensorStructure A Tensor_detail::Structure
 *
 * \param matrix The CompressedMatrix to fill
 * \param ell_max The maximum ylm ell value.
 * \param number_of_ell_modes_to_kill How many top ell modes to set to zero.
 * \param half_power The half power $\sigma$ for more complicated filtering.
 */
template <typename TensorStructure>
void FillFilter(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    size_t ell_max, size_t number_of_ell_modes_to_kill,
    std::optional<size_t> half_power);

};  // namespace ylm::TensorYlm
