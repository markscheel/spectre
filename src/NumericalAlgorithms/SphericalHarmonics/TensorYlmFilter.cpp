// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <complex>
#include <optional>

#include "DataStructures/SparseMatrixFiller.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/WignerThreeJ.hpp"

namespace ylm::TensorYlm {

namespace {
// Inner loops of the rank-1 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_one(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const int mdest, const int msrc,
                     const std::complex<double>& coefjp, WignerThreeJ& threej_j,
                     WignerThreeJ& threej_p) {
  auto add_element = [&filler, &iter_src, &iter_dest, dest_comp_index,
                      src_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  for (size_t ell = threej_j.l1_min(); ell <= threej_j.l1_max(); ++ell) {
    if (ell <= ell_max and static_cast<int>(ell) >= mdest) {
      const std::complex<double> correction =
          coefjp * threej_j(ell) * threej_p(ell);
      if (msrc > 0) {
        // Main term.
        // ReRe
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(correction.real());

        // ReIm
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(-correction.imag());

        // ImIm
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(correction.real());

        // ImRe
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(correction.imag());
      } else {
        // We are multiplying by Tlmsrc but we should be
        // multiplying by (Tlmsrc)^star (-1)^msrc
        const double sign = (msrc % 2 == 0 ? 1.0 : -1.0);
        // ReRe
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(sign * correction.real());

        // ReIm
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(sign * correction.imag());

        // ImRe
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(sign * correction.imag());

        // ImIm
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(-sign * correction.real());
      }
    }
  }
}

// Inner loops of the rank-2 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_two(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t lprime, const int mprime, const size_t lbar,
                     const int mbar, const int mtilde, const double coeflbar,
                     const double coeflprime, WignerThreeJ& threej_lbar,
                     WignerThreeJ& threej_ltilde, const std::vector<int>& mbars,
                     const std::vector<int>& mtildes,
                     const std::array<helpers::BasisVector, 3>& src_bvs,
                     const std::array<helpers::BasisVector, 3>& dest_bvs,
                     const double SymmFactor, const double sign_y,
                     std::vector<WignerThreeJ>& threej_pqs,
                     std::vector<WignerThreeJ>& threej_uvs) {
  auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                      dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mbar_indx = 0;
  for (int p = -1; p <= 1; p += 2) {
    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
      if (mbar == mbars[mbar_indx] and mprime + mbar >= 0) {
        // The 2nd clause in the above if-statement:
        // Fill only nonnegative m_dest, since that is all we store.
        const int m_dest = mprime + mbar;
        size_t mtilde_indx = 0;
        for (int u = -1; u <= 1; u += 2) {
          for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
            if (mtilde == mtildes[mtilde_indx]) {
              std::complex<double> k_coefs = helpers::bv_to_k(src_bvs[0], u) *
                                             helpers::bv_to_k(dest_bvs[0], p) *
                                             helpers::bv_to_k(src_bvs[1], v) *
                                             helpers::bv_to_k(dest_bvs[1], q);
              const int m_src = mprime - mtilde;
              std::complex<double> coef3j =
                  -coeflprime * coeflbar * k_coefs * sign_y;
              for (size_t l_dest = std::max(
                       std::max(
                           static_cast<size_t>(abs(static_cast<int>(lprime) -
                                                   static_cast<int>(lbar))),
                           static_cast<size_t>(abs(mprime + mbar))),
                       static_cast<size_t>(
                           std::max(abs(mtilde - mprime), m_dest)));
                   l_dest <= std::min(lprime + lbar, ell_max); ++l_dest) {
                const double sign_lbar =
                    ((lprime + l_dest + lbar) % 2 == 0 ? 1.0 : -1.0);
                // Corrects for permutation of columns in
                // threej_lba
                const std::complex<double> correction =
                    coef3j * sign_lbar * threej_pqs[mbar_indx](lbar) *
                    threej_uvs[mtilde_indx](lbar) * threej_lbar(l_dest) *
                    threej_ltilde(l_dest) * SymmFactor;
                if (m_src > 0) {
                  // Main term.
                  // ReRe
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(correction.real());

                  // ReIm
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::b);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(-correction.imag());

                  // ImIm
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::b);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(correction.real());

                  // ImRe
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(correction.imag());
                } else {
                  const double sign = (m_src % 2 == 0 ? 1.0 : -1.0);
                  // ReRe
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(sign * correction.real());

                  // ReIm
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::b);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(sign * correction.imag());

                  // ImRe
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(sign * correction.imag());

                  // ImIm
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::b);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(-sign * correction.real());
                }
              }
            }
          }
        }
      }
    }
  }
};

// Inner loops of the rank-3 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
template <typename Symm>
void inner_loops_three(
    SparseMatrixFiller& filler, SpherepackIterator& iter_src,
    SpherepackIterator& iter_dest, const size_t src_comp_index,
    const size_t dest_comp_index, const size_t ell_max, const size_t lprime,
    const double coeflprime, const int mprime, const size_t lhat,
    const int mhat, WignerThreeJ& threej_mhat, const int mcheck,
    WignerThreeJ& threej_mcheck, const size_t mbar_indx, const int p,
    const int q, const int r, const int mr, const std::vector<int>& mbars,
    const std::vector<int>& mtildes,
    const std::array<helpers::BasisVector, 3>& src_bvs,
    const std::array<helpers::BasisVector, 3>& dest_bvs,
    const size_t src_multiplicity, std::vector<WignerThreeJ>& threej_pqs,
    std::vector<WignerThreeJ>& threej_uvs, std::vector<WignerThreeJ>& threej_ws,
    std::vector<WignerThreeJ>& threej_rs, const double sign_coef3j) {
  auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                      dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  const int m_dest = mprime + mcheck;
  size_t mtilde_indx = 0;
  for (int u = -1; u <= 1; u += 2) {
    for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
      for (size_t lbar = static_cast<size_t>(
               std::max(abs(mbars[mbar_indx]), abs(mtildes[mtilde_indx])));
           lbar <= 2; ++lbar) {
        for (int w = -1; w <= 1; w += 2) {
          const int mw = helpers::bv_to_m(src_bvs[0], w);
          if (mtildes[mtilde_indx] - mw == mhat and
              lhat >= static_cast<size_t>(std::max(
                          abs(static_cast<int>(lbar) - 1),
                          std::max(abs(mr + mbars[mbar_indx]),
                                   abs(mw - mtildes[mtilde_indx])))) and
              lhat <= lbar + 1) {
            const int m_src = mprime - mhat;
            const double SymmFactor = [src_multiplicity, lbar]() {
              if constexpr (std::is_same_v<Symmetry<0, 1, 2>, Symm>) {
                // "abc" symmetry
                (void)src_multiplicity;
                return 1.0;
              } else {
                // any other symmetry
                return (src_multiplicity / 2.0) * (lbar % 2 == 0 ? 2.0 : 0.0);
              }
            }();
            const double sign_mtilde =
                ((mtildes[mtilde_indx] - mbars[mbar_indx]) % 2 == 0 ? 1.0
                                                                    : -1.0);
            const double coeflhat = 0.5 * (2 * lhat + 1);
            const double coeflbar = 0.5 * (2 * lbar + 1);
            std::complex<double> k_coefs = helpers::bv_to_k(src_bvs[1], u) *
                                           helpers::bv_to_k(dest_bvs[1], p) *
                                           helpers::bv_to_k(src_bvs[2], v) *
                                           helpers::bv_to_k(dest_bvs[2], q) *
                                           helpers::bv_to_k(dest_bvs[0], r) *
                                           helpers::bv_to_k(src_bvs[0], w);
            const std::complex<double> coef3j =
                -coeflprime * coeflbar * coeflhat * k_coefs * sign_coef3j;
            for (size_t l_dest = static_cast<size_t>(std::max(
                     abs(static_cast<int>(lprime) - static_cast<int>(lhat)),
                     std::max(abs(mhat - mprime), abs(mcheck + mprime))));
                 l_dest <= lprime + lhat; ++l_dest) {
              if (l_dest <= ell_max and l_dest >= m_dest) {
                const double sign_lhat =
                    ((lprime + l_dest + lhat) % 2 == 0 ? 1.0 : -1.0);
                // The division inside the index of the following
                // quantities is integer division.  Note that
                // q,v,r,w,v are always odd.
                const double threej_pq =
                    threej_pqs[static_cast<size_t>((q + 1) / 2 + p + 1)](lbar);
                const double threej_uv =
                    threej_uvs[static_cast<size_t>((v + 1) / 2 + u + 1)](lbar);
                const double threej_r = threej_rs[static_cast<size_t>(
                    lbar + (r + 1) * 3 / 2 + 6 * ((q + 1) / 2 + p + 1))](lhat);
                const double threej_w = threej_ws[static_cast<size_t>(
                    lbar + (w + 1) * 3 / 2 + 6 * ((v + 1) / 2 + u + 1))](lhat);
                const std::complex<double> correction =
                    coef3j * threej_pq * threej_uv * threej_r * threej_w *
                    threej_mhat(l_dest) * threej_mcheck(l_dest) * sign_lhat *
                    sign_mtilde * SymmFactor;
                if (m_src > 0) {
                  // Main term.
                  // ReRe
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(correction.real());

                  // ReIm
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::b);
                  add_element(-correction.imag());

                  // ImIm
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(correction.real());

                  // ImRe
                  iter_src.set(l_dest, static_cast<size_t>(m_src),
                               SpherepackIterator::CoefficientArray::a);
                  add_element(correction.imag());
                } else {
                  const double sign = (m_src % 2 == 0 ? 1.0 : -1.0);
                  // ReRe
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::a);
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::a);
                  add_element(sign * correction.real());

                  // ReIm
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::b);
                  add_element(sign * correction.imag());

                  // ImIm
                  iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                SpherepackIterator::CoefficientArray::b);
                  add_element(-sign * correction.real());

                  // ImRe
                  iter_src.set(l_dest, static_cast<size_t>(-m_src),
                               SpherepackIterator::CoefficientArray::a);
                  add_element(sign * correction.imag());
                }
              }
            }
          }
        }
      }
    }
  }
};
}  // namespace

template <typename TensorStructure>
void FillFilter(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    const size_t ell_max, const size_t number_of_ell_modes_to_kill,
    const std::optional<size_t> half_power) {
  static constexpr size_t num_independent_components = TensorStructure::size();
  static constexpr size_t rank = TensorStructure::rank();
  static constexpr auto tensor_index_list =
      TensorStructure::storage_to_tensor_index();

  static_assert(rank > 0 and rank < 4, "Implemented only for ranks 1,2,3");

  const size_t lcutplus =
      ell_max - static_cast<int>(number_of_ell_modes_to_kill);
  const size_t lcutminus =
      half_power.has_value()
          ? size_t(std::ceil((lcutplus + 1) *
                             pow(std::numeric_limits<double>::epsilon() / 36.0,
                                 1.0 / (2.0 * double(half_power.value())))))
          : lcutplus + 1;

  SpherepackIterator iter_src(ell_max, ell_max, 1, zero_m_is_real = true);
  SpherepackIterator iter_dest(ell_max, ell_max, 1, zero_m_is_real = true);
  SparseMatrixFiller filler(square(num_independent_components) *
                                iter_src.spherepack_array_size() *
                                iter_dest.spherepack_array_size(),
                            true);

  for (size_t dest_comp_index = 0; dest_comp_index < num_independent_components;
       dest_comp_index++) {
    const auto dest_indices = tensor_index_list[dest_comp_index];
    const auto dest_bvs = helpers::to_cart_basis_vector(dest_indices);

    std::vector<WignerThreeJ> threej_pqs;
    std::vector<int> mbars;
    if constexpr (rank > 1) {
      threej_pqs.reserve(4);
      mbars.reserve(4);
      for (int p = -1; p <= 1; p += 2) {
        for (int q = -1; q <= 1; q += 2) {
          mbars.push_back(helpers::bv_to_m(dest_bvs[rank - 2], p) +
                          helpers::bv_to_m(dest_bvs[rank - 1], q));
          threej_pqs.emplace_back(1, helpers::bv_to_m(dest_bvs[rank - 2], p), 1,
                                  helpers::bv_to_m(dest_bvs[rank - 1], q));
        }
      }
    } else {
      // For rank 1 we don't need threej_pqs and mbars but we need to declare
      // them anyway for scoping; they will just be unused.
      (void)threej_pqs;
      (void)mbars;
    }
    for (size_t src_comp_index = 0; src_comp_index < num_independent_components;
         src_comp_index++) {
      const auto src_indices = tensor_index_list[src_comp_index];
      const auto src_bvs = helpers::to_cart_basis_vector(src_indices);
      const size_t src_multiplicity =
          TensorStructure::multiplicity(src_comp_index);

      std::vector<WignerThreeJ> threej_uvs;
      std::vector<int> mtildes;
      if constexpr (rank > 1) {
        threej_uvs.reserve(4);
        mtildes.reserve(4);
        for (int u = -1; u <= 1; u += 2) {
          for (int v = -1; v <= 1; v += 2) {
            mtildes.push_back(-(helpers::bv_to_m(src_bvs[rank - 2], u) +
                                helpers::bv_to_m(src_bvs[rank - 1], v)));
            threej_uvs.emplace_back(1, helpers::bv_to_m(src_bvs[rank - 2], u),
                                    1, helpers::bv_to_m(src_bvs[rank - 1], v));
          }
        }
      } else {
        // For rank 1 we don't need threej_uvs and mtildes but we need to
        // declare them anyway for scoping; they will just be unused.
        (void)threej_uvs;
        (void)mtildes;
      }

      std::vector<WignerThreeJ> threej_rs;
      std::vector<WignerThreeJ> threej_ws;
      if constexpr (rank > 2) {
        threej_rs.reserve(24);
        for (int mbar : mbars) {
          for (int r = -1; r <= 1; r += 2) {
            const int mr = helpers::bv_to_m(dest_bvs[0], r);
            for (int lbar = 0; lbar <= 2; ++lbar) {
              threej_rs.emplace_back(1, mr, lbar, mbar);
            }
          }
        }
        threej_ws.reserve(24);
        for (int mtilde : mtildes) {
          for (int w = -1; w <= 1; w += 2) {
            const int mw = helpers::bv_to_m(src_bvs[0], w);
            for (int lbar = 0; lbar <= 2; ++lbar) {
              threej_ws.emplace_back(1, mw, lbar, -mtilde);
            }
          }
        }
      } else {
        // Unneeded except for rank 3.
        (void)threej_rs;
        (void)threej_ws;
      }
      for (size_t lprime = lcutminus; lprime <= ell_max + rank; ++lprime) {
        const double coeflprime =
            half_power.has_value() and lprime <= lcutplus
                ? 0.5 * (2 * lprime + 1) *
                      (1.0 -
                       exp(-36.0 *
                           IPow(double(lprime) / double(lcutplus + 1),
                                2 * static_cast<int>(half_power.value()))))
                : 0.5 * (2 * lprime + 1);
        for (int mprime = -lprime; mprime <= static_cast<int>(lprime);
             ++mprime) {
          // Here is where the formulas differ for different ranks.
          if constexpr (rank == 1) {
            for (int p = -1; p <= 1; p += 2) {
              const int mdest = mprime + helpers::bv_to_m(dest_bvs[0], p);
              if (mdest >= 0) {
                // Fill only nonnegative m, since that is all we store
                WignerThreeJ threej_p(lprime, mprime, 1,
                                      helpers::bv_to_m(dest_bvs[0], p));
                for (int j = -1; j <= 1; j += 2) {
                  WignerThreeJ threej_j(lprime, mprime, 1,
                                        helpers::bv_to_m(src_bvs[0], j));
                  const int msrc = mprime + helpers::bv_to_m(src_bvs[0], j);
                  std::complex<double> coefjp =
                      -coeflprime * helpers::bv_to_k(src_bvs[0], j) *
                      helpers::bv_to_k(dest_bvs[0], p);
                  if ((mdest + msrc) % 2 != 0.0) {
                    coefjp *= -1.0;
                  }
                  if (src_bvs[0] == helpers::BasisVector::y) {
                    coefjp *= -1.0;
                  }
                  inner_loops_one(filler, iter_src, iter_dest, src_comp_index,
                                  dest_comp_index, ell_max, mdest, msrc, coefjb,
                                  threej_j, threej_p);
                }
              }
            }
          } else if constexpr (rank == 2) {
            for (int lbar = 0; lbar <= 2; ++lbar) {
              const double SymmFactor = [&src_multiplicity]() {
                if constexpr (std::is_same_v<Symmetry<0, 1>,
                                             TensorStructure::symmetry>) {
                  // "ab" symmetry
                  (void)src_multiplicity;
                  return 1.0;
                } else {
                  // "aa" symmetry
                  return (src_multiplicity / 2.0) *
                         (1 + (lbar % 2 == 0 ? 1.0 : -1.0));
                }
              }();
              const double coeflbar = 0.5 * (2 * lbar + 1);
              for (int mbar = -lbar; mbar <= lbar; ++mbar) {
                WignerThreeJ threej_lbar(lprime, mprime, lbar, mbar);
                for (int mtilde = -lbar; mtilde <= lbar; ++mtilde) {
                  WignerThreeJ threej_ltilde(lprime, -mprime, lbar, mtilde);
                  inner_loops_two(filler, iter_src, iter_dest, src_comp_index,
                                  dest_comp_index, ell_max, lprime, mprime,
                                  lbar, mbar, mtilde, coeflbar, coeflprime,
                                  threej_lbar, threej_ltilde, mbars, mtildes,
                                  src_bvs, dest_bvs, SymmFactor, sign_y,
                                  threej_pqs, threej_uvs);
                }
              }
            }
          } else if constexpr (rank == 3) {
            for (int lhat = 0; lhat <= 3; ++lhat) {
              for (int mhat = -lhat; mhat <= lhat; ++mhat) {
                WignerThreeJ threej_mhat(lprime, -mprime, lhat, mhat);
                for (int mcheck = -lhat; mcheck <= lhat; ++mcheck) {
                  WignerThreeJ threej_mcheck(lprime, mprime, lhat, mcheck);
                  size_t mbar_indx = 0;
                  for (int p = -1; p <= 1; p += 2) {
                    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
                      for (int r = -1; r <= 1; r += 2) {
                        const int mr = helpers::bv_to_m(dest_bvs[0], r);
                        if (mcheck == mr + mbars[mbar_indx] and
                            mprime + mcheck >= 0) {
                          inner_loops_three<TensorStructure::symmetry>(
                              filler, iter_src, iter_dest, src_comp_index,
                              dest_comp_index, ell_max, lprime, coeflprime,
                              mprime, lhat, mhat, threej_mhat, mcheck,
                              threej_mcheck, mbar_indx, p, q, r, mr, mbars,
                              mtildes, src_bvs, dest_bvs, threej_pqs,
                              threej_uvs, threej_ws, threej_rs, sign_coef3j);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  filler.fill(matrix);
}

// Explicit instantiations
template FillFilter<typename tnsr::i::structure>(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    size_t ell_max, size_t number_of_ell_modes_to_kill,
    std::optional<size_t> half_power);
template FillFilter<typename tnsr::ii<DataVector, 3>::structure>(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    size_t ell_max, size_t number_of_ell_modes_to_kill,
    std::optional<size_t> half_power);
template FillFilter<typename tnsr::ij<DataVector, 3>::structure>(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    size_t ell_max, size_t number_of_ell_modes_to_kill,
    std::optional<size_t> half_power);
template FillFilter<typename tnsr::ijj<DataVector, 3>::structure>(
    gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*> matrix,
    size_t ell_max, size_t number_of_ell_modes_to_kill,
    std::optional<size_t> half_power);

};  // namespace ylm::TensorYlm
