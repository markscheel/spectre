// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename Symm = Symmetry<>, typename IndexList = index_list<>>
class SparseMatrixCollection;
/// \endcond

/*!
 * \brief Holds a set of sparse matrices that act as a larger sparse matrix
 * that can be used to multiply an entire tensor.
 *
 * \tparam Symm Symmetry specification, as in Tensor
 * \tparam IndexList indices typelist, as in Tensor
 */
template <typename Symm, template <typename...> class IndexList,
          typename... Indices>
class SparseMatrixCollection<Symm, IndexList<Indices...>> {
 public:
  using structure = Tensor_detail::Structure<Symm, Indices...>;
  static constexpr size_t num_independent_components = structure::size();

  /// Does the equivalent of result = M * rhs,
  /// where M is the SparseMatrixCollection.
  void apply(
      gsl::not_null<Tensor<DataVector, Symm, IndexList<Indices...>>*> result,
      const Tensor<DataVector, Symm, IndexList<Indices...>>& rhs);

  /// Uses a SparseMatrixFiller to fill one of the internal matrices,
  /// the one that connects the src_storage_index of 'rhs' to the
  /// dest_storage_index of 'result'.
  void fill(const SparseMatrixFiller& filler, size_t dest_storage_index,
            size_t src_storage_index) {
    filler.fill(make_not_null(&gsl::at(
        matrices_,
        src_storage_index + num_independent_components * dest_storage_index)));
  }

 private:
  std::array<blaze::CompressedMatrix<double, blaze::rowMajor>,
             num_independent_components * num_independent_components>
      matrices_;
};

template <typename Symm, template <typename...> class IndexList,
          typename... Indices>
void SparseMatrixCollection<Symm, IndexList<Indices...>>::apply(
    gsl::not_null<Tensor<DataVector, Symm, IndexList<Indices...>>*> result,
    const Tensor<DataVector, Symm, IndexList<Indices...>>& rhs) {
  for (size_t dest_storage_index = 0;
       dest_storage_index < num_independent_components; ++dest_storage_index) {
    auto& result_comp = result[dest_storage_index];
    result_comp = 0.0;
    for (size_t src_storage_index = 0;
         src_storage_index < num_independent_components; ++src_storage_index) {
      const auto& result_comp = rhs[src_storage_index];
      const auto& matrix =
          gsl::at(matrices_, src_storage_index + num_independent_components *
                                                     dest_storage_index);
      result_comp += matrix * rhs_comp;
    }
  }
}
