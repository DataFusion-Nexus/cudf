/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/optional>

#include <type_traits>
#include <utility>

namespace cudf {
namespace detail {

template <typename Comparator>
struct queryable_lexicographic_comparator {
  cuda::std::optional<Comparator> comparator{};

  queryable_lexicographic_comparator() = default;
  explicit queryable_lexicographic_comparator(Comparator value) : comparator{std::move(value)} {}

  __device__ bool operator()(size_type lhs, size_type rhs) const { return (*comparator)(lhs, rhs); }
};

template <sort_method method, typename Comparator>
std::size_t sorted_order_lexicographic_temp_storage_bytes(size_type num_rows,
                                                          Comparator comparator,
                                                          rmm::cuda_stream_view stream)
{
  auto temporary_bytes = std::size_t{0};
  auto input           = cuda::counting_iterator<size_type>{0};
  auto* output         = static_cast<size_type*>(nullptr);
  if constexpr (method == sort_method::STABLE) {
    cub::DeviceMergeSort::StableSortKeysCopy(
      nullptr, temporary_bytes, input, output, num_rows, comparator, stream.value());
  } else {
    cub::DeviceMergeSort::SortKeysCopy(
      nullptr, temporary_bytes, input, output, num_rows, comparator, stream.value());
  }
  return temporary_bytes;
}

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,rmm::device_async_resource_ref
 * )
 *
 * @tparam stable Whether to use stable sort
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <sort_method method>
std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return cudf::make_numeric_column(
      data_type(type_to_id<size_type>()), 0, mask_state::UNALLOCATED, stream, mr);
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null_precedence size.");
  }

  // fast-path for single column sort
  if (input.num_columns() == 1 and not cudf::is_nested(input.column(0).type())) {
    auto const single_col = input.column(0);
    auto const col_order  = column_order.empty() ? order::ASCENDING : column_order.front();
    auto const null_prec  = null_precedence.empty() ? null_order::BEFORE : null_precedence.front();
    return sorted_order<method>(single_col, col_order, null_prec, stream, mr);
  }

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
  auto const do_sort                       = [&](auto const comparator) {
    using queryable_comparator =
      queryable_lexicographic_comparator<std::remove_cvref_t<decltype(comparator)>>;
    auto const queryable = queryable_comparator{comparator};
    auto temporary_bytes =
      sorted_order_lexicographic_temp_storage_bytes<method>(input.num_rows(), queryable, stream);
    auto temporary      = rmm::device_buffer(temporary_bytes, stream, mr);
    auto input_indices  = cuda::counting_iterator<size_type>{0};
    auto output_indices = mutable_indices_view.begin<size_type>();
    if constexpr (method == sort_method::STABLE) {
      cub::DeviceMergeSort::StableSortKeysCopy(temporary.data(),
                                               temporary_bytes,
                                               input_indices,
                                               output_indices,
                                               input.num_rows(),
                                               queryable,
                                               stream.value());
    } else {
      cub::DeviceMergeSort::SortKeysCopy(temporary.data(),
                                         temporary_bytes,
                                         input_indices,
                                         output_indices,
                                         input.num_rows(),
                                         queryable,
                                         stream.value());
    }
  };

  auto const comp = cudf::detail::row::lexicographic::self_comparator(
    input, column_order, null_precedence, stream, mr);
  if (cudf::detail::has_nested_columns(input)) {
    auto const comparator = comp.less<true>(nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  } else {
    auto const comparator = comp.less<false>(nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  }

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
