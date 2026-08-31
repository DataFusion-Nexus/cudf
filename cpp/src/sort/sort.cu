/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_impl.cuh"
#include "sort_radix.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

namespace cudf {
namespace detail {
std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  return sorted_order<sort_method::UNSTABLE>(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");

  auto sorted_order = detail::sorted_order(
    keys, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());

  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  // fast-path sort conditions: single, fixed-width column with no nulls
  if (input.num_columns() == 1 && is_radix_sortable(input.column(0))) {
    auto order  = (column_order.empty() ? order::ASCENDING : column_order.front());
    auto output = sort_radix(input.column(0), order == order::ASCENDING, stream, mr);
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(std::move(output));
    return std::make_unique<table>(std::move(columns));
  }
  return detail::sort_by_key(input, input, column_order, null_precedence, stream, mr);
}

}  // namespace detail

namespace {

std::size_t lexicographic_checked_add(std::size_t lhs, std::size_t rhs)
{
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() - rhs,
               "sorted-order lexicographic byte count overflowed");
  return lhs + rhs;
}

std::size_t lexicographic_checked_mul(std::size_t lhs, std::size_t rhs)
{
  if (lhs == 0 || rhs == 0) { return 0; }
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() / rhs,
               "sorted-order lexicographic byte count overflowed");
  return lhs * rhs;
}

std::vector<column_view> lexicographic_metadata_columns(
  size_type num_rows,
  std::vector<data_type> const& key_types,
  std::vector<std::int64_t> const& null_counts)
{
  std::vector<column_view> columns;
  columns.reserve(key_types.size());
  for (std::size_t index = 0; index < key_types.size(); ++index) {
    auto const null_count = static_cast<size_type>(null_counts[index]);
    auto const null_mask  = null_count == 0 ? nullptr : reinterpret_cast<bitmask_type const*>(1);
    if (key_types[index].id() == type_id::STRING) {
      CUDF_EXPECTS(num_rows < std::numeric_limits<size_type>::max(),
                   "sorted-order lexicographic string row count exceeds offsets capacity");
      auto const offsets = column_view{data_type{type_id::INT32},
                                       static_cast<size_type>(num_rows + 1),
                                       reinterpret_cast<void const*>(1),
                                       nullptr,
                                       0};
      columns.emplace_back(key_types[index],
                           num_rows,
                           reinterpret_cast<void const*>(1),
                           null_mask,
                           null_count,
                           0,
                           std::vector<column_view>{offsets});
    } else {
      columns.emplace_back(
        key_types[index], num_rows, reinterpret_cast<void const*>(1), null_mask, null_count);
    }
  }
  return columns;
}

}  // namespace

sorted_order_lexicographic_preflight_result sorted_order_lexicographic_preflight(
  std::int64_t num_rows,
  std::vector<data_type> const& key_types,
  std::vector<std::int64_t> const& null_counts,
  bool stable,
  std::vector<order> const& key_orders,
  std::vector<null_order> const& null_precedence,
  std::int32_t device)
{
  CUDF_EXPECTS(num_rows >= 0, "sorted-order lexicographic rows must be non-negative");
  CUDF_EXPECTS(num_rows <= std::numeric_limits<size_type>::max(),
               "sorted-order lexicographic rows exceed cudf::size_type");
  CUDF_EXPECTS(key_types.size() >= 2,
               "sorted-order lexicographic preflight requires at least two keys");
  CUDF_EXPECTS(key_types.size() == null_counts.size() && key_types.size() == key_orders.size() &&
                 key_types.size() == null_precedence.size(),
               "sorted-order lexicographic metadata lengths differ");
  CUDF_EXPECTS(key_types.size() <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "sorted-order lexicographic key count exceeds cudf::size_type");
  CUDF_EXPECTS(!stable, "sorted-order lexicographic preflight supports only unstable sorting");
  CUDF_EXPECTS(device >= 0, "sorted-order lexicographic device must be non-negative");
  for (std::size_t index = 0; index < key_types.size(); ++index) {
    CUDF_EXPECTS(!is_nested(key_types[index]) && is_relationally_comparable(key_types[index]),
                 "sorted-order lexicographic key type is unsupported");
    CUDF_EXPECTS(null_counts[index] >= 0 && null_counts[index] <= num_rows,
                 "sorted-order lexicographic null count is invalid");
    CUDF_EXPECTS(key_orders[index] == order::ASCENDING || key_orders[index] == order::DESCENDING,
                 "sorted-order lexicographic key order is invalid");
    CUDF_EXPECTS(
      null_precedence[index] == null_order::BEFORE || null_precedence[index] == null_order::AFTER,
      "sorted-order lexicographic null precedence is invalid");
  }
  if (num_rows == 0) { return {}; }

  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device}};
  auto const rows    = static_cast<size_type>(num_rows);
  auto const columns = lexicographic_metadata_columns(rows, key_types, null_counts);
  auto const view_bytes =
    std::accumulate(columns.begin(),
                    columns.end(),
                    std::size_t{alignof(column_device_view) - 1},
                    [](std::size_t total, column_view const& column) {
                      return lexicographic_checked_add(total, column_device_view::extent(column));
                    });
  auto comparator_state_bytes = view_bytes;
  comparator_state_bytes      = lexicographic_checked_add(
    comparator_state_bytes, lexicographic_checked_mul(key_types.size(), sizeof(order)));
  comparator_state_bytes = lexicographic_checked_add(
    comparator_state_bytes, lexicographic_checked_mul(key_types.size(), sizeof(null_order)));
  comparator_state_bytes = lexicographic_checked_add(
    comparator_state_bytes, lexicographic_checked_mul(key_types.size(), sizeof(int)));

  using row_comparator =
    decltype(std::declval<detail::row::lexicographic::self_comparator>().less<false>(
      nullate::DYNAMIC{false}));
  using queryable_comparator = detail::queryable_lexicographic_comparator<row_comparator>;
  auto const temporary_workspace_bytes =
    detail::sorted_order_lexicographic_temp_storage_bytes<detail::sort_method::UNSTABLE>(
      rows, queryable_comparator{}, get_default_stream());
  auto const retained_order_bytes =
    lexicographic_checked_mul(static_cast<std::size_t>(rows), sizeof(size_type));
  auto active_phase_peak_bytes =
    lexicographic_checked_add(comparator_state_bytes, temporary_workspace_bytes);
  active_phase_peak_bytes =
    lexicographic_checked_add(active_phase_peak_bytes, retained_order_bytes);
  return {comparator_state_bytes,
          temporary_workspace_bytes,
          retained_order_bytes,
          active_phase_peak_bytes};
}

std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sorted_order(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_by_key(values, keys, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
