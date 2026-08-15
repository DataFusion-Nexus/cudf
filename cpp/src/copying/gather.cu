/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/iterator/transform_iterator.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace detail {

namespace {

std::size_t checked_byte_product(std::size_t count, std::size_t element_size, char const* message)
{
  CUDF_EXPECTS(element_size == 0 || count <= std::numeric_limits<std::size_t>::max() / element_size,
               message);
  return count * element_size;
}

std::size_t checked_byte_sum(std::size_t lhs, std::size_t rhs, char const* message)
{
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() - rhs, message);
  return lhs + rhs;
}

struct fixed_width_dispatch_check {
  template <typename Element>
  bool operator()() const
  {
    return cudf::is_rep_layout_compatible<Element>();
  }
};

bool gather_fixed_width_dispatchable(data_type type)
{
  return cudf::type_dispatcher<dispatch_storage_type>(type, fixed_width_dispatch_check{});
}

gather_fixed_width_dont_check_preflight_result gather_fixed_width_dont_check_preflight_impl(
  std::int64_t output_rows,
  std::vector<gather_fixed_width_column_metadata> const& source_columns,
  std::int32_t)
{
  CUDF_EXPECTS(output_rows >= 0,
               "gather fixed-width DONT_CHECK preflight rows must be non-negative");
  CUDF_EXPECTS(output_rows <= std::numeric_limits<size_type>::max(),
               "gather fixed-width DONT_CHECK preflight rows exceed cudf::size_type");

  gather_fixed_width_dont_check_preflight_result result{};
  auto const rows         = static_cast<std::size_t>(output_rows);
  auto const column_count = source_columns.size();
  bool any_nullable       = false;

  result.gather_map_bytes = checked_byte_product(
    rows, sizeof(size_type), "gather fixed-width DONT_CHECK map byte count overflowed");

  for (auto const& column : source_columns) {
    CUDF_EXPECTS(column.num_children >= 0,
                 "gather fixed-width DONT_CHECK preflight child count must be non-negative");
    CUDF_EXPECTS(column.num_children == 0,
                 "gather fixed-width DONT_CHECK preflight does not support nested columns");
    CUDF_EXPECTS(cudf::is_fixed_width(column.type),
                 "gather fixed-width DONT_CHECK preflight requires fixed-width columns");
    CUDF_EXPECTS(not cudf::is_fixed_point(column.type),
                 "gather fixed-width DONT_CHECK preflight does not support decimal columns");
    CUDF_EXPECTS(gather_fixed_width_dispatchable(column.type),
                 "gather fixed-width DONT_CHECK preflight type is not gather-dispatchable");

    auto const column_bytes =
      checked_byte_product(rows,
                           cudf::size_of(column.type),
                           "gather fixed-width DONT_CHECK output byte count overflowed");
    result.output_data_bytes =
      checked_byte_sum(result.output_data_bytes,
                       column_bytes,
                       "gather fixed-width DONT_CHECK output byte sum overflowed");

    if (column.nullable) {
      any_nullable = true;
      auto const mask_bytes =
        cudf::bitmask_allocation_size_bytes(static_cast<size_type>(output_rows));
      result.output_null_mask_bytes =
        checked_byte_sum(result.output_null_mask_bytes,
                         mask_bytes,
                         "gather fixed-width DONT_CHECK output null-mask byte sum overflowed");
    }
  }

  if (any_nullable) {
    result.target_mask_pointer_array_bytes = checked_byte_product(
      column_count,
      sizeof(bitmask_type*),
      "gather fixed-width DONT_CHECK target-mask pointer byte count overflowed");

    auto const views_bytes =
      checked_byte_product(column_count,
                           sizeof(column_device_view),
                           "gather fixed-width DONT_CHECK table-device-view byte count overflowed");
    result.source_table_device_view_bytes =
      checked_byte_sum(views_bytes,
                       alignof(column_device_view) - 1,
                       "gather fixed-width DONT_CHECK table-device-view alignment overflowed");

    result.valid_count_array_bytes =
      checked_byte_product(column_count,
                           sizeof(size_type),
                           "gather fixed-width DONT_CHECK valid-count byte count overflowed");

    result.native_temporary_workspace_bytes =
      checked_byte_sum(result.target_mask_pointer_array_bytes,
                       result.source_table_device_view_bytes,
                       "gather fixed-width DONT_CHECK temporary byte sum overflowed");
    result.native_temporary_workspace_bytes =
      checked_byte_sum(result.native_temporary_workspace_bytes,
                       result.valid_count_array_bytes,
                       "gather fixed-width DONT_CHECK temporary byte sum overflowed");
  }

  result.active_phase_peak_bytes =
    checked_byte_sum(result.gather_map_bytes,
                     result.output_data_bytes,
                     "gather fixed-width DONT_CHECK active-phase byte sum overflowed");
  result.active_phase_peak_bytes =
    checked_byte_sum(result.active_phase_peak_bytes,
                     result.output_null_mask_bytes,
                     "gather fixed-width DONT_CHECK active-phase byte sum overflowed");
  result.active_phase_peak_bytes =
    checked_byte_sum(result.active_phase_peak_bytes,
                     result.native_temporary_workspace_bytes,
                     "gather fixed-width DONT_CHECK active-phase byte sum overflowed");
  return result;
}

}  // namespace

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

  // create index type normalizing iterator for the gather_map
  auto map_begin = indexalator_factory::make_input_iterator(gather_map);
  auto map_end   = map_begin + gather_map.size();

  if (neg_indices == negative_index_policy::ALLOWED) {
    cudf::size_type n_rows = source_table.num_rows();
    auto idx_converter     = cuda::proclaim_return_type<size_type>(
      [n_rows] __device__(size_type in) { return in < 0 ? in + n_rows : in; });
    return gather(source_table,
                  thrust::make_transform_iterator(map_begin, idx_converter),
                  thrust::make_transform_iterator(map_end, idx_converter),
                  bounds_policy,
                  stream,
                  mr);
  }
  return gather(source_table, map_begin, map_end, bounds_policy, stream, mr);
}

std::unique_ptr<table> gather(table_view const& source_table,
                              device_span<size_type const> const gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(gather_map.size() <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "gather map size exceeds the column size limit",
               std::overflow_error);
  auto map_col = column_view(data_type{type_to_id<size_type>()},
                             static_cast<size_type>(gather_map.size()),
                             gather_map.data(),
                             nullptr,
                             0);
  return detail::gather(source_table, map_col, bounds_policy, neg_indices, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const index_policy = is_unsigned(gather_map.type()) ? negative_index_policy::NOT_ALLOWED
                                                           : negative_index_policy::ALLOWED;

  return detail::gather(source_table, gather_map, bounds_policy, index_policy, stream, mr);
}

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::gather(source_table, gather_map, bounds_policy, neg_indices, stream, mr);
}

gather_fixed_width_dont_check_preflight_result gather_fixed_width_dont_check_preflight(
  std::int64_t output_rows,
  std::vector<gather_fixed_width_column_metadata> const& source_columns,
  std::int32_t device)
{
  return detail::gather_fixed_width_dont_check_preflight_impl(output_rows, source_columns, device);
}

gather_fixed_width_dont_check_preflight_result gather_fixed_width_dont_check_preflight(
  table_view const& source_table, std::int64_t output_rows, std::int32_t device)
{
  std::vector<gather_fixed_width_column_metadata> source_columns;
  source_columns.reserve(source_table.num_columns());
  for (auto const& column : source_table) {
    source_columns.push_back({column.type(), column.nullable(), column.num_children()});
  }
  return detail::gather_fixed_width_dont_check_preflight_impl(output_rows, source_columns, device);
}

}  // namespace cudf
