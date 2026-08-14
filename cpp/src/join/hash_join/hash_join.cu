/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/storage.cuh>
#include <cuda/iterator>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cudf::detail {

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  if (left.is_empty() || right.is_empty()) { return true; }
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }
  return false;
}

void validate_hash_join_probe(table_view const& right, table_view const& left, bool has_nulls)
{
  CUDF_EXPECTS(0 != left.num_columns(), "Hash join left table is empty", std::invalid_argument);
  CUDF_EXPECTS(right.num_columns() == left.num_columns(),
               "Mismatch in number of columns to be joined on",
               std::invalid_argument);
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::have_same_types(right, left),
               "Mismatch in joining column data types",
               cudf::data_type_error);
}

namespace {
std::size_t checked_add(std::size_t lhs, std::size_t rhs)
{
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() - rhs,
               "hash join pre-build reservation size overflow",
               std::overflow_error);
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs)
{
  CUDF_EXPECTS(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
               "hash join pre-build reservation size overflow",
               std::overflow_error);
  return lhs * rhs;
}

std::size_t hash_join_slot_storage_reservation_size(cudf::size_type rows, double load_factor)
{
  using slot_t    = hash_table_t::value_type;
  using probing_t = hash_table_t::probing_scheme_type;
  using storage_t = cuco::storage<hash_table_t::bucket_size>;

  auto const extent = cuco::make_valid_extent<probing_t, storage_t>(
    cuco::extent<std::size_t>{static_cast<std::size_t>(rows)}, load_factor);
  auto const capacity_slots  = static_cast<std::size_t>(extent);
  constexpr auto alignment   = hash_table_t::storage_ref_type::alignment;
  constexpr auto extra_slots = (alignment - 1) / sizeof(slot_t) + 1;
  return checked_mul(checked_add(capacity_slots, extra_slots), sizeof(slot_t));
}

std::size_t flat_table_device_view_reservation_size(cudf::table_view const& table)
{
  auto view_bytes = std::size_t{0};
  for (auto const& column : table) {
    view_bytes = checked_add(view_bytes, cudf::column_device_view::extent(column));
  }
  return checked_add(view_bytes, alignof(cudf::column_device_view) - 1);
}

bool has_unsupported_retained_preprocessing_buffers(cudf::table_view const& table)
{
  return cudf::detail::has_nested_columns(table);
}

template <typename Offset>
std::size_t flat_string_null_preprocessing_reservation_size(cudf::column_view const& column,
                                                            rmm::cuda_stream_view stream)
{
  constexpr auto metadata_chunk_rows = std::size_t{4096};
  constexpr auto bits_per_word       = sizeof(cudf::bitmask_type) * CHAR_BIT;
  constexpr auto max_mask_words = (metadata_chunk_rows + bits_per_word - 1) / bits_per_word + 1;
  auto host_offsets             = std::array<Offset, metadata_chunk_rows + 1>{};
  auto host_mask                = std::array<cudf::bitmask_type, max_mask_words>{};
  auto const offsets            = cudf::strings_column_view{column}.offsets();

  auto valid_chars_bytes = std::size_t{0};
  auto has_nonempty_null = false;
  for (auto row_begin = std::size_t{0}; row_begin < static_cast<std::size_t>(column.size());) {
    auto const chunk_rows =
      std::min(metadata_chunk_rows, static_cast<std::size_t>(column.size()) - row_begin);
    auto const absolute_row_begin =
      checked_add(static_cast<std::size_t>(column.offset()), row_begin);
    auto const first_mask_word = absolute_row_begin / bits_per_word;
    auto const end_bit         = checked_add(absolute_row_begin, chunk_rows);
    auto const mask_word_count = checked_add((end_bit - 1) / bits_per_word, 1) - first_mask_word;

    CUDF_CUDA_TRY(cudaMemcpyAsync(host_offsets.data(),
                                  offsets.data<Offset>() + absolute_row_begin,
                                  checked_mul(checked_add(chunk_rows, 1), sizeof(Offset)),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_mask.data(),
                                  column.null_mask() + first_mask_word,
                                  checked_mul(mask_word_count, sizeof(cudf::bitmask_type)),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();

    for (auto row = std::size_t{0}; row < chunk_rows; ++row) {
      auto const begin = host_offsets[row];
      auto const end   = host_offsets[row + 1];
      CUDF_EXPECTS(begin >= 0 && end >= begin,
                   "invalid string offsets while sizing hash join preprocessing",
                   std::invalid_argument);
      auto const row_bytes    = static_cast<std::size_t>(end - begin);
      auto const absolute_bit = checked_add(absolute_row_begin, row);
      auto const relative_bit = absolute_bit - first_mask_word * bits_per_word;
      if (cudf::bit_is_set(host_mask.data(), static_cast<cudf::size_type>(relative_bit))) {
        valid_chars_bytes = checked_add(valid_chars_bytes, row_bytes);
      } else {
        has_nonempty_null = has_nonempty_null || row_bytes != 0;
      }
    }
    row_begin = checked_add(row_begin, chunk_rows);
  }

  if (!has_nonempty_null) { return 0; }

  auto const offsets_count = checked_add(static_cast<std::size_t>(column.size()), 1);
  auto const offset_width =
    valid_chars_bytes >= static_cast<std::size_t>(cudf::strings::get_offset64_threshold())
      ? sizeof(int64_t)
      : sizeof(int32_t);
  auto const offsets_bytes   = checked_mul(offsets_count, offset_width);
  auto const null_mask_bytes = cudf::bitmask_allocation_size_bytes(column.size());
  return checked_add(checked_add(offsets_bytes, valid_chars_bytes), null_mask_bytes);
}

std::size_t flat_null_preprocessing_reservation_size(cudf::table_view const& table,
                                                     rmm::cuda_stream_view stream)
{
  auto bytes = std::size_t{0};
  for (auto const& column : table) {
    if (column.type().id() != cudf::type_id::STRING || !column.has_nulls()) { continue; }

    auto const offsets_type = cudf::strings_column_view{column}.offsets().type().id();
    switch (offsets_type) {
      case cudf::type_id::INT32:
        bytes = checked_add(
          bytes, flat_string_null_preprocessing_reservation_size<int32_t>(column, stream));
        break;
      case cudf::type_id::INT64:
        bytes = checked_add(
          bytes, flat_string_null_preprocessing_reservation_size<int64_t>(column, stream));
        break;
      default: CUDF_FAIL("string offsets must use INT32 or INT64", cudf::data_type_error);
    }
  }
  return bytes;
}

void build_hash_join(
  cudf::table_view const& right,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_right,
  cudf::detail::hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != right.num_columns(), "Selected right dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != right.num_rows(), "Right side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& right, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(right)) {
      hash_table.insert(iter, iter + right.num_rows(), stream.value());
    } else {
      auto const stencil = cuda::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      hash_table.insert_if(iter, iter + right.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(right)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nulls, preprocessed_right};

    insert_rows(right, d_hasher);
  } else {
    auto const row_hash = detail::row::hash::row_hasher{preprocessed_right};
    auto const d_hasher = row_hash.device_hasher(nulls);

    insert_rows(right, d_hasher);
  }
}
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : hash_join{right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr)}
{
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _has_nulls(has_nulls),
    _is_empty{right.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _impl{std::make_unique<impl>(impl{typename impl::hash_table_t{
      cuco::extent{static_cast<size_t>(right.num_rows())},
      load_factor,
      cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
      {},
      {},
      {},
      {},
      rmm::mr::polymorphic_allocator<char>{mr},
      stream.value()}})},
    _right{right},
    _preprocessed_right{cudf::detail::row::equality::preprocessed_table::create(
      _right, stream, rmm::device_async_resource_ref{mr})}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != right.num_columns(), "Hash join right table is empty", std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);

  if (_is_empty) { return; }

  auto const row_bitmask =
    cudf::detail::bitmask_and(right, stream, rmm::device_async_resource_ref{mr}).first;
  cudf::detail::build_hash_join(_right,
                                _preprocessed_right,
                                _impl->_hash_table,
                                _has_nulls,
                                _nulls_equal,
                                reinterpret_cast<bitmask_type const*>(row_bitmask.data()),
                                stream);
}

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template <typename Hasher>
hash_join<Hasher>::~hash_join() = default;

template hash_join<hash_join_hasher>::~hash_join();

}  // namespace cudf::detail

namespace cudf {

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& right,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : hash_join(right,
              nullable_join::YES,
              compare_nulls,
              cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
              stream,
              std::move(mr))
{
}

hash_join::hash_join(cudf::table_view const& right,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl_type const>(
      right, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream, std::move(mr))}
{
}

std::size_t hash_join::pre_build_reservation_size(cudf::table_view const& build,
                                                  double load_factor,
                                                  rmm::cuda_stream_view stream,
                                                  [[maybe_unused]] rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join right table is empty", std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);
  CUDF_EXPECTS(
    !cudf::detail::has_unsupported_retained_preprocessing_buffers(build),
    "hash join pre-build reservation sizing does not yet support nested build-key tables",
    std::invalid_argument);

  auto const hash_table_bytes =
    cudf::detail::hash_join_slot_storage_reservation_size(build.num_rows(), load_factor);
  auto const view_bytes = cudf::detail::flat_table_device_view_reservation_size(build);
  auto const preprocessing_bytes =
    cudf::detail::flat_null_preprocessing_reservation_size(build, stream);
  return cudf::detail::checked_add(cudf::detail::checked_add(hash_table_bytes, view_bytes),
                                   preprocessing_bytes);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& left,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->full_join(left, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& left,
                                       rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(left, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream) const
{
  return _impl->left_join_size(left, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(left, stream, mr);
}

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& left,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& left,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(left, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_inner_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_left_join(cudf::join_partition_context const& context,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_left_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_full_join(cudf::join_partition_context const& context,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_full_join(context, stream, mr);
}

}  // namespace cudf
