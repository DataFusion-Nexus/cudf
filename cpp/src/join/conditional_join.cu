/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/conditional_join.hpp"
#include "join/conditional_join_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace detail {
namespace {
constexpr int DEFAULT_CACHE_SIZE = 128;

void validate_conditional_full_join_materializable_size(std::size_t output_size)
{
  CUDF_EXPECTS(output_size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "conditional full join output size exceeds cudf::size_type maximum",
               std::overflow_error);
}

struct unmatched_right_row {
  cudf::device_span<size_type const> right_row_has_match;

  __device__ bool operator()(size_type row_index) const noexcept
  {
    return right_row_has_match[row_index] == 0;
  }
};

struct conditional_full_join_setup_data {
  cudf::nullate::DYNAMIC has_nulls;
  ast::detail::expression_parser parser;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> left_table;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> right_table;
  detail::grid_1d config;
  thread_index_type shmem_size_per_block;
};

conditional_full_join_setup_data setup_conditional_full_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const has_nulls =
    cudf::nullate::DYNAMIC{binary_predicate.may_evaluate_null(left, right, stream)};

  auto parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  auto left_table  = table_device_view::create(left, stream, mr);
  auto right_table = table_device_view::create(right, stream, mr);

  detail::grid_1d const config(left.num_rows(), DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  return {has_nulls,
          std::move(parser),
          std::move(left_table),
          std::move(right_table),
          config,
          shmem_size_per_block};
}

void validate_conditional_full_join_predicate(table_view const& left,
                                              table_view const& right,
                                              ast::expression const& binary_predicate,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const has_nulls =
    cudf::nullate::DYNAMIC{binary_predicate.may_evaluate_null(left, right, stream)};
  auto parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);
}
}  // namespace

std::unique_ptr<rmm::device_uvector<size_type>> conditional_join_anti_semi(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  join_kind join_type,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (right.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_ANTI_JOIN: return get_trivial_left_join_indices(left, stream, mr).first;
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_ANTI_JOIN: [[fallthrough]];
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a Boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  detail::grid_1d const config(left.num_rows(), DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // TODO: Remove the output_size parameter. It is not needed because the
  // output size is bounded by the size of the left table.
  std::size_t join_size;
  if (output_size.has_value()) {
    join_size = *output_size;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    cudf::detail::device_scalar<std::size_t> size(0, stream, mr);
    if (has_nulls) {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table, *right_table, join_type, parser.device_expression_data, false, size.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table, *right_table, join_type, parser.device_expression_data, false, size.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    join_size = size.value(stream);
  }

  cudf::detail::device_scalar<std::size_t> write_index(
    0, stream, cudf::get_current_device_resource_ref());

  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();

  if (has_nulls) {
    conditional_join_anti_semi<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        join_output_l,
        write_index.data(),
        parser.device_expression_data,
        join_size);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    conditional_join_anti_semi<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        join_output_l,
        write_index.data(),
        parser.device_expression_data,
        join_size);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return left_indices;
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_join(table_view const& left,
                 table_view const& right,
                 ast::expression const& binary_predicate,
                 join_kind join_type,
                 std::optional<std::size_t> output_size,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right.num_rows() == 0) {
    switch (join_type) {
      // Left, left anti, and full all return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left, stream, mr);
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left.num_rows() == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        auto ret_flipped = get_trivial_left_join_indices(right, stream, mr);
        return std::pair(std::move(ret_flipped.second), std::move(ret_flipped.first));
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows());
  detail::grid_1d const config(swap_tables ? right.num_rows() : left.num_rows(),
                               DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;
  join_kind const kernel_join_type =
    join_type == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : join_type;

  // If the join size was not provided as an input, compute it here.
  std::size_t join_size;
  if (output_size.has_value()) {
    join_size = *output_size;
  } else {
    // Allocate storage for the counter used to get the size of the join output
    cudf::detail::device_scalar<std::size_t> size(0, stream, mr);
    if (has_nulls) {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          kernel_join_type,
          parser.device_expression_data,
          swap_tables,
          size.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
          *left_table,
          *right_table,
          kernel_join_type,
          parser.device_expression_data,
          swap_tables,
          size.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    join_size = size.value(stream);
  }

  // The initial early exit clauses guarantee that we will not reach this point
  // unless both the left and right tables are non-empty. Under that
  // constraint, neither left nor full joins can return an empty result since
  // at minimum we are guaranteed null matches for all non-matching rows. In
  // all other cases (inner, left semi, and left anti joins) if we reach this
  // point we can safely return an empty result.
  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  cudf::detail::device_scalar<std::size_t> write_index(
    0, stream, cudf::get_current_device_resource_ref());

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();
  auto const& join_output_r = right_indices->data();

  if (has_nulls) {
    conditional_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        kernel_join_type,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    conditional_join<DEFAULT_JOIN_BLOCK_SIZE, DEFAULT_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        kernel_join_type,
        join_output_l,
        join_output_r,
        write_index.data(),
        parser.device_expression_data,
        join_size,
        swap_tables);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto join_indices = std::pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (join_type == join_kind::FULL_JOIN) {
    join_indices = detail::finalize_full_join(
      std::move(join_indices), left.num_rows(), right.num_rows(), stream, mr);
  }
  return join_indices;
}

std::size_t compute_conditional_join_output_size(table_view const& left,
                                                 table_view const& right,
                                                 ast::expression const& binary_predicate,
                                                 join_kind join_type,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  // Until we add logic to handle the number of non-matches in the right table,
  // full joins are not supported in this function. Note that this does not
  // prevent actually performing full joins since we do that by calculating the
  // left join and then concatenating the complementary right indices.
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");

  // We can immediately filter out cases where one table is empty. In
  // some cases, we return all the rows of the other table with a corresponding
  // null index for the empty table; in others, we return an empty output.
  if (right.num_rows() == 0) {
    switch (join_type) {
      // Left, left anti, and full all return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return left.num_rows();
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN: return 0;
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left.num_rows() == 0) {
    switch (join_type) {
      // Left, left anti, left semi, and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN: return 0;
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: return right.num_rows();
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // Prepare output column. Whether or not the output column is nullable is
  // determined by whether any of the columns in the input table are nullable.
  // If none of the input columns actually contain nulls, we can still use the
  // non-nullable version of the expression evaluation code path for
  // performance, so we capture that information as well.
  auto const has_nulls = binary_predicate.may_evaluate_null(left, right, stream);

  auto const parser =
    ast::detail::expression_parser{binary_predicate, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  auto swap_tables = (join_type == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows());
  detail::grid_1d const config(swap_tables ? right.num_rows() : left.num_rows(),
                               DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Allocate storage for the counter used to get the size of the join output
  cudf::detail::device_scalar<std::size_t> size(0, stream, mr);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  if (has_nulls) {
    compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        parser.device_expression_data,
        swap_tables,
        size.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    compute_conditional_join_output_size<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        join_type,
        parser.device_expression_data,
        swap_tables,
        size.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return size.value(stream);
}

class conditional_full_join_size_data {
 public:
  conditional_full_join_size_data(table_view left, table_view right, rmm::cuda_stream_view stream)
    : _left{left}, _right{right}, _stream{stream}
  {
  }

  conditional_full_join_size_data(table_view left,
                                  table_view right,
                                  rmm::cuda_stream_view stream,
                                  conditional_full_join_setup_data setup,
                                  std::size_t left_outer_output_size,
                                  std::size_t right_complement_size,
                                  rmm::device_uvector<size_type> matches_per_left_row,
                                  rmm::device_uvector<size_type> right_row_has_match)
    : _left{left},
      _right{right},
      _stream{stream},
      _setup{std::move(setup)},
      _left_outer_output_size{left_outer_output_size},
      _right_complement_size{right_complement_size},
      _matches_per_left_row{std::move(matches_per_left_row)},
      _right_row_has_match{std::move(right_row_has_match)}
  {
  }

  [[nodiscard]] std::size_t output_size() const
  {
    if (!_setup.has_value()) {
      return static_cast<std::size_t>(_left.num_rows()) +
             static_cast<std::size_t>(_right.num_rows());
    }
    return _left_outer_output_size + _right_complement_size;
  }

  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  materialize_indices(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(stream == _stream,
                 "conditional full join retained state must be materialized on the stream used to "
                 "create it");

    if (!_setup.has_value()) {
      if (_right.num_rows() == 0) { return get_trivial_left_join_indices(_left, stream, mr); }
      auto right_only = get_trivial_left_join_indices(_right, stream, mr);
      return std::pair(std::move(right_only.second), std::move(right_only.first));
    }

    auto& setup           = *_setup;
    auto const total_size = output_size();
    auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(total_size, stream, mr);
    auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(total_size, stream, mr);

    auto const left_matches_span = cudf::device_span<size_type const>{
      _matches_per_left_row->begin(), static_cast<std::size_t>(_left.num_rows())};
    auto left_offsets =
      rmm::device_uvector<size_type>{static_cast<std::size_t>(_left.num_rows()), stream, mr};
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                           left_matches_span.begin(),
                           left_matches_span.end(),
                           left_offsets.begin());

    if (setup.has_nulls) {
      conditional_left_join_from_size_data<DEFAULT_JOIN_BLOCK_SIZE, true>
        <<<setup.config.num_blocks,
           setup.config.num_threads_per_block,
           setup.shmem_size_per_block,
           stream.value()>>>(*setup.left_table,
                             *setup.right_table,
                             setup.parser.device_expression_data,
                             left_offsets.data(),
                             left_indices->data(),
                             right_indices->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      conditional_left_join_from_size_data<DEFAULT_JOIN_BLOCK_SIZE, false>
        <<<setup.config.num_blocks,
           setup.config.num_threads_per_block,
           setup.shmem_size_per_block,
           stream.value()>>>(*setup.left_table,
                             *setup.right_table,
                             setup.parser.device_expression_data,
                             left_offsets.data(),
                             left_indices->data(),
                             right_indices->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    if (_right_complement_size > 0) {
      auto const complement_begin = static_cast<std::ptrdiff_t>(_left_outer_output_size);
      auto left_complement_begin  = left_indices->begin() + complement_begin;
      auto right_complement_begin = right_indices->begin() + complement_begin;
      auto const right_match_span = cudf::device_span<size_type const>{
        _right_row_has_match->begin(), static_cast<std::size_t>(_right.num_rows())};

      thrust::fill(rmm::exec_policy_nosync(stream, mr),
                   left_complement_begin,
                   left_complement_begin + static_cast<std::ptrdiff_t>(_right_complement_size),
                   JoinNoMatch);
      thrust::copy_if(rmm::exec_policy_nosync(stream, mr),
                      thrust::counting_iterator<size_type>(0),
                      thrust::counting_iterator<size_type>(_right.num_rows()),
                      right_complement_begin,
                      unmatched_right_row{right_match_span});
    }

    return std::pair(std::move(left_indices), std::move(right_indices));
  }

 private:
  table_view _left;
  table_view _right;
  rmm::cuda_stream_view _stream;
  std::optional<conditional_full_join_setup_data> _setup{};
  std::size_t _left_outer_output_size = 0;
  std::size_t _right_complement_size  = 0;
  std::optional<rmm::device_uvector<size_type>> _matches_per_left_row{};
  std::optional<rmm::device_uvector<size_type>> _right_row_has_match{};
};

std::unique_ptr<conditional_full_join_size_data> compute_conditional_full_join_size_data(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (right.num_rows() == 0 || left.num_rows() == 0) {
    validate_conditional_full_join_predicate(left, right, binary_predicate, stream, mr);
    validate_conditional_full_join_materializable_size(static_cast<std::size_t>(left.num_rows()) +
                                                       static_cast<std::size_t>(right.num_rows()));
    return std::make_unique<conditional_full_join_size_data>(left, right, stream);
  }

  auto setup = setup_conditional_full_join(left, right, binary_predicate, stream, mr);

  auto matches_per_left_row =
    rmm::device_uvector<size_type>{static_cast<std::size_t>(left.num_rows()), stream, mr};
  auto right_row_has_match =
    rmm::device_uvector<size_type>{static_cast<std::size_t>(right.num_rows()), stream, mr};
  thrust::fill(
    rmm::exec_policy_nosync(stream, mr), right_row_has_match.begin(), right_row_has_match.end(), 0);

  if (setup.has_nulls) {
    compute_conditional_full_join_size_data<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<setup.config.num_blocks,
         setup.config.num_threads_per_block,
         setup.shmem_size_per_block,
         stream.value()>>>(*setup.left_table,
                           *setup.right_table,
                           setup.parser.device_expression_data,
                           matches_per_left_row.data(),
                           right_row_has_match.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    compute_conditional_full_join_size_data<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<setup.config.num_blocks,
         setup.config.num_threads_per_block,
         setup.shmem_size_per_block,
         stream.value()>>>(*setup.left_table,
                           *setup.right_table,
                           setup.parser.device_expression_data,
                           matches_per_left_row.data(),
                           right_row_has_match.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto const left_outer_output_size = thrust::reduce(rmm::exec_policy_nosync(stream, mr),
                                                     matches_per_left_row.begin(),
                                                     matches_per_left_row.end(),
                                                     std::size_t{0});
  auto const right_complement_size =
    static_cast<std::size_t>(thrust::count(rmm::exec_policy_nosync(stream, mr),
                                           right_row_has_match.begin(),
                                           right_row_has_match.end(),
                                           size_type{0}));
  validate_conditional_full_join_materializable_size(left_outer_output_size +
                                                     right_complement_size);

  return std::make_unique<conditional_full_join_size_data>(left,
                                                           right,
                                                           stream,
                                                           std::move(setup),
                                                           left_outer_output_size,
                                                           right_complement_size,
                                                           std::move(matches_per_left_row),
                                                           std::move(right_row_has_match));
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view const& left,
                       table_view const& right,
                       ast::expression const& binary_predicate,
                       std::optional<std::size_t> output_size,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(
    left, right, binary_predicate, join_kind::INNER_JOIN, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(
    left, right, binary_predicate, join_kind::LEFT_JOIN, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join(
    left, right, binary_predicate, join_kind::FULL_JOIN, {}, stream, mr);
}

conditional_full_join_size_data::~conditional_full_join_size_data() = default;

conditional_full_join_size_data::conditional_full_join_size_data(
  std::unique_ptr<cudf::detail::conditional_full_join_size_data> impl)
  : _impl{std::move(impl)}
{
}

std::size_t conditional_full_join_size_data::output_size() const { return _impl->output_size(); }

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join_size_data::materialize_indices(rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) const
{
  return _impl->materialize_indices(stream, mr);
}

std::unique_ptr<conditional_full_join_size_data> conditional_full_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::unique_ptr<conditional_full_join_size_data>{new conditional_full_join_size_data{
    detail::compute_conditional_full_join_size_data(left, right, binary_predicate, stream, mr)}};
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join_anti_semi(
    left, right, binary_predicate, join_kind::LEFT_SEMI_JOIN, output_size, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::conditional_join_anti_semi(
    left, right, binary_predicate, join_kind::LEFT_ANTI_JOIN, output_size, stream, mr);
}

std::size_t conditional_inner_join_size(table_view const& left,
                                        table_view const& right,
                                        ast::expression const& binary_predicate,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::INNER_JOIN, stream, mr);
}

std::size_t conditional_left_join_size(table_view const& left,
                                       table_view const& right,
                                       ast::expression const& binary_predicate,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::LEFT_JOIN, stream, mr);
}

std::size_t conditional_left_semi_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::size_t conditional_left_anti_join_size(table_view const& left,
                                            table_view const& right,
                                            ast::expression const& binary_predicate,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_conditional_join_output_size(
    left, right, binary_predicate, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace cudf
