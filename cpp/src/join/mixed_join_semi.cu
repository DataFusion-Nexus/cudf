/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_filter_join_common_utils.cuh"
#include "mixed_join_kernels_semi.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <memory>
#include <optional>

namespace cudf {
namespace detail {

namespace {
struct mixed_single_gather_mask {
  join_kind kind;
  device_span<bool const> keep_mask;

  __device__ bool operator()(size_type row_index) const noexcept
  {
    return keep_mask[row_index] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};
}  // namespace

class mixed_left_semi_anti_join_size_data {
 public:
  explicit mixed_left_semi_anti_join_size_data(size_type output_size) : _output_size{output_size} {}

  mixed_left_semi_anti_join_size_data(join_kind kind,
                                      size_type left_rows,
                                      size_type output_size,
                                      rmm::device_uvector<bool> keep_mask)
    : _kind{kind},
      _left_rows{left_rows},
      _output_size{output_size},
      _keep_mask{std::move(keep_mask)}
  {
  }

  [[nodiscard]] std::size_t output_size() const { return static_cast<std::size_t>(_output_size); }

  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> materialize_indices(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    auto result = std::make_unique<rmm::device_uvector<size_type>>(
      static_cast<std::size_t>(_output_size), stream, mr);
    if (!_keep_mask.has_value()) {
      thrust::sequence(rmm::exec_policy_nosync(stream, mr), result->begin(), result->end());
      return result;
    }

    thrust::copy_if(
      rmm::exec_policy_nosync(stream, mr),
      thrust::counting_iterator<size_type>(0),
      thrust::counting_iterator<size_type>(_left_rows),
      result->begin(),
      mixed_single_gather_mask{
        _kind, device_span<bool const>(_keep_mask->data(), static_cast<std::size_t>(_left_rows))});
    return result;
  }

 private:
  join_kind _kind = join_kind::LEFT_SEMI_JOIN;
  size_type _left_rows{};
  size_type _output_size{};
  std::optional<rmm::device_uvector<bool>> _keep_mask{};
};

std::unique_ptr<mixed_left_semi_anti_join_size_data> compute_mixed_join_semi_size_data(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::INNER_JOIN) and (join_type != join_kind::LEFT_JOIN) and
                 (join_type != join_kind::FULL_JOIN),
               "Inner, left, and full joins should use mixed_join.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const outer_num_rows{left_num_rows};

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  // The retained empty states intentionally hold no device mask.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Anti returns all the row indices from left; no matches can exist.
      case join_kind::LEFT_ANTI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(left_num_rows);
      // Semi returns empty output because no matches can exist.
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(0);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Anti and semi joins both return empty sets.
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<mixed_left_semi_anti_join_size_data>(0);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) or cudf::has_nulls(right_equality) or
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& left                  = left_equality;
  auto& right                 = right_equality;
  auto left_view              = table_device_view::create(left, stream, mr);
  auto right_view             = table_device_view::create(right, stream, mr);
  auto left_conditional_view  = table_device_view::create(left_conditional, stream, mr);
  auto right_conditional_view = table_device_view::create(right_conditional, stream, mr);

  auto const preprocessed_right =
    cudf::detail::row::equality::preprocessed_table::create(right, stream, mr);
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream, mr);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_left, preprocessed_right};
  auto const equality_left = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const right_nulls    = cudf::nullate::DYNAMIC{cudf::has_nulls(right)};
  auto const row_hash_right = cudf::detail::row::hash::row_hasher{preprocessed_right};

  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  auto const row_comparator_right =
    cudf::detail::row::equality::two_table_comparator{preprocessed_right, preprocessed_right};
  auto const equality_right_equality =
    row_comparator_right.equal_to<false>(right_nulls, compare_nulls);
  auto const preprocessed_right_condtional =
    cudf::detail::row::equality::preprocessed_table::create(right_conditional, stream, mr);
  auto const row_comparator_conditional_right = cudf::detail::row::equality::two_table_comparator{
    preprocessed_right_condtional, preprocessed_right_condtional};
  auto const equality_right_conditional =
    row_comparator_conditional_right.equal_to<false>(right_nulls, compare_nulls);

  hash_set_type row_set{{static_cast<std::size_t>(right.num_rows())},
                        cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                        cuco::empty_key{JoinNoMatch},
                        {equality_right_equality, equality_right_conditional},
                        {row_hash_right.device_hasher(right_nulls)},
                        {},
                        {},
                        rmm::mr::polymorphic_allocator<char>{mr},
                        {stream.value()}};

  auto iter = cuda::counting_iterator<cudf::size_type>{0};

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(right))) {
    row_set.insert_async(iter, iter + right_num_rows, stream.value());
  } else {
    cuda::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] = cudf::detail::bitmask_and(right, stream, mr);
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    row_set.insert_if_async(iter, iter + right_num_rows, stencil, pred, stream.value());
  }

  detail::grid_1d const config(outer_num_rows * hash_set_type::cg_size, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block =
    parser.shmem_per_thread *
    cuco::detail::int_div_ceil(config.num_threads_per_block, hash_set_type::cg_size);

  auto const row_hash  = cudf::detail::row::hash::row_hasher{preprocessed_left};
  auto const hash_left = row_hash.device_hasher(has_nulls);

  hash_set_ref_type const row_set_ref = row_set.ref(cuco::contains).rebind_hash_function(hash_left);

  // Vector used to indicate indices from the left table which are present in output
  auto left_table_keep_mask =
    rmm::device_uvector<bool>(static_cast<std::size_t>(left.num_rows()), stream, mr);

  launch_mixed_join_semi(has_nulls,
                         *left_conditional_view,
                         *right_conditional_view,
                         *left_view,
                         *right_view,
                         equality_left,
                         row_set_ref,
                         cudf::device_span<bool>(left_table_keep_mask),
                         parser.device_expression_data,
                         config,
                         shmem_size_per_block,
                         stream);

  auto const output_size = static_cast<size_type>(thrust::count_if(
    rmm::exec_policy_nosync(stream, mr),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(left_num_rows),
    mixed_single_gather_mask{join_type,
                             device_span<bool const>(left_table_keep_mask.data(),
                                                     static_cast<std::size_t>(left_num_rows))}));

  return std::make_unique<mixed_left_semi_anti_join_size_data>(
    join_type, left_num_rows, output_size, std::move(left_table_keep_mask));
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_join_semi(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto state = compute_mixed_join_semi_size_data(left_equality,
                                                 right_equality,
                                                 left_conditional,
                                                 right_conditional,
                                                 binary_predicate,
                                                 compare_nulls,
                                                 join_type,
                                                 stream,
                                                 mr);
  return state->materialize_indices(stream, mr);
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_SEMI_JOIN,
                                 stream,
                                 mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_ANTI_JOIN,
                                 stream,
                                 mr);
}

mixed_left_semi_anti_join_size_data::~mixed_left_semi_anti_join_size_data() = default;

mixed_left_semi_anti_join_size_data::mixed_left_semi_anti_join_size_data(
  std::unique_ptr<cudf::detail::mixed_left_semi_anti_join_size_data> impl)
  : _impl{std::move(impl)}
{
}

std::size_t mixed_left_semi_anti_join_size_data::output_size() const
{
  return _impl->output_size();
}

std::unique_ptr<rmm::device_uvector<size_type>>
mixed_left_semi_anti_join_size_data::materialize_indices(rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr) const
{
  return _impl->materialize_indices(stream, mr);
}

std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_semi_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::unique_ptr<mixed_left_semi_anti_join_size_data>{
    new mixed_left_semi_anti_join_size_data{
      detail::compute_mixed_join_semi_size_data(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_SEMI_JOIN,
                                                stream,
                                                mr)}};
}

std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_anti_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::unique_ptr<mixed_left_semi_anti_join_size_data>{
    new mixed_left_semi_anti_join_size_data{
      detail::compute_mixed_join_semi_size_data(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_ANTI_JOIN,
                                                stream,
                                                mr)}};
}

}  // namespace cudf
