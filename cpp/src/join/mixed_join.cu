/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/uninitialized_fill.h>

#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Probes the equality hash table for the given join kind.
 *
 * The hash table is built on the right equality table and probed with the left equality table,
 * yielding the index pairs that the conditional predicate is subsequently applied to.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
equality_join_indices(cudf::hash_join const& hash_joiner,
                      table_view const& left_equality,
                      join_kind join_type,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  switch (join_type) {
    case join_kind::INNER_JOIN: return hash_joiner.inner_join(left_equality, {}, stream, mr);
    case join_kind::LEFT_JOIN: return hash_joiner.left_join(left_equality, {}, stream, mr);
    case join_kind::FULL_JOIN: return hash_joiner.full_join(left_equality, {}, stream, mr);
    default: CUDF_FAIL("Invalid join kind.");
  }
}

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(table_view const& left_equality,
           table_view const& right_equality,
           table_view const& left_conditional,
           table_view const& right_conditional,
           ast::expression const& binary_predicate,
           null_equality compare_nulls,
           join_kind join_type,
           output_size_data_type const& output_size_data,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
               "Left semi and anti joins should use mixed_join_semi.");
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  // hash_join requires a non-empty build (right) table.
  if (right_conditional.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left_conditional, stream, mr);
      case join_kind::INNER_JOIN:
        return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
      default: CUDF_FAIL("Invalid join kind.");
    }
  }

  // A full join is a left join plus the unmatched-right complement. Build the left-outer result and
  // append the complement with finalize_full_join rather than splitting failed pairs, which would
  // emit spurious unmatched rows for keys that also match elsewhere.
  if (join_type == join_kind::FULL_JOIN) {
    auto left_outer = mixed_join(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_JOIN,
                                 std::nullopt,
                                 stream,
                                 mr);
    return finalize_full_join(
      std::move(left_outer), left_conditional.num_rows(), right_conditional.num_rows(), stream, mr);
  }

  auto const hash_joiner = cudf::hash_join{
    right_equality, compare_nulls, stream, cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
  auto const [left_indices, right_indices] =
    equality_join_indices(hash_joiner, left_equality, join_type, stream, mr);

  auto const output_size = output_size_data.has_value()
                             ? std::optional<std::size_t>{output_size_data->first}
                             : std::nullopt;

  return detail::filter_join_indices(left_conditional,
                                     right_conditional,
                                     *left_indices,
                                     *right_indices,
                                     binary_predicate,
                                     join_type,
                                     output_size,
                                     stream,
                                     mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_output_size(table_view const& left_equality,
                               table_view const& right_equality,
                               table_view const& left_conditional,
                               table_view const& right_conditional,
                               ast::expression const& binary_predicate,
                               null_equality compare_nulls,
                               join_kind join_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");
  CUDF_EXPECTS(
    (join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
    "Left semi and anti join size estimation should use compute_mixed_join_output_size_semi.");
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  // hash_join requires a non-empty build (right) table.
  if (right_conditional.num_rows() == 0) {
    auto const left_num_rows = left_conditional.num_rows();
    if (join_type == join_kind::LEFT_JOIN) {
      auto counts =
        rmm::device_uvector<size_type>(static_cast<std::size_t>(left_num_rows), stream, mr);
      thrust::uninitialized_fill(
        rmm::exec_policy_nosync(stream, mr), counts.begin(), counts.end(), size_type{1});
      return {static_cast<std::size_t>(left_num_rows),
              std::make_unique<rmm::device_uvector<size_type>>(std::move(counts))};
    }
    return {0, std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  }

  auto const hash_joiner = cudf::hash_join{
    right_equality, compare_nulls, stream, cuda::mr::any_resource<cuda::mr::device_accessible>{mr}};
  auto const [left_indices, right_indices] =
    equality_join_indices(hash_joiner, left_equality, join_type, stream, mr);

  return cudf::filter_join_indices_output_size(left_conditional,
                                               right_conditional,
                                               *left_indices,
                                               *right_indices,
                                               binary_predicate,
                                               join_type,
                                               stream,
                                               mr);
}

class mixed_full_join_size_data {
 public:
  mixed_full_join_size_data(table_view left_conditional,
                            table_view right_conditional,
                            rmm::cuda_stream_view stream)
    : _stream{stream},
      _output_size{static_cast<std::size_t>(left_conditional.num_rows()) +
                   static_cast<std::size_t>(right_conditional.num_rows())},
      _left_outer_output_size{right_conditional.num_rows() == 0
                                ? static_cast<std::size_t>(left_conditional.num_rows())
                                : 0},
      _right_complement_size{right_conditional.num_rows() == 0
                               ? 0
                               : static_cast<std::size_t>(right_conditional.num_rows())},
      _empty_conditionals{std::pair{left_conditional, right_conditional}}
  {
  }

  mixed_full_join_size_data(rmm::cuda_stream_view stream,
                            std::size_t left_outer_output_size,
                            std::size_t right_complement_size,
                            std::unique_ptr<rmm::device_uvector<size_type>> left_indices,
                            std::unique_ptr<rmm::device_uvector<size_type>> right_indices)
    : _stream{stream},
      _output_size{left_outer_output_size + right_complement_size},
      _left_outer_output_size{left_outer_output_size},
      _right_complement_size{right_complement_size},
      _left_indices{std::move(left_indices)},
      _right_indices{std::move(right_indices)}
  {
  }

  [[nodiscard]] std::size_t output_size() const { return _output_size; }

  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  materialize_indices(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(stream == _stream,
                 "mixed full join retained state must be materialized on the stream used to "
                 "create it");

    if (_empty_conditionals.has_value()) {
      auto const& [left_conditional, right_conditional] = *_empty_conditionals;
      if (right_conditional.num_rows() == 0) {
        return get_trivial_left_join_indices(left_conditional, stream, mr);
      }
      auto right_only = get_trivial_left_join_indices(right_conditional, stream, mr);
      return std::pair(std::move(right_only.second), std::move(right_only.first));
    }

    auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(_output_size, stream, mr);
    auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(_output_size, stream, mr);
    thrust::copy(rmm::exec_policy_nosync(stream, mr),
                 _left_indices->begin(),
                 _left_indices->end(),
                 left_indices->begin());
    thrust::copy(rmm::exec_policy_nosync(stream, mr),
                 _right_indices->begin(),
                 _right_indices->end(),
                 right_indices->begin());
    return std::pair(std::move(left_indices), std::move(right_indices));
  }

 private:
  rmm::cuda_stream_view _stream;
  std::size_t _output_size;
  std::size_t _left_outer_output_size;
  std::size_t _right_complement_size;
  std::optional<std::pair<table_view, table_view>> _empty_conditionals{};
  std::unique_ptr<rmm::device_uvector<size_type>> _left_indices;
  std::unique_ptr<rmm::device_uvector<size_type>> _right_indices;
};

std::unique_ptr<mixed_full_join_size_data> compute_mixed_full_join_size_data(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  if (right_conditional.num_rows() == 0 || left_conditional.num_rows() == 0) {
    return std::make_unique<mixed_full_join_size_data>(left_conditional, right_conditional, stream);
  }

  // Finalizing at size time retains the matched-right evidence and exact counts;
  // materialization then copies immutable retained state instead of re-evaluating the predicate
  // over borrowed inputs.
  auto left_outer                   = mixed_join(left_equality,
                               right_equality,
                               left_conditional,
                               right_conditional,
                               binary_predicate,
                               compare_nulls,
                               join_kind::LEFT_JOIN,
                               std::nullopt,
                               stream,
                               mr);
  auto const left_outer_output_size = left_outer.first->size();
  auto finalized                    = finalize_full_join(
    std::move(left_outer), left_conditional.num_rows(), right_conditional.num_rows(), stream, mr);
  return std::make_unique<mixed_full_join_size_data>(
    stream,
    left_outer_output_size,
    finalized.first->size() - left_outer_output_size,
    std::move(finalized.first),
    std::move(finalized.second));
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::INNER_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
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
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::INNER_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::LEFT_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
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
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::FULL_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

mixed_full_join_size_data::~mixed_full_join_size_data() = default;

mixed_full_join_size_data::mixed_full_join_size_data(
  std::unique_ptr<cudf::detail::mixed_full_join_size_data> impl)
  : _impl{std::move(impl)}
{
}

std::size_t mixed_full_join_size_data::output_size() const { return _impl->output_size(); }

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join_size_data::materialize_indices(rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr) const
{
  return _impl->materialize_indices(stream, mr);
}

std::unique_ptr<mixed_full_join_size_data> mixed_full_join_size(
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
  return std::unique_ptr<mixed_full_join_size_data>{
    new mixed_full_join_size_data{detail::compute_mixed_full_join_size_data(left_equality,
                                                                            right_equality,
                                                                            left_conditional,
                                                                            right_conditional,
                                                                            binary_predicate,
                                                                            compare_nulls,
                                                                            stream,
                                                                            mr)}};
}

}  // namespace cudf
