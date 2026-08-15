/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <utility>

/**
 * @file
 * @brief APIs for conditional joins based on an arbitrary AST predicate expression.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

namespace detail {
class conditional_full_join_size_data;
}  // namespace detail

/**
 * @brief Retained exact output-size data for a conditional full join.
 *
 * This object owns the device-side state needed to observe the exact full-join
 * output row count and later materialize exact-sized left and right gather maps
 * without recomputing the output size.
 *
 * The original input column buffers and predicate inputs used to create this
 * object must remain alive and unmodified until the retained state is
 * destroyed: materialization re-evaluates the predicate over the borrowed
 * buffers and emits rows at offsets computed from the retained size data.
 */
class conditional_full_join_size_data {
 public:
  conditional_full_join_size_data() = delete;
  ~conditional_full_join_size_data();
  conditional_full_join_size_data(conditional_full_join_size_data const&)            = delete;
  conditional_full_join_size_data(conditional_full_join_size_data&&)                 = delete;
  conditional_full_join_size_data& operator=(conditional_full_join_size_data const&) = delete;
  conditional_full_join_size_data& operator=(conditional_full_join_size_data&&)      = delete;

  /**
   * @brief Returns the exact number of rows produced by the conditional full join.
   */
  [[nodiscard]] std::size_t output_size() const;

  /**
   * @brief Materializes exact-sized left and right gather maps from retained state.
   *
   * The retained state owns stream-ordered device allocations created by
   * `conditional_full_join_size`. To preserve asynchronous lifetime safety, this
   * method must be called on the same stream used to create the retained state.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches;
   * must equal the stream passed to `conditional_full_join_size`
   * @param mr Device memory resource used to allocate the returned gather maps
   *
   * @return A pair of vectors [`left_indices`, `right_indices`] with exactly
   * `output_size()` rows each.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  materialize_indices(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

 private:
  explicit conditional_full_join_size_data(
    std::unique_ptr<cudf::detail::conditional_full_join_size_data> impl);

  std::unique_ptr<cudf::detail::conditional_full_join_size_data> _impl;

  friend std::unique_ptr<conditional_full_join_size_data> conditional_full_join_size(
    table_view const& left,
    table_view const& right,
    ast::expression const& binary_predicate,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional inner join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view const& left,
                       table_view const& right,
                       ast::expression const& binary_predicate,
                       std::optional<std::size_t> output_size = {},
                       rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true,
 * or null matches for rows in left that have no match in right.
 *
 * The first returned vector contains all the row indices from the left
 * table (in unspecified order). The corresponding value in the
 * second returned vector is either (1) the row index of the matched row
 * from the right table, if there is a match or (2) an unspecified
 * out-of-bounds value.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{0, 1, 2}, {None, 0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional left join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      std::optional<std::size_t> output_size = {},
                      rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true,
 * or null matches for rows in either table that have no match in the other.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{0, 1, 2, None}, {None, 0, 1, 2}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional full join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes and retains exact output-size data for a conditional full join.
 *
 * The returned object owns device-side conditional full join size state. It can
 * report the exact output size and later materialize exact-sized left and right
 * gather maps under explicit execution overrides.
 *
 * The returned state must be materialized on the same stream passed to this
 * function.
 *
 * The original input column buffers and predicate inputs used to create this
 * object must remain alive and unmodified until the retained state is
 * destroyed.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 * @throw std::overflow_error if the exact output size exceeds `cudf::size_type`.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for retained state allocations
 *
 * @return Retained conditional full join size data.
 */
std::unique_ptr<conditional_full_join_size_data> conditional_full_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Returns an index vector corresponding to all rows in the left table
 * for which there exists some row in the right table where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {1, 2}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {1}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct the result of
 * performing a conditional left semi join between two tables `left` and
 * `right` .
 */
std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size = {},
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left table
 * for which there does not exist any row in the right table where the
 * predicate evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {0}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {0, 2}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct the result of
 * performing a conditional left anti join between two tables `left` and
 * `right` .
 */
std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size = {},
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional inner join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_inner_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left semi join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_semi_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left anti join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_anti_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
