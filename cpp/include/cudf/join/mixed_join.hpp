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

#include <optional>
#include <utility>

/**
 * @file
 * @brief APIs for joining tables using a combination of equality and conditional predicates
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

namespace detail {
class mixed_left_semi_anti_join_size_data;
}  // namespace detail

/**
 * @brief Retained exact output-size data for a mixed left semi or left anti join.
 *
 * This object owns the device-side left-row selection mask computed by
 * `mixed_left_semi_join_size` or `mixed_left_anti_join_size`, exposing the exact
 * output row count and later materializing an exact-sized left gather map
 * without probing the mixed join again. Materialization reads only the retained
 * mask; it does not access the input tables or re-evaluate the predicate.
 *
 * The retained device allocations are stream-ordered on the stream passed to
 * the size function, so `materialize_indices` must be called on that same
 * stream.
 */
class mixed_left_semi_anti_join_size_data {
 public:
  mixed_left_semi_anti_join_size_data() = delete;
  ~mixed_left_semi_anti_join_size_data();
  mixed_left_semi_anti_join_size_data(mixed_left_semi_anti_join_size_data const&) = delete;
  mixed_left_semi_anti_join_size_data(mixed_left_semi_anti_join_size_data&&)      = delete;
  mixed_left_semi_anti_join_size_data& operator=(mixed_left_semi_anti_join_size_data const&) =
    delete;
  mixed_left_semi_anti_join_size_data& operator=(mixed_left_semi_anti_join_size_data&&) = delete;

  /**
   * @brief Returns the exact number of selected left rows.
   */
  [[nodiscard]] std::size_t output_size() const;

  /**
   * @brief Materializes an exact-sized left gather map from retained state.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches;
   * must equal the stream passed to the size function that created this state
   * @param mr Device memory resource used to allocate the returned indices
   *
   * @return A vector of selected left-row indices with exactly `output_size()` rows.
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> materialize_indices(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

 private:
  explicit mixed_left_semi_anti_join_size_data(
    std::unique_ptr<cudf::detail::mixed_left_semi_anti_join_size_data> impl);

  std::unique_ptr<cudf::detail::mixed_left_semi_anti_join_size_data> _impl;

  friend std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_semi_join_size(
    table_view const& left_equality,
    table_view const& right_equality,
    table_view const& left_conditional,
    table_view const& right_conditional,
    ast::expression const& binary_predicate,
    null_equality compare_nulls,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
  friend std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_anti_join_size(
    table_view const& left_equality,
    table_view const& right_equality,
    table_view const& left_conditional,
    table_view const& right_conditional,
    ast::expression const& binary_predicate,
    null_equality compare_nulls,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};

/**
 * @brief Type alias for output size data used in mixed joins.
 *
 * This type represents an optional pair containing:
 * - The exact output size of the join operation
 * - A device span of per-row match counts for each row in the larger input table
 */
using output_size_data_type = std::optional<std::pair<std::size_t, device_span<size_type const>>>;

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_inner_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed inner join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(table_view const& left_equality,
                 table_view const& right_equality,
                 table_view const& left_conditional,
                 table_view const& right_conditional,
                 ast::expression const& binary_predicate,
                 null_equality compare_nulls            = null_equality::EQUAL,
                 output_size_data_type output_size_data = {},
                 rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables,
 * or null matches for rows in left that have no match in right.
 *
 * The first returned vector contains the row indices from the left
 * tables that have a match in the right tables (in unspecified order).
 * The corresponding value in the second returned vector is either (1)
 * the row index of the matched row from the right tables, or (2) an
 * unspecified out-of-bounds value.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_left_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed left join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls            = null_equality::EQUAL,
                output_size_data_type output_size_data = {},
                rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables,
 * or null matches for rows in either pair of tables that have no matches in
 * the other pair.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_full_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed full join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls            = null_equality::EQUAL,
                output_size_data_type output_size_data = {},
                rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left tables
 * where the columns of the equality table are equal and the predicate
 * evaluates to true on the conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right), the
 * left row is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {1}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector of indices from the left table that have matches in the right table.
 */
std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes and retains exact output-size data for a mixed left semi join.
 *
 * The returned state owns the left-row selection mask; it reports the exact
 * number of selected rows and materializes the gather map without probing the
 * mixed join again.
 *
 * @throw cudf::logic_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the paired equality/conditional row counts do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for the retained state allocations
 *
 * @return Retained mixed left semi join size data.
 */
std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_semi_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Returns an index vector corresponding to all rows in the left tables
 * for which there is no row in the right tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right), the
 * left row is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {0, 2}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector of indices from the left table that do not have matches in the right table.
 */
std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes and retains exact output-size data for a mixed left anti join.
 *
 * The returned state owns the left-row selection mask; it reports the exact
 * number of selected rows and materializes the gather map without probing the
 * mixed join again.
 *
 * @throw cudf::logic_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the paired equality/conditional row counts do not match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for the retained state allocations
 *
 * @return Retained mixed left anti join size data.
 */
std::unique_ptr<mixed_left_semi_anti_join_size_data> mixed_left_anti_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * mixed inner join between the specified tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right),
 * that pair is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair containing the size that would result from performing the
 * requested join and the number of matches for each row in one of the two
 * tables. Which of the two tables is an implementation detail and should not
 * be relied upon, simply passed to the corresponding `mixed_inner_join` API as
 * is.
 */
std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * mixed left join between the specified tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right),
 * that pair is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair containing the size that would result from performing the
 * requested join and the number of matches for each row in one of the two
 * tables. Which of the two tables is an implementation detail and should not
 * be relied upon, simply passed to the corresponding `mixed_left_join` API as
 * is.
 */
std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
