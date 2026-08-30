/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>

/**
 * @file
 * @brief API for counting the number of distinct elements in a column
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief Device-memory bytes retained by the fixed-width distinct-count workspace preflight.
 */
struct distinct_count_workspace_preflight_result {
  std::size_t preprocessing_retained_bytes{};
  std::size_t static_set_storage_bytes{};
  std::size_t insertion_count_bytes{};
  std::size_t active_workspace_peak_bytes{};
};

/**
 * @addtogroup column_reduction
 * @{
 */

/**
 * @brief Count the distinct elements in the column_view.
 *
 * Given an input column_view, number of distinct elements in this column_view is returned.
 *
 * If `null_handling` is null_policy::EXCLUDE and `nan_handling` is  nan_policy::NAN_IS_NULL, both
 * `NaN` and `null` values are ignored. If `null_handling` is null_policy::EXCLUDE and
 * `nan_handling` is nan_policy::NAN_IS_VALID, only `null` is ignored, `NaN` is considered in
 * distinct count.
 *
 * `null`s are handled as equal.
 *
 * @param[in] input The column_view whose distinct elements will be counted
 * @param[in] null_handling flag to include or ignore `null` while counting
 * @param[in] nan_handling flag to consider `NaN==null` or not
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of distinct rows in the table
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Count the distinct rows in a table.
 *
 * @param[in] input Table whose distinct rows will be counted
 * @param[in] nulls_equal flag to denote if null elements should be considered equal.
 *            nulls are not equal if null_equality::UNEQUAL.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of distinct rows in the table
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal    = null_equality::EQUAL,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Compute the allocation-free workspace bound for a fixed-width global distinct count.
 *
 * This preflight only accepts one non-null, non-nested, fixed-width key column. The returned peak
 * includes the preprocessing reservation and static-set storage because both are live during
 * insertion. No device memory is allocated.
 *
 * @param[in] num_rows Number of input rows, bounded by `cudf::size_type`.
 * @param[in] key_type Type of the single key column.
 * @param[in] null_count Number of null rows; only zero is supported.
 * @param[in] key_count Number of key columns; only one is supported.
 *
 * @return The exact retained preprocessing, static-set, and active peak bytes for the supported
 * input shape.
 *
 * @throws std::invalid_argument for unsupported shape or negative arguments.
 * @throws std::overflow_error when the row count or byte accounting overflows.
 */
CUDF_EXPORT distinct_count_workspace_preflight_result distinct_count_workspace_preflight(
  std::int64_t num_rows,
  data_type key_type,
  std::int64_t null_count,
  std::int32_t key_count);

/** @} */

}  // namespace CUDF_EXPORT cudf
