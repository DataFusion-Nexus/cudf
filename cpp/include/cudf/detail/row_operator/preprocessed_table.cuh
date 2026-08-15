/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace row {

// forward declarations
namespace primitive {
class row_equality_comparator;

template <template <typename> class Hash>
class row_hasher;
}  // namespace primitive

namespace hash {
class row_hasher;
}  // namespace hash

namespace equality {

/**
 * @brief Preprocessed table for use with row equality comparison or row hashing
 *
 */
struct preprocessed_table {
  /**
   * @brief Factory to construct preprocessed_table for use with
   * row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   * @param mr Device memory resource used for preprocessing state.
   * @return A preprocessed table as shared pointer
   */
  static std::shared_ptr<preprocessed_table> create(
    table_view const& table,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns the exact device bytes `create` would retain for `table`.
   *
   * Covers the preprocessed table-device-view storage and any null masks or sanitized
   * columns retained by null push-down. The call only reads table metadata and allocates
   * no device memory.
   *
   * @param table The table that would be preprocessed
   * @param stream CUDA stream used to inspect null/offset metadata
   * @return Exact retained device bytes
   *
   * @throw std::invalid_argument if `table` fails the same compatibility checks as `create`
   */
  [[nodiscard]] static std::size_t create_reservation_size(table_view const& table,
                                                           rmm::cuda_stream_view stream);

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

 private:
  friend class self_comparator;
  friend class two_table_comparator;
  friend class ::cudf::detail::row::hash::row_hasher;
  friend class ::cudf::detail::row::primitive::row_equality_comparator;

  template <template <typename> class Hash>
  friend class ::cudf::detail::row::primitive::row_hasher;

  using table_device_view_owner =
    decltype(table_device_view::create(std::declval<table_view>(),
                                       std::declval<rmm::cuda_stream_view>(),
                                       std::declval<rmm::device_async_resource_ref>()));

  preprocessed_table(table_device_view_owner&& table,
                     std::vector<rmm::device_buffer>&& null_buffers,
                     std::vector<std::unique_ptr<column>>&& tmp_columns)
    : _t(std::move(table)),
      _null_buffers(std::move(null_buffers)),
      _tmp_columns(std::move(tmp_columns))
  {
  }

  table_device_view_owner _t;
  std::vector<rmm::device_buffer> _null_buffers;
  std::vector<std::unique_ptr<column>> _tmp_columns;
};

}  // namespace equality

namespace hash {

using preprocessed_table = row::equality::preprocessed_table;

}  // namespace hash
}  // namespace row
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
