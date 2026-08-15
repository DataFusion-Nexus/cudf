/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/parquet/parquet_common.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

namespace {

/**
 * @brief Dispatches the fetch task for each source index and collects the results
 *
 * Dispatches sequentially or using host worker pool depending on the number of sources.
 *
 * @tparam Task Callable invocable as `fetch_task(std::size_t source_idx)`
 * @param num_sources Number of sources to process
 * @param fetch_task Task to run for each source index
 * @return Vector of results, one per source, in source order
 */
template <typename Task>
auto dispatch_fetch_tasks(std::size_t num_sources, Task fetch_task)
{
  using result_type = std::invoke_result_t<Task, std::size_t>;

  auto constexpr parallel_threshold = 32;

  std::vector<result_type> results;
  results.reserve(num_sources);

  if (num_sources < parallel_threshold) {
    // Run sequentially to avoid task dispatch overhead
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_sources),
                  [&](std::size_t source_idx) { results.emplace_back(fetch_task(source_idx)); });
  } else {
    // Dispatch the tasks to the host worker pool
    std::vector<std::future<result_type>> tasks;
    tasks.reserve(num_sources);
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_sources),
                  [&](std::size_t source_idx) {
                    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
                      [&fetch_task, source_idx]() { return fetch_task(source_idx); }));
                  });
    std::transform(tasks.begin(), tasks.end(), std::back_inserter(results), [](auto& task) {
      return task.get();
    });
  }
  return results;
}

/**
 * @copydoc cudf::io::parquet::fetch_footers_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  // Look up runtime configuration once, as late as possible.
  auto const metadata_size_hint = cudf::io::parquet::metadata_size_hint();
  // Helper to fetch footer from a datasource
  auto const fetch_footer = [metadata_size_hint](cudf::io::datasource& datasource) {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);
    size_t const len          = datasource.size();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");

    auto const speculative_read_size =
      std::min(len, std::max(metadata_size_hint, static_cast<size_t>(ender_len)));
    auto const speculative_read_offset = len - speculative_read_size;

    auto speculative_buffer = datasource.host_read(speculative_read_offset, speculative_read_size);
    CUDF_EXPECTS(speculative_buffer->size() == speculative_read_size,
                 "Failed to read Parquet speculative metadata bytes");

    auto const ender = reinterpret_cast<file_ender_s const*>(
      speculative_buffer->data() + speculative_buffer->size() - ender_len);

    if (speculative_read_offset == 0) {
      auto const header = reinterpret_cast<file_header_s const*>(speculative_buffer->data());
      CUDF_EXPECTS(header->magic == detail::parquet_magic, "Corrupted header");
    }

    CUDF_EXPECTS(ender->magic == detail::parquet_magic, "Corrupted footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    auto const footer_offset = len - ender->footer_len - ender_len;
    if (footer_offset >= speculative_read_offset) {
      // fastpath: the speculative read includes the full footer.
      auto const footer_start_offset = footer_offset - speculative_read_offset;
      CUDF_EXPECTS(footer_start_offset + ender->footer_len <= speculative_buffer->size(),
                   "Speculative metadata read did not include full footer bytes");
      std::vector<uint8_t> footer_bytes(ender->footer_len);
      std::memcpy(
        footer_bytes.data(), speculative_buffer->data() + footer_start_offset, ender->footer_len);
      return cudf::io::datasource::buffer::create(std::move(footer_bytes));
    }

    // The speculative read only got part of the footer. Read the missing prefix, then stitch.
    auto const missing_prefix_size = speculative_read_offset - footer_offset;
    auto missing_prefix            = datasource.host_read(footer_offset, missing_prefix_size);
    CUDF_EXPECTS(missing_prefix->size() == missing_prefix_size,
                 "Failed to read the missing footer prefix bytes");
    std::vector<uint8_t> footer_bytes(ender->footer_len);
    std::memcpy(footer_bytes.data(), missing_prefix->data(), missing_prefix_size);
    auto const footer_suffix_size = ender->footer_len - missing_prefix_size;
    std::memcpy(
      footer_bytes.data() + missing_prefix_size, speculative_buffer->data(), footer_suffix_size);
    return cudf::io::datasource::buffer::create(std::move(footer_bytes));
  };

  return dispatch_fetch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_footer(datasources[source_idx].get());
  });
}

/**
 * @copydoc cudf::io::parquet::fetch_page_indexes_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_EXPECTS(datasources.size() == page_index_bytes_per_source.size(),
               "Encountered mismatch in number of datasources and page index byte ranges");

  // Helper to fetch page index bytes from a datasource
  auto const fetch_page_index = [](cudf::io::datasource& datasource,
                                   cudf::io::text::byte_range_info const& page_index_bytes) {
    CUDF_EXPECTS(
      page_index_bytes.offset() >= 0 and
        std::cmp_less_equal(page_index_bytes.offset() + page_index_bytes.size(), datasource.size()),
      std::format("Invalid page index byte range: offset={}, size={}, datasource_size={}",
                  page_index_bytes.offset(),
                  page_index_bytes.size(),
                  datasource.size()),
      std::out_of_range);
    return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
  };

  return dispatch_fetch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_page_index(datasources[source_idx].get(), page_index_bytes_per_source[source_idx]);
  });
}

using device_spans_per_source_type = std::vector<cudf::device_span<uint8_t const>>;

struct coalesced_range {
  std::size_t source_idx;
  std::size_t offset;
  std::size_t size;
  std::size_t buffer_offset;
  std::size_t ordered_range_begin;
  std::size_t ordered_range_end;
};

std::size_t parquet_remote_coalesce_gap_bytes()
{
  constexpr auto default_gap = std::size_t{64} * 1024;
  auto const* raw            = std::getenv("LIBCUDF_PARQUET_REMOTE_COALESCE_GAP_BYTES");
  if (raw == nullptr) { return default_gap; }

  std::string_view const text{raw};
  if (text.empty() || std::any_of(text.begin(), text.end(), [](char const value) {
        return value < '0' || value > '9';
      })) {
    return default_gap;
  }

  std::size_t parsed{};
  auto const result = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return default_gap; }
  return parsed;
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<device_spans_per_source_type>,
           std::future<void>>
fetch_byte_ranges_to_device_async_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::host_span<cudf::io::text::byte_range_info const> const>
    byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static std::mutex host_read_mutex;
  static std::mutex device_read_mutex;

  auto const num_sources = datasources.size();

  CUDF_EXPECTS(num_sources == byte_ranges_per_source.size(),
               "Encountered mismatch in number of datasources and the number of byte range spans");

  // Total number of byte ranges across all sources
  auto const total_byte_ranges =
    std::accumulate(byte_ranges_per_source.begin(),
                    byte_ranges_per_source.end(),
                    std::size_t{0},
                    [](auto acc, auto const& ranges) { return acc + ranges.size(); });

  auto checked_add = [](std::size_t left, std::size_t right, char const* message) {
    CUDF_EXPECTS(
      left <= std::numeric_limits<std::size_t>::max() - right, message, std::overflow_error);
    return left + right;
  };

  using normalized_range = std::pair<std::size_t, std::size_t>;
  std::vector<std::vector<normalized_range>> normalized_ranges(num_sources);
  std::vector<std::vector<std::size_t>> range_buffer_offsets(num_sources);
  std::vector<std::size_t> physical_range_order;
  std::vector<coalesced_range> read_schedule;
  std::vector<std::size_t> source_buffer_sizes(num_sources, 0);
  physical_range_order.reserve(total_byte_ranges);
  read_schedule.reserve(total_byte_ranges);

  auto const coalesce_gap_bytes = parquet_remote_coalesce_gap_bytes();
  for (std::size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    auto const& byte_ranges = byte_ranges_per_source[source_idx];
    auto& ranges            = normalized_ranges[source_idx];
    auto& buffer_offsets    = range_buffer_offsets[source_idx];
    ranges.reserve(byte_ranges.size());
    buffer_offsets.resize(byte_ranges.size());

    for (auto const& range : byte_ranges) {
      CUDF_EXPECTS(range.offset() >= 0, "Parquet byte range offset must be non-negative");
      CUDF_EXPECTS(range.size() >= 0, "Parquet byte range size must be non-negative");
      auto const offset = static_cast<std::size_t>(range.offset());
      auto const size   = static_cast<std::size_t>(range.size());
      checked_add(offset, size, "Parquet byte range endpoint overflowed size_t");
      ranges.emplace_back(offset, size);
    }

    auto const ordered_begin = physical_range_order.size();
    for (std::size_t range_idx = 0; range_idx < ranges.size(); ++range_idx) {
      if (ranges[range_idx].second != 0) { physical_range_order.emplace_back(range_idx); }
    }
    auto const ordered_end = physical_range_order.size();
    std::stable_sort(
      physical_range_order.begin() + ordered_begin,
      physical_range_order.begin() + ordered_end,
      [&](std::size_t lhs, std::size_t rhs) { return ranges[lhs].first < ranges[rhs].first; });

    for (std::size_t order_idx = ordered_begin; order_idx < ordered_end;) {
      auto const first_range_idx = physical_range_order[order_idx];
      auto const group_offset    = ranges[first_range_idx].first;
      auto group_end             = checked_add(group_offset,
                                   ranges[first_range_idx].second,
                                   "Parquet coalesced range endpoint overflowed size_t");
      auto next_order_idx        = order_idx + 1;

      while (next_order_idx < ordered_end) {
        auto const next_range_idx = physical_range_order[next_order_idx];
        auto const next_offset    = ranges[next_range_idx].first;
        if (next_offset < group_end || next_offset - group_end > coalesce_gap_bytes) { break; }
        group_end = checked_add(next_offset,
                                ranges[next_range_idx].second,
                                "Parquet coalesced range endpoint overflowed size_t");
        ++next_order_idx;
      }

      auto const group_size    = group_end - group_offset;
      auto const buffer_offset = source_buffer_sizes[source_idx];
      source_buffer_sizes[source_idx] =
        checked_add(buffer_offset, group_size, "Parquet source buffer size overflowed size_t");
      read_schedule.push_back(
        {source_idx, group_offset, group_size, buffer_offset, order_idx, next_order_idx});
      order_idx = next_order_idx;
    }
  }

  for (auto const& range : read_schedule) {
    auto const& ranges   = normalized_ranges[range.source_idx];
    auto& buffer_offsets = range_buffer_offsets[range.source_idx];
    for (auto order_idx = range.ordered_range_begin; order_idx < range.ordered_range_end;
         ++order_idx) {
      auto const range_idx   = physical_range_order[order_idx];
      auto const range_delta = ranges[range_idx].first - range.offset;
      buffer_offsets[range_idx] =
        checked_add(range.buffer_offset,
                    range_delta,
                    "Parquet coalesced range buffer offset overflowed size_t");
    }
  }

  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.reserve(num_sources);
  for (auto const buffer_size : source_buffer_sizes) {
    column_chunk_buffers.emplace_back(
      cudf::util::round_up_safe(buffer_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
      stream,
      mr);
  }

  std::vector<device_spans_per_source_type> column_chunk_data_per_source(num_sources);
  for (std::size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    auto const& ranges         = normalized_ranges[source_idx];
    auto const& buffer_offsets = range_buffer_offsets[source_idx];
    auto& column_chunk_data    = column_chunk_data_per_source[source_idx];
    column_chunk_data.reserve(ranges.size());
    auto* const buffer_data = static_cast<uint8_t*>(column_chunk_buffers[source_idx].data());
    for (std::size_t range_idx = 0; range_idx < ranges.size(); ++range_idx) {
      auto const size        = ranges[range_idx].second;
      auto const* range_data = size == 0 ? buffer_data : buffer_data + buffer_offsets[range_idx];
      column_chunk_data.emplace_back(range_data, size);
    }
  }

  using host_read_buffer = std::unique_ptr<cudf::io::datasource::buffer>;

  // Vectors to hold futures from datasource
  std::vector<std::future<size_t>> device_read_tasks{};
  std::vector<std::future<host_read_buffer>> host_read_tasks{};
  device_read_tasks.reserve(read_schedule.size());
  host_read_tasks.reserve(read_schedule.size());

  // Vectors to store intermediate host buffers and relevant pointers
  std::vector<host_read_buffer> host_buffers{};
  std::vector<void const*> copy_srcs{};
  std::vector<void*> copy_dsts{};
  std::vector<size_t> copy_sizes{};
  copy_dsts.reserve(read_schedule.size());
  copy_sizes.reserve(read_schedule.size());

  auto destination_for = [&](coalesced_range const& range) {
    return static_cast<uint8_t*>(column_chunk_buffers[range.source_idx].data()) +
           range.buffer_offset;
  };

  // Schedule host reads holding the `host_read_mutex` so that all reads for a caller thread
  // are scheduled without interleaving with reads from other threads yielding better pipelining
  {
    std::scoped_lock<std::mutex> lock(host_read_mutex);

    for (auto const& range : read_schedule) {
      auto& datasource = datasources[range.source_idx].get();
      if (not datasource.is_device_read_preferred(range.size)) {
        // Asynchronously read column chunk data to a host buffer
        host_read_tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
          [&datasource, offset = range.offset, size = range.size]() -> host_read_buffer {
            return datasource.host_read(offset, size);
          }));
        copy_dsts.push_back(static_cast<void*>(destination_for(range)));
        copy_sizes.push_back(range.size);
      }
    }
  }

  // Complete host reads
  if (not host_read_tasks.empty()) {
    copy_srcs.reserve(host_read_tasks.size());
    host_buffers.reserve(host_read_tasks.size());

    for (auto& task : host_read_tasks) {
      host_buffers.emplace_back(task.get());
      copy_srcs.push_back(host_buffers.back().get()->data());
    }
  }

  // `device_read_async` is not guaranteed to follow stream-ordering (see datasource API docs)
  stream.synchronize();

  // Schedule device reads holding the `device_read_mutex` so that all reads for a caller thread
  // are scheduled without interleaving with reads from other threads yielding better pipelining
  {
    std::scoped_lock<std::mutex> lock(device_read_mutex);

    for (auto const& range : read_schedule) {
      auto& datasource = datasources[range.source_idx].get();
      // Directly read the column chunk data to the device buffer if supported
      if (datasource.is_device_read_preferred(range.size)) {
        device_read_tasks.emplace_back(
          datasource.device_read_async(range.offset, range.size, destination_for(range), stream));
      }
    }

    // Schedule a batched memcpy from host buffers to device
    if (not host_buffers.empty()) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
        copy_dsts.data(), copy_srcs.data(), copy_sizes.data(), copy_dsts.size(), stream));
    }
  }

  // Synchronize stream if `memcpy_batch_async` was called to safely discard the host buffers
  if (not host_buffers.empty()) { stream.synchronize(); }

  auto sync_function = [](decltype(device_read_tasks) device_read_tasks) {
    for (auto& task : device_read_tasks) {
      task.get();
    }
  };
  return {std::move(column_chunk_buffers),
          std::move(column_chunk_data_per_source),
          std::async(std::launch::deferred, sync_function, std::move(device_read_tasks))};
}

}  // namespace

[[nodiscard]] std::size_t metadata_size_hint()
{
  static constexpr auto default_metadata_size_hint = std::size_t{64} * 1024;
  return cudf::detail::getenv_or<std::size_t>("LIBCUDF_PARQUET_METADATA_SIZE_HINT",
                                              default_metadata_size_hint);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  CUDF_FUNC_RANGE();
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  auto footer_buffers = fetch_footers_to_host_impl({datasources.data(), datasources.size()});
  return std::move(footer_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  CUDF_FUNC_RANGE();
  return fetch_footers_to_host_impl(datasources);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::io::text::byte_range_info, 1> page_index_bytes_per_source{page_index_bytes};

  auto page_index_buffers = fetch_page_indexes_to_host_impl(
    {datasources.data(), datasources.size()},
    {page_index_bytes_per_source.data(), page_index_bytes_per_source.size()});
  return std::move(page_index_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_FUNC_RANGE();
  return fetch_page_indexes_to_host_impl(datasources, page_index_bytes_per_source);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(cudf::io::datasource& datasource,
                                  std::span<cudf::io::text::byte_range_info const> byte_ranges,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::host_span<cudf::io::text::byte_range_info const>, 1> byte_ranges_per_source{
    cudf::host_span<cudf::io::text::byte_range_info const>{byte_ranges.data(), byte_ranges.size()}};

  auto [buffers, fetched_byte_ranges, fut] = fetch_byte_ranges_to_device_async_impl(
    {datasources.data(), datasources.size()},
    {byte_ranges_per_source.data(), byte_ranges_per_source.size()},
    stream,
    mr);

  return {std::move(buffers), std::move(fetched_byte_ranges.front()), std::move(fut)};
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<std::vector<cudf::device_span<uint8_t const>>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<std::vector<cudf::io::text::byte_range_info> const> byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Convert input vectors into host spans for the implementation
  std::vector<cudf::host_span<cudf::io::text::byte_range_info const>> byte_range_spans_per_source;
  byte_range_spans_per_source.reserve(byte_ranges_per_source.size());
  for (auto const& ranges : byte_ranges_per_source) {
    byte_range_spans_per_source.emplace_back(ranges);
  }
  return fetch_byte_ranges_to_device_async_impl(
    datasources,
    {byte_range_spans_per_source.data(), byte_range_spans_per_source.size()},
    stream,
    mr);
}

}  // namespace cudf::io::parquet
