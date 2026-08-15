/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_radix.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <limits>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {

template <typename F>
struct float_pair {
  size_type s;  // index bias for sorting nan
  F f;          // actual float value to sort
};

template <typename F>
struct float_decomposer {
  __device__ cuda::std::tuple<size_type&, F&> operator()(float_pair<F>& key) const
  {
    return {key.s, key.f};
  }
};

template <typename F>
struct float_to_pair_and_seq {
  F const* fs;
  __device__ cuda::std::pair<float_pair<F>, size_type> operator()(cudf::size_type idx) const
  {
    auto const f = fs[idx];
    auto const s = (isnan(f) * (idx + 1));  // multiplier helps keep the sort stable for NaNs
    return {float_pair<F>{s, f}, idx};
  }
};

template <typename T>
std::size_t radix_sort_pairs_temp_bytes(T const* d_keys_in,
                                        T* d_keys_out,
                                        size_type const* d_values_in,
                                        size_type* d_values_out,
                                        size_type num_items,
                                        bool ascending,
                                        cudaStream_t stream)
{
  std::size_t temporary_bytes = 0;
  if (ascending) {
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temporary_bytes,
                                    d_keys_in,
                                    d_keys_out,
                                    d_values_in,
                                    d_values_out,
                                    num_items,
                                    0,
                                    sizeof(T) * 8,
                                    stream);
  } else {
    cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                              temporary_bytes,
                                              d_keys_in,
                                              d_keys_out,
                                              d_values_in,
                                              d_values_out,
                                              num_items,
                                              0,
                                              sizeof(T) * 8,
                                              stream);
  }
  return temporary_bytes;
}

template <typename T>
std::size_t radix_sort_float_pairs_temp_bytes(float_pair<T> const* d_keys_in,
                                              float_pair<T>* d_keys_out,
                                              size_type const* d_values_in,
                                              size_type* d_values_out,
                                              size_type num_items,
                                              bool ascending,
                                              cudaStream_t stream)
{
  std::size_t temporary_bytes = 0;
  auto const decomposer       = float_decomposer<T>{};
  auto const end_bit          = sizeof(float_pair<T>) * 8;
  if (ascending) {
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temporary_bytes,
                                    d_keys_in,
                                    d_keys_out,
                                    d_values_in,
                                    d_values_out,
                                    num_items,
                                    decomposer,
                                    0,
                                    end_bit,
                                    stream);
  } else {
    cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                              temporary_bytes,
                                              d_keys_in,
                                              d_keys_out,
                                              d_values_in,
                                              d_values_out,
                                              num_items,
                                              decomposer,
                                              0,
                                              end_bit,
                                              stream);
  }
  return temporary_bytes;
}

/**
 * @brief Sorts fixed-width columns using faster thrust sort
 *
 * Should not be called if `input.has_nulls()==true`
 */
struct sorted_order_radix_fn {
  column_view const& input;      // keys to sort
  mutable_column_view& indices;  // output of sort
  bool ascending;                // true for ascending sort
  rmm::cuda_stream_view stream;  // for allocation and kernel launches

  template <typename T>
  void radix_sort()
  {
    auto d_in   = input.begin<T>();
    auto output = rmm::device_uvector<T>(input.size(), stream);
    auto d_out  = output.begin();  // not returned
    auto seqs   = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     seqs.begin(),
                     seqs.end(),
                     0);
    auto dv_in  = seqs.begin();
    auto dv_out = indices.begin<cudf::size_type>();

    auto const n       = input.size();
    auto const sv      = stream.value();
    auto const end_bit = sizeof(T) * 8;

    // cub radix sort implementation is always stable
    auto tmp_bytes = radix_sort_pairs_temp_bytes(d_in, d_out, dv_in, dv_out, n, ascending, sv);
    auto tmp_stg   = rmm::device_buffer(tmp_bytes, stream);
    if (ascending) {
      cub::DeviceRadixSort::SortPairs(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, 0, end_bit, sv);
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_floating_point<T>())
  {
    auto pair_in = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_in    = pair_in.begin();
    // pair_out/d_out is not returned to the caller but used as an intermediate
    auto pair_out = rmm::device_uvector<float_pair<T>>(input.size(), stream);
    auto d_out    = pair_out.begin();
    auto vals     = rmm::device_uvector<size_type>(indices.size(), stream);
    auto dv_in    = vals.begin();
    auto dv_out   = indices.begin<cudf::size_type>();

    auto zip_out = thrust::make_zip_iterator(d_in, dv_in);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      zip_out,
                      float_to_pair_and_seq<T>{input.begin<T>()});

    auto const sv = stream.value();
    auto const n  = input.size();
    // cub radix sort implementation is always stable
    auto tmp_bytes =
      radix_sort_float_pairs_temp_bytes(d_in, d_out, dv_in, dv_out, n, ascending, sv);
    auto tmp_stg          = rmm::device_buffer(tmp_bytes, stream);
    auto const decomposer = float_decomposer<T>{};
    auto const end_bit    = sizeof(float_pair<T>) * 8;
    if (ascending) {
      cub::DeviceRadixSort::SortPairs(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(
        tmp_stg.data(), tmp_bytes, d_in, d_out, dv_in, dv_out, n, decomposer, 0, end_bit, sv);
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_chrono<T>())
  {
    using rep_type = typename T::rep;
    radix_sort<rep_type>();
  }

  template <typename T>
  void operator()()
    requires(cudf::is_fixed_width<T>() and !cudf::is_chrono<T>() and !cudf::is_floating_point<T>())
  {
    radix_sort<T>();
  }

  template <typename T>
  void operator()()
    requires(not cudf::is_fixed_width<T>())
  {
    CUDF_UNREACHABLE("invalid type for radix sort");
  }
};

std::size_t checked_byte_sum(std::size_t lhs, std::size_t rhs, char const* message)
{
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() - rhs, message);
  return lhs + rhs;
}

struct sorted_order_radix_preflight_fn {
  size_type num_rows;
  bool ascending;
  sorted_order_radix_preflight_result result;

  template <typename T>
  sorted_order_radix_preflight_result operator()()
  {
    if constexpr (cudf::is_chrono<T>()) {
      using rep_type = typename T::rep;
      fill<rep_type>();
    } else if constexpr (cudf::is_integral_not_bool<T>()) {
      fill<T>();
    } else if constexpr (cudf::is_floating_point<T>()) {
      fill<T>();
    } else {
      CUDF_UNREACHABLE("invalid type for sorted-order radix preflight");
    }
    return result;
  }

 private:
  template <typename T>
  void fill()
  {
    auto const row_count = static_cast<std::size_t>(num_rows);
    auto const key_bytes =
      row_count * sizeof(std::conditional_t<cudf::is_floating_point<T>(), float_pair<T>, T>);
    result.key_input_bytes  = cudf::is_floating_point<T>() ? key_bytes : 0;
    result.key_output_bytes = key_bytes;
    result.sequence_bytes   = row_count * sizeof(size_type);
    auto const stream       = cudf::get_default_stream().value();
    if constexpr (cudf::is_floating_point<T>()) {
      result.temporary_workspace_bytes = radix_sort_float_pairs_temp_bytes<T>(
        nullptr, nullptr, nullptr, nullptr, num_rows, ascending, stream);
    } else {
      result.temporary_workspace_bytes = radix_sort_pairs_temp_bytes<T>(
        nullptr, nullptr, nullptr, nullptr, num_rows, ascending, stream);
    }
    result.retained_order_bytes    = result.sequence_bytes;
    result.active_phase_peak_bytes = checked_byte_sum(
      result.key_input_bytes, result.key_output_bytes, "sorted-order radix byte count overflowed");
    result.active_phase_peak_bytes = checked_byte_sum(result.active_phase_peak_bytes,
                                                      result.sequence_bytes,
                                                      "sorted-order radix byte count overflowed");
    result.active_phase_peak_bytes = checked_byte_sum(result.active_phase_peak_bytes,
                                                      result.temporary_workspace_bytes,
                                                      "sorted-order radix byte count overflowed");
    result.active_phase_peak_bytes = checked_byte_sum(result.active_phase_peak_bytes,
                                                      result.retained_order_bytes,
                                                      "sorted-order radix byte count overflowed");
  }
};

sorted_order_radix_preflight_result sorted_order_radix_preflight_impl(std::int64_t num_rows,
                                                                      data_type key_type,
                                                                      std::int64_t null_count,
                                                                      std::int32_t key_count,
                                                                      bool stable,
                                                                      order key_order,
                                                                      std::int32_t device)
{
  CUDF_EXPECTS(num_rows >= 0, "sorted-order radix preflight rows must be non-negative");
  CUDF_EXPECTS(num_rows <= std::numeric_limits<size_type>::max(),
               "sorted-order radix preflight rows exceed cudf::size_type");
  CUDF_EXPECTS(null_count == 0, "sorted-order radix preflight requires a non-null key");
  CUDF_EXPECTS(key_count == 1, "sorted-order radix preflight requires exactly one key");
  CUDF_EXPECTS(!stable, "sorted-order radix preflight supports only unstable sorting");
  CUDF_EXPECTS(key_order == order::ASCENDING || key_order == order::DESCENDING,
               "sorted-order radix preflight received an invalid key order");
  CUDF_EXPECTS(device >= 0, "sorted-order radix preflight device must be non-negative");
  CUDF_EXPECTS(cudf::is_integral_not_bool(key_type) || cudf::is_chrono(key_type) ||
                 cudf::is_floating_point(key_type),
               "sorted-order radix preflight key type is unsupported");
  if (num_rows == 0) { return {}; }

  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device}};
  sorted_order_radix_preflight_fn fn{static_cast<size_type>(num_rows),
                                     key_order == order::ASCENDING};
  return cudf::type_dispatcher<dispatch_storage_type>(key_type, fn);
}
}  // namespace

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for most primitive types.
 *
 * @param input Column to sort
 * @param indices The result of the sort
 * @param ascending Sort order
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void sorted_order_radix(column_view const& input,
                        mutable_column_view& indices,
                        bool ascending,
                        rmm::cuda_stream_view stream)
{
  cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), sorted_order_radix_fn{input, indices, ascending, stream});
}
}  // namespace detail

sorted_order_radix_preflight_result sorted_order_radix_preflight(std::int64_t num_rows,
                                                                 data_type key_type,
                                                                 std::int64_t null_count,
                                                                 std::int32_t key_count,
                                                                 bool stable,
                                                                 order key_order,
                                                                 std::int32_t device)
{
  return detail::sorted_order_radix_preflight_impl(
    num_rows, key_type, null_count, key_count, stable, key_order, device);
}
}  // namespace cudf
