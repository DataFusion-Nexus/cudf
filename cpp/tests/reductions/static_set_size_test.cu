/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/reduction.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuco/static_set.cuh>

#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <atomic>
#include <mutex>
#include <numeric>
#include <unordered_set>
#include <vector>

class StaticSetSizeAllocationTest : public cudf::test::BaseFixture {};

TEST_F(StaticSetSizeAllocationTest, HistogramReleasesEachAllocationExactlyOnce)
{
  rmm::mr::cuda_memory_resource upstream;
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> deallocations{0};
  std::atomic<std::size_t> unknown_deallocations{0};
  std::mutex live_mutex;
  std::unordered_set<void*> live_allocations;
  rmm::mr::callback_memory_resource counting{
    [&upstream, &allocations, &live_mutex, &live_allocations](
      std::size_t bytes, rmm::cuda_stream_view stream, void*) {
      auto* ptr = upstream.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      allocations.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard lock{live_mutex};
      live_allocations.insert(ptr);
      return ptr;
    },
    [&upstream, &deallocations, &unknown_deallocations, &live_mutex, &live_allocations](
      void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void*) {
      deallocations.fetch_add(1, std::memory_order_relaxed);
      bool known;
      {
        std::lock_guard lock{live_mutex};
        known = live_allocations.erase(ptr) == 1;
      }
      if (known) {
        upstream.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      } else {
        unknown_deallocations.fetch_add(1, std::memory_order_relaxed);
      }
    }};

  std::vector<int32_t> input_values(4096);
  std::iota(input_values.begin(), input_values.end(), 0);
  cudf::test::fixed_width_column_wrapper<int32_t> input(input_values.begin(), input_values.end());

  {
    cudf::test::scoped_current_device_resource current{counting};
    auto const aggregation = cudf::make_histogram_aggregation<cudf::reduce_aggregation>();
    auto result = cudf::reduce(input, *aggregation, cudf::data_type{cudf::type_id::INT64});
    ASSERT_NE(result, nullptr);

    using set_type = cuco::static_set<int32_t,
                                      cuco::extent<std::size_t>,
                                      cuda::thread_scope_device,
                                      cuda::std::equal_to<int32_t>,
                                      cuco::linear_probing<1, cuco::default_hash_function<int32_t>>,
                                      rmm::mr::polymorphic_allocator<char>>;
    auto const stream = cudf::get_default_stream();
    set_type empty_set{cuco::extent<std::size_t>{4096},
                       cuco::empty_key<int32_t>{-1},
                       {},
                       {},
                       {},
                       {},
                       rmm::mr::polymorphic_allocator<char>{counting},
                       stream.value()};
    EXPECT_EQ(empty_set.size(stream), 0);
  }

  cudf::get_default_stream().synchronize();
  EXPECT_GT(allocations.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(deallocations.load(std::memory_order_relaxed),
            allocations.load(std::memory_order_relaxed));
  EXPECT_EQ(unknown_deallocations.load(std::memory_order_relaxed), 0);
  EXPECT_TRUE(live_allocations.empty());
}
