/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda_runtime.h>

#include <BS_thread_pool.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::detail {

/**
 * @brief Sentinel value indicating a thread not assigned to any pool level.
 *
 * This is the initial value of `thread_pool_level` for all threads. Once a thread
 * executes its first task from a pool, `thread_pool_level` is set to that pool's level
 * and never changes back to this value.
 */
constexpr int THREAD_POOL_LEVEL_NONE = -1;
static_assert(THREAD_POOL_LEVEL_NONE == -1, "THREAD_POOL_LEVEL_NONE must be -1");

/**
 * @brief Thread-local variable indicating which pool level this thread belongs to.
 *
 * Used by `host_worker_pool()` to route tasks to the correct nesting level.
 */
CUDF_EXPORT extern thread_local int thread_pool_level;

/**
 * @brief Thread pool wrapper that marks its threads with ownership.
 *
 * This wrapper ensures that threads know which pool they belong to, enabling
 * automatic routing to the correct nesting level.
 */
class CUDF_EXPORT hierarchical_thread_pool {
  BS::thread_pool pool_;
  int level_;

 public:
  /**
   * @brief Construct a new tiered thread pool.
   *
   * @param num_threads Number of threads in the pool
   * @param level Pool level
   */
  hierarchical_thread_pool(std::size_t num_threads, int level);

  /**
   * @brief Submit task and mark the executing thread with ownership.
   *
   * When a thread first executes a task, it's marked with the pool level.
   * This ownership persists for the lifetime of the thread.
   *
   * @tparam F Callable type
   * @param task Task to execute
   * @return Future for the task result
   */
  template <typename F>
  auto submit_task(F&& task)
  {
    // Wrap task in shared_ptr so lambda can call it without being mutable.
    // This is required because BS::thread_pool stores the lambda and calls
    // it from a const context.
    auto task_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(task));

    // Capture the submitting thread's CUDA device so worker threads operate on
    // the correct device. Without this, worker threads default to device 0
    int device_id = 0;
    cudaGetDevice(&device_id);

    return pool_.submit_task([task_ptr, level = level_, device_id]() {
      // Mark this thread as owned by this pool's level (happens once per thread)
      if (thread_pool_level == THREAD_POOL_LEVEL_NONE) { thread_pool_level = level; }

      cudaSetDevice(device_id);

      return (*task_ptr)();
    });
  }

  /**
   * @brief Get the number of threads in this pool.
   */
  [[nodiscard]] std::size_t get_thread_count() const { return pool_.get_thread_count(); }
};

/**
 * @brief Drain every valid future, retaining no exception from cleanup.
 */
template <typename T>
void drain_futures(std::vector<std::future<T>>& futures) noexcept
{
  for (auto& future : futures) {
    try {
      if (future.valid()) { future.get(); }
    } catch (...) {
      // A caller that has a primary failure must not lose it to cleanup.
    }
  }
}

/**
 * @brief Keep submitted host tasks alive through scope unwinding.
 *
 * This guard covers failures while submitting tasks as well as failures while collecting them.
 */
template <typename T>
class future_drain_guard {
  std::vector<std::future<T>>& futures_;

 public:
  explicit future_drain_guard(std::vector<std::future<T>>& futures) noexcept : futures_{futures} {}

  future_drain_guard(future_drain_guard const&)            = delete;
  future_drain_guard& operator=(future_drain_guard const&) = delete;

  ~future_drain_guard() noexcept { drain_futures(futures_); }
};

/**
 * @brief Collect a group of host tasks while draining work after a failure.
 *
 * Results are consumed in submission order. If a future or the result consumer throws, all
 * remaining submitted futures are still retrieved before the first exception is rethrown. A
 * drain exception is cleanup state and is intentionally ignored so it cannot replace the primary
 * failure payload.
 *
 * @tparam T Future result type.
 * @tparam Consumer Callable receiving each result, or no argument for `void` futures.
 * @param futures Futures returned by a host worker pool.
 * @param consumer Callable invoked for successfully retrieved results.
 */
template <typename T, typename Consumer>
void get_all_futures(std::vector<std::future<T>>& futures, Consumer&& consumer)
{
  std::exception_ptr primary_failure;
  auto next = futures.begin();

  while (next != futures.end()) {
    auto current = next++;
    try {
      if constexpr (std::is_void_v<T>) {
        current->get();
        std::invoke(consumer);
      } else {
        std::invoke(consumer, current->get());
      }
    } catch (...) {
      primary_failure = std::current_exception();
      break;
    }
  }

  // Keep every worker-referenced object alive until every submitted task has completed.
  drain_futures(futures);

  if (primary_failure) { std::rethrow_exception(primary_failure); }
}

/**
 * @brief Collect host task futures when their results are not needed.
 */
template <typename T>
void get_all_futures(std::vector<std::future<T>>& futures)
{
  get_all_futures(futures, [](auto&&...) {});
}

/**
 * @brief Retrieves the appropriate thread pool based on the calling thread's context.
 *
 * The returned pool is always different from the calling thread's pool.
 *
 * @return Reference to the thread pool
 */
hierarchical_thread_pool& host_worker_pool();
}  // namespace cudf::detail
