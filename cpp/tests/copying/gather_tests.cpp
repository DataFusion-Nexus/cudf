/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/iterator>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GatherTest, cudf::test::NumericTypes);

struct GatherZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(GatherZeroColumnTest, PreservesRowCount)
{
  cudf::table_view source{std::vector<cudf::column_view>{}, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{{0, 2, 4, 1}};
  auto result = cudf::gather(source, gather_map);
  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 4);
}

TYPED_TEST(GatherTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullOdds)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather odd-valued indices
  auto map_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data  = cuda::constant_iterator{0};
  auto expect_valid = cudf::test::iterators::all_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullEvens)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather even-valued indices
  auto map_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });
  auto expect_valid = cudf::test::iterators::no_nulls();
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};

  // Every element is invalid
  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::all_nulls();

  // Create a gather map that gathers to random locations
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_map_data.begin(), host_map_data.end(), g);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{
    data, data + source_size, validity};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(host_map_data.begin(),
                                                             host_map_data.end());

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Check that the result is also all invalid
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, MultiColReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  constexpr cudf::size_type n_cols = 3;

  auto data = cuda::counting_iterator{0};
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, MultiColNulls)
{
  constexpr cudf::size_type source_size{1000};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");

  constexpr cudf::size_type n_cols = 3;

  auto data     = cuda::counting_iterator{0};
  auto validity = cudf::test::iterators::nulls_at_multiples_of(2);

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size, validity));
    source_columns.push_back(source_column_wrappers[i]);
  }

  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Expected data
  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - i - 1; });
  auto expect_valid = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

class GatherNullableTest : public cudf::test::BaseFixture {};

TEST_F(GatherNullableTest, NullableNoNulls)
{
  constexpr cudf::size_type source_size{1000};
  auto source_zero                            = cudf::make_fixed_width_scalar<int32_t>(0);
  std::unique_ptr<cudf::column> source_column = cudf::sequence(source_size, *source_zero);

  auto valid_mask = cudf::create_null_mask(source_size, cudf::mask_state::ALL_VALID);
  source_column->set_null_mask(std::move(valid_mask), 0);
  cudf::table_view source_table({source_column->view(), source_column->view()});

  auto gather_zero = cudf::make_fixed_width_scalar<int32_t>(0);

  std::unique_ptr<cudf::column> gather_map = cudf::sequence(source_size, *gather_zero);
  std::unique_ptr<cudf::table> result =
    cudf::gather(source_table, gather_map->view(), cudf::out_of_bounds_policy::DONT_CHECK);

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

namespace {

void expect_same_preflight(cudf::gather_fixed_width_dont_check_preflight_result const& lhs,
                           cudf::gather_fixed_width_dont_check_preflight_result const& rhs)
{
  EXPECT_EQ(lhs.gather_map_bytes, rhs.gather_map_bytes);
  EXPECT_EQ(lhs.output_data_bytes, rhs.output_data_bytes);
  EXPECT_EQ(lhs.output_null_mask_bytes, rhs.output_null_mask_bytes);
  EXPECT_EQ(lhs.target_mask_pointer_array_bytes, rhs.target_mask_pointer_array_bytes);
  EXPECT_EQ(lhs.source_table_device_view_bytes, rhs.source_table_device_view_bytes);
  EXPECT_EQ(lhs.valid_count_array_bytes, rhs.valid_count_array_bytes);
  EXPECT_EQ(lhs.native_temporary_workspace_bytes, rhs.native_temporary_workspace_bytes);
  EXPECT_EQ(lhs.active_phase_peak_bytes, rhs.active_phase_peak_bytes);
}

}  // namespace

TEST(GatherFixedWidthPreflight, ZeroRowsKeepNullableOwnerTemporaries)
{
  int device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  std::vector<cudf::gather_fixed_width_column_metadata> columns{
    {cudf::data_type{cudf::type_id::INT32}, false, 0},
    {cudf::data_type{cudf::type_id::FLOAT32}, true, 0},
    {cudf::data_type{cudf::type_id::INT64}, true, 0}};
  auto const result = cudf::gather_fixed_width_dont_check_preflight(0, columns, -1);

  auto const expected_pointer_bytes = columns.size() * sizeof(cudf::bitmask_type*);
  auto const expected_view_bytes =
    columns.size() * sizeof(cudf::column_device_view) + alignof(cudf::column_device_view) - 1;
  auto const expected_count_bytes = columns.size() * sizeof(cudf::size_type);
  auto const expected_temporary_bytes =
    expected_pointer_bytes + expected_view_bytes + expected_count_bytes;

  EXPECT_EQ(result.gather_map_bytes, 0);
  EXPECT_EQ(result.output_data_bytes, 0);
  EXPECT_EQ(result.output_null_mask_bytes, 0);
  EXPECT_EQ(result.target_mask_pointer_array_bytes, expected_pointer_bytes);
  EXPECT_EQ(result.source_table_device_view_bytes, expected_view_bytes);
  EXPECT_EQ(result.valid_count_array_bytes, expected_count_bytes);
  EXPECT_EQ(result.native_temporary_workspace_bytes, expected_temporary_bytes);
  EXPECT_EQ(result.active_phase_peak_bytes, expected_temporary_bytes);

  std::vector<cudf::gather_fixed_width_column_metadata> non_nullable{
    {cudf::data_type{cudf::type_id::INT32}, false, 0}};
  auto const empty_result = cudf::gather_fixed_width_dont_check_preflight(0, non_nullable, device);
  EXPECT_EQ(empty_result.active_phase_peak_bytes, 0);
  EXPECT_EQ(empty_result.native_temporary_workspace_bytes, 0);
}

TEST(GatherFixedWidthPreflight, SupportedTypesAndCheckedComponents)
{
  std::vector<cudf::gather_fixed_width_column_metadata> columns{
    {cudf::data_type{cudf::type_id::INT8}, false, 0},
    {cudf::data_type{cudf::type_id::INT32}, true, 0},
    {cudf::data_type{cudf::type_id::FLOAT64}, true, 0}};
  constexpr std::int64_t rows{7};
  auto const result = cudf::gather_fixed_width_dont_check_preflight(rows, columns, 0);

  auto const expected_map_bytes = static_cast<std::size_t>(rows) * sizeof(cudf::size_type);
  auto const expected_data_bytes =
    static_cast<std::size_t>(rows) * (sizeof(std::int8_t) + sizeof(std::int32_t) + sizeof(double));
  auto const expected_mask_bytes    = 2 * cudf::bitmask_allocation_size_bytes(rows);
  auto const expected_pointer_bytes = columns.size() * sizeof(cudf::bitmask_type*);
  auto const expected_view_bytes =
    columns.size() * sizeof(cudf::column_device_view) + alignof(cudf::column_device_view) - 1;
  auto const expected_count_bytes = columns.size() * sizeof(cudf::size_type);
  auto const expected_temporary_bytes =
    expected_pointer_bytes + expected_view_bytes + expected_count_bytes;
  auto const expected_peak =
    expected_map_bytes + expected_data_bytes + expected_mask_bytes + expected_temporary_bytes;

  EXPECT_EQ(result.gather_map_bytes, expected_map_bytes);
  EXPECT_EQ(result.output_data_bytes, expected_data_bytes);
  EXPECT_EQ(result.output_null_mask_bytes, expected_mask_bytes);
  EXPECT_EQ(result.target_mask_pointer_array_bytes, expected_pointer_bytes);
  EXPECT_EQ(result.source_table_device_view_bytes, expected_view_bytes);
  EXPECT_EQ(result.valid_count_array_bytes, expected_count_bytes);
  EXPECT_EQ(result.native_temporary_workspace_bytes, expected_temporary_bytes);
  EXPECT_EQ(result.active_phase_peak_bytes, expected_peak);
}

TEST(GatherFixedWidthPreflight, MetadataAndTableViewOverloadsAgree)
{
  cudf::test::fixed_width_column_wrapper<int32_t> first{{1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> second{{4.0F, 3.0F, 2.0F, 1.0F}, {1, 0, 1, 1}};
  cudf::table_view source({first, second});
  std::vector<cudf::gather_fixed_width_column_metadata> columns{
    {cudf::data_type{cudf::type_id::INT32}, false, 0},
    {cudf::data_type{cudf::type_id::FLOAT32}, true, 0}};

  auto const metadata_result = cudf::gather_fixed_width_dont_check_preflight(3, columns, 0);
  auto const table_result    = cudf::gather_fixed_width_dont_check_preflight(source, 3, -1);
  expect_same_preflight(metadata_result, table_result);
}

TEST(GatherFixedWidthPreflight, MetadataQueryDoesNotAllocate)
{
  std::vector<cudf::gather_fixed_width_column_metadata> columns{
    {cudf::data_type{cudf::type_id::INT32}, true, 0}};
  auto const original = cudf::get_current_device_resource_ref();
  auto target         = rmm::mr::statistics_resource_adaptor{original};

  {
    cudf::test::scoped_current_device_resource current_scope{
      cuda::mr::any_resource<cuda::mr::device_accessible>{target}};
    auto const result = cudf::gather_fixed_width_dont_check_preflight(4096, columns, -1);
    EXPECT_GT(result.active_phase_peak_bytes, 0);
    EXPECT_EQ(target.get_allocations_counter().total, 0);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
}

TEST(GatherFixedWidthPreflight, MetadataVectorAggregatesAtRowLimitDoNotAllocate)
{
  std::vector<cudf::gather_fixed_width_column_metadata> columns{
    {cudf::data_type{cudf::type_id::INT8}, true, 0},
    {cudf::data_type{cudf::type_id::INT16}, true, 0},
    {cudf::data_type{cudf::type_id::INT32}, true, 0},
    {cudf::data_type{cudf::type_id::INT64}, true, 0}};
  auto const max_rows = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  auto const original = cudf::get_current_device_resource_ref();
  auto target         = rmm::mr::statistics_resource_adaptor{original};

  {
    cudf::test::scoped_current_device_resource current_scope{
      cuda::mr::any_resource<cuda::mr::device_accessible>{target}};
    auto const result = cudf::gather_fixed_width_dont_check_preflight(
      static_cast<std::int64_t>(max_rows), columns, -1);
    auto const expected_mask_bytes =
      cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(max_rows));
    auto const expected_data_bytes    = max_rows * (sizeof(std::int8_t) + sizeof(std::int16_t) +
                                                 sizeof(std::int32_t) + sizeof(std::int64_t));
    auto const expected_masks         = columns.size() * expected_mask_bytes;
    auto const expected_pointer_bytes = columns.size() * sizeof(cudf::bitmask_type*);
    auto const expected_view_bytes =
      columns.size() * sizeof(cudf::column_device_view) + alignof(cudf::column_device_view) - 1;
    auto const expected_count_bytes = columns.size() * sizeof(cudf::size_type);
    auto const expected_temporary =
      expected_pointer_bytes + expected_view_bytes + expected_count_bytes;
    auto const expected_peak = max_rows * sizeof(cudf::size_type) + expected_data_bytes +
                               expected_masks + expected_temporary;

    EXPECT_EQ(result.output_data_bytes, expected_data_bytes);
    EXPECT_EQ(result.output_null_mask_bytes, expected_masks);
    EXPECT_EQ(result.native_temporary_workspace_bytes, expected_temporary);
    EXPECT_EQ(result.active_phase_peak_bytes, expected_peak);
    EXPECT_EQ(target.get_allocations_counter().total, 0);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
}

TEST(GatherFixedWidthPreflight, RejectsUnsupportedMetadataWithoutAllocation)
{
  auto const original = cudf::get_current_device_resource_ref();
  auto target         = rmm::mr::statistics_resource_adaptor{original};

  {
    cudf::test::scoped_current_device_resource current_scope{
      cuda::mr::any_resource<cuda::mr::device_accessible>{target}};
    std::vector<cudf::gather_fixed_width_column_metadata> columns{
      {cudf::data_type{cudf::type_id::INT32}, false, 0}};
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(-1, columns, -1), cudf::logic_error);
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(
                   std::numeric_limits<std::int64_t>::max(), columns, -1),
                 cudf::logic_error);
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(
                   4, {{cudf::data_type{cudf::type_id::DECIMAL32, -2}, false, 0}}, -1),
                 cudf::logic_error);
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(
                   4, {{cudf::data_type{cudf::type_id::LIST}, false, 1}}, -1),
                 cudf::logic_error);
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(
                   4, {{cudf::data_type{cudf::type_id::STRING}, false, 0}}, -1),
                 cudf::logic_error);
    EXPECT_THROW(cudf::gather_fixed_width_dont_check_preflight(
                   4, {{cudf::data_type{cudf::type_id::INT32}, false, -1}}, -1),
                 cudf::logic_error);
    EXPECT_EQ(target.get_allocations_counter().total, 0);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
}

TEST(GatherFixedWidthPreflight, ZeroRowNullableOwnerPeakMatchesPreflight)
{
  cudf::test::fixed_width_column_wrapper<int32_t> first{{1, 2, 3, 4}, {1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> second{{4, 3, 2, 1}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map;
  cudf::table_view source({first, second});
  auto const preflight = cudf::gather_fixed_width_dont_check_preflight(source, 0, -1);

  auto const upstream = cudf::get_current_device_resource_ref();
  auto ambient        = rmm::mr::statistics_resource_adaptor{upstream};
  auto supplied       = rmm::mr::statistics_resource_adaptor{upstream};
  auto const stream   = cudf::test::get_default_stream();

  {
    cudf::test::scoped_current_device_resource current_scope{
      cuda::mr::any_resource<cuda::mr::device_accessible>{ambient}};
    {
      auto result = cudf::gather(source,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 rmm::device_async_resource_ref{supplied});
      stream.synchronize();
      EXPECT_EQ(supplied.get_bytes_counter().peak, preflight.active_phase_peak_bytes);
      EXPECT_EQ(ambient.get_allocations_counter().total, 0);
      EXPECT_EQ(result->num_rows(), 0);
      EXPECT_EQ(result->num_columns(), 2);
    }
    stream.synchronize();
    EXPECT_EQ(supplied.get_bytes_counter().value, 0);
  }
}

TEST(GatherFixedWidthPreflight, OwnerPeakMatchesOutputAndTemporaryFacts)
{
  cudf::test::fixed_width_column_wrapper<int32_t> first{{1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> second{{4.0F, 3.0F, 2.0F, 1.0F}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> third{{4, 3, 2, 1}, {1, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{{0, 1, 2, 3}};
  cudf::table_view source({first, second, third});
  auto const preflight = cudf::gather_fixed_width_dont_check_preflight(source, 4, -1);

  auto const upstream = cudf::get_current_device_resource_ref();
  auto ambient        = rmm::mr::statistics_resource_adaptor{upstream};
  auto supplied       = rmm::mr::statistics_resource_adaptor{upstream};
  auto const stream   = cudf::test::get_default_stream();

  {
    cudf::test::scoped_current_device_resource current_scope{
      cuda::mr::any_resource<cuda::mr::device_accessible>{ambient}};
    {
      auto result = cudf::gather(source,
                                 gather_map,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 rmm::device_async_resource_ref{supplied});
      stream.synchronize();
      auto const expected_peak = preflight.active_phase_peak_bytes - preflight.gather_map_bytes;
      auto const expected_live = preflight.output_data_bytes + preflight.output_null_mask_bytes;
      EXPECT_EQ(supplied.get_bytes_counter().peak, expected_peak);
      EXPECT_EQ(supplied.get_bytes_counter().value, expected_live);
      EXPECT_EQ(ambient.get_allocations_counter().total, 0);
      CUDF_TEST_EXPECT_TABLES_EQUAL(source, result->view());
    }
    stream.synchronize();
    EXPECT_EQ(supplied.get_bytes_counter().value, 0);
  }
}
