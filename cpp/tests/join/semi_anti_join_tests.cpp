/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using column_vector  = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

enum class join_implementation { FILTERED_JOIN, MARK_JOIN, MARK_JOIN_PREFILTER };

struct SemiAntiJoinTest : public cudf::test::BaseFixture,
                          public ::testing::WithParamInterface<join_implementation> {};

struct FilteredJoinProbeStateTest : public cudf::test::BaseFixture {};

struct FilteredJoinPreBuildReservationTest : public cudf::test::BaseFixture {};

namespace {
[[nodiscard]] bool use_mark_join(join_implementation implementation)
{
  return implementation != join_implementation::FILTERED_JOIN;
}

[[nodiscard]] cudf::join_prefilter prefilter_mode(join_implementation implementation)
{
  return implementation == join_implementation::MARK_JOIN_PREFILTER ? cudf::join_prefilter::YES
                                                                    : cudf::join_prefilter::NO;
}

// Helper to perform semi/anti join via either filtered_join or mark_join
std::unique_ptr<cudf::table> left_semi_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls  = cudf::null_equality::EQUAL,
  join_implementation implementation = join_implementation::FILTERED_JOIN)
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);

  if (!use_mark_join(implementation)) {
    cudf::filtered_join obj(right_selected, compare_nulls, cudf::get_default_stream());
    auto const join_indices = obj.semi_join(
      left_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  } else {
    cudf::mark_join obj(
      left_selected, compare_nulls, prefilter_mode(implementation), cudf::get_default_stream());
    auto const join_indices = obj.semi_join(
      right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  }
}

std::unique_ptr<cudf::table> left_anti_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls  = cudf::null_equality::EQUAL,
  join_implementation implementation = join_implementation::FILTERED_JOIN)
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);

  if (!use_mark_join(implementation)) {
    cudf::filtered_join obj(right_selected, compare_nulls, cudf::get_default_stream());
    auto const join_indices = obj.anti_join(
      left_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  } else {
    cudf::mark_join obj(
      left_selected, compare_nulls, prefilter_mode(implementation), cudf::get_default_stream());
    auto const join_indices = obj.anti_join(
      right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  }
}

void expect_probe_state_matches_join(cudf::table_view const& right,
                                     cudf::table_view const& left,
                                     std::vector<cudf::size_type> const& expected_semi,
                                     std::vector<cudf::size_type> const& expected_anti)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  cudf::filtered_join obj(right, cudf::null_equality::EQUAL, 0.5, stream, mr);

  auto const semi_state = obj.begin_left_semi_probe(left, stream, mr);
  EXPECT_EQ(semi_state->output_size(), expected_semi.size());
  auto const semi_indices = semi_state->materialize_indices(stream, mr);
  EXPECT_EQ(semi_indices->size(), expected_semi.size());
  EXPECT_EQ(semi_indices->capacity(), expected_semi.size());
  auto const semi_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*semi_indices}};
  column_wrapper<cudf::size_type> semi_expected(expected_semi.begin(), expected_semi.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(semi_expected, semi_col);

  auto const anti_state = obj.begin_left_anti_probe(left, stream, mr);
  EXPECT_EQ(anti_state->output_size(), expected_anti.size());
  auto const anti_indices = anti_state->materialize_indices(stream, mr);
  EXPECT_EQ(anti_indices->size(), expected_anti.size());
  EXPECT_EQ(anti_indices->capacity(), expected_anti.size());
  auto const anti_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*anti_indices}};
  column_wrapper<cudf::size_type> anti_expected(expected_anti.begin(), expected_anti.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(anti_expected, anti_col);
}

std::pair<rmm::device_buffer, cudf::size_type> make_mask(std::vector<bool> const& validity)
{
  if (std::all_of(validity.begin(), validity.end(), [](bool v) { return v; })) {
    return {rmm::device_buffer{}, 0};
  }
  auto const values = std::vector<cudf::size_type>(validity.size(), 0);
  auto col =
    column_wrapper<cudf::size_type>(values.begin(), values.end(), validity.begin()).release();
  auto const null_count = col->null_count();
  auto contents         = col->release();
  return {std::move(*contents.null_mask), null_count};
}

std::unique_ptr<cudf::column> assemble_list_column(std::unique_ptr<cudf::column> offsets,
                                                   std::unique_ptr<cudf::column> child,
                                                   cudf::size_type size,
                                                   std::vector<bool> const& validity = {})
{
  auto [null_mask, null_count] =
    validity.empty() ? std::pair{rmm::device_buffer{}, cudf::size_type{0}} : make_mask(validity);
  column_vector children;
  children.push_back(std::move(offsets));
  children.push_back(std::move(child));
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                        size,
                                        rmm::device_buffer{},
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

std::unique_ptr<cudf::column> assemble_struct_column(column_vector children,
                                                     cudf::size_type size,
                                                     std::vector<bool> const& validity = {})
{
  auto [null_mask, null_count] =
    validity.empty() ? std::pair{rmm::device_buffer{}, cudf::size_type{0}} : make_mask(validity);
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                        size,
                                        rmm::device_buffer{},
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

// Rows: "aa", null (payload "bbb"), "c": the null row keeps a non-empty payload.
std::unique_ptr<cudf::column> make_dirty_null_string_column()
{
  auto offsets               = column_wrapper<int32_t>{0, 2, 5, 6}.release();
  auto chars_contents        = column_wrapper<char>{'a', 'a', 'b', 'b', 'b', 'c'}.release()->release();
  auto [null_mask, null_count] = make_mask({true, false, true});
  column_vector children;
  children.push_back(std::move(offsets));
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                        3,
                                        std::move(*chars_contents.data),
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

std::unique_ptr<cudf::column> make_dirty_null_string_column_int64()
{
  auto offsets               = column_wrapper<int64_t>{0, 2, 5, 6}.release();
  auto chars_contents        = column_wrapper<char>{'a', 'a', 'b', 'b', 'b', 'c'}.release()->release();
  auto [null_mask, null_count] = make_mask({true, false, true});
  column_vector children;
  children.push_back(std::move(offsets));
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                        3,
                                        std::move(*chars_contents.data),
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

void expect_exact_pre_build_reservation(cudf::table_view const& right, double load_factor)
{
  auto const stream    = cudf::get_default_stream();
  auto const predicted = cudf::filtered_join::pre_build_reservation_size(
    right, load_factor, stream, cudf::get_current_device_resource_ref());

  auto measured_mr =
    rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto measured = std::size_t{0};
  {
    cudf::filtered_join obj(right,
                            cudf::null_equality::EQUAL,
                            load_factor,
                            stream,
                            rmm::device_async_resource_ref{measured_mr});
    stream.synchronize();
    measured = static_cast<std::size_t>(measured_mr.get_bytes_counter().value);
  }
  EXPECT_EQ(predicted, measured);
}
}  // namespace

TEST_F(FilteredJoinProbeStateTest, MaterializesRetainedSemiAntiProbeState)
{
  column_wrapper<int32_t> right_col{2, 4};
  column_wrapper<int32_t> left_col{1, 2, 2, 4};
  auto const right = cudf::table_view{{right_col}};
  auto const left  = cudf::table_view{{left_col}};

  expect_probe_state_matches_join(right, left, {1, 2, 3}, {0});
}

TEST_F(FilteredJoinProbeStateTest, HandlesAllMatchAndNoMatchCases)
{
  {
    column_wrapper<int32_t> right_col{1, 2, 3};
    column_wrapper<int32_t> left_col{1, 2, 3};
    expect_probe_state_matches_join(
      cudf::table_view{{right_col}}, cudf::table_view{{left_col}}, {0, 1, 2}, {});
  }
  {
    column_wrapper<int32_t> right_col{7, 8};
    column_wrapper<int32_t> left_col{1, 2, 3};
    expect_probe_state_matches_join(
      cudf::table_view{{right_col}}, cudf::table_view{{left_col}}, {}, {0, 1, 2});
  }
}

TEST_F(FilteredJoinProbeStateTest, EmptyInputsUseExactStateAndMaterializationSizes)
{
  cudf::table empty_right{};
  cudf::table empty_left{};
  column_wrapper<int32_t> nonempty_col{10, 20, 30};
  auto const nonempty = cudf::table_view{{nonempty_col}};
  auto const stream   = cudf::get_default_stream();
  auto const mr       = cudf::get_current_device_resource_ref();

  {
    cudf::filtered_join obj(empty_right, cudf::null_equality::EQUAL, 0.5, stream, mr);
    auto const anti_state = obj.begin_left_anti_probe(nonempty, stream, mr);
    EXPECT_EQ(anti_state->output_size(), nonempty.num_rows());
    EXPECT_EQ(anti_state->device_allocated_size_bytes(), 0);
    auto const anti_indices = anti_state->materialize_indices(stream, mr);
    EXPECT_EQ(anti_indices->size(), nonempty.num_rows());
    EXPECT_EQ(anti_indices->capacity(), nonempty.num_rows());
    auto const anti_col =
      cudf::column_view{cudf::device_span<cudf::size_type const>{*anti_indices}};
    column_wrapper<cudf::size_type> expected{0, 1, 2};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, anti_col);
  }
  {
    cudf::filtered_join obj(nonempty, cudf::null_equality::EQUAL, 0.5, stream, mr);
    auto const semi_state = obj.begin_left_semi_probe(empty_left, stream, mr);
    EXPECT_EQ(semi_state->output_size(), 0);
    EXPECT_EQ(semi_state->device_allocated_size_bytes(), 0);
    auto const semi_indices = semi_state->materialize_indices(stream, mr);
    EXPECT_EQ(semi_indices->size(), 0);
    EXPECT_EQ(semi_indices->capacity(), 0);
  }
}

TEST_F(FilteredJoinProbeStateTest, RetainedProbeMatchesExpectedResultsOnNestedInputs)
{
  cudf::test::lists_column_wrapper<int32_t> right_col{{1, 2}, {3}, {}};
  cudf::test::lists_column_wrapper<int32_t> left_col{{3}, {1, 2}, {9}, {}};
  auto const right = cudf::table_view{{right_col}};
  auto const left  = cudf::table_view{{left_col}};

  expect_probe_state_matches_join(right, left, {0, 1, 3}, {2});
}

TEST_F(FilteredJoinProbeStateTest, RetainedProbeStateReportsMembershipMapBytes)
{
  column_wrapper<int32_t> right_col{2, 4};
  column_wrapper<int32_t> left_col{1, 2, 2, 4};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  cudf::filtered_join obj(
    cudf::table_view{{right_col}}, cudf::null_equality::EQUAL, 0.5, stream, mr);

  auto const state = obj.begin_left_semi_probe(cudf::table_view{{left_col}}, stream, mr);
  EXPECT_EQ(state->output_size(), 3);
  EXPECT_EQ(state->device_allocated_size_bytes(), 4 * sizeof(bool));
}

TEST_F(FilteredJoinProbeStateTest, ExplicitMemoryResourceProvenance)
{
  column_wrapper<int32_t> right_col0{{1, 2, 3}, {true, false, true}};
  column_wrapper<int32_t> right_col1{{7, 8, 9}, {true, true, false}};
  column_wrapper<int32_t> left_col0{{1, 2, 1}, {true, true, true}};
  column_wrapper<int32_t> left_col1{{7, 8, 8}, {true, true, true}};
  auto const right  = cudf::table_view{{right_col0, right_col1}};
  auto const left   = cudf::table_view{{left_col0, left_col1}};
  auto const stream = cudf::get_default_stream();

  auto ambient_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto supplied_mr =
    rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  {
    cudf::test::scoped_current_device_resource current_scope{ambient_mr};
    auto const mr = rmm::device_async_resource_ref{supplied_mr};
    auto join = cudf::filtered_join(right, cudf::null_equality::UNEQUAL, 0.5, stream, mr);
    auto const semi_state = join.begin_left_semi_probe(left, stream, mr);
    auto const anti_state = join.begin_left_anti_probe(left, stream, mr);
    auto const semi       = semi_state->materialize_indices(stream, mr);
    auto const anti       = anti_state->materialize_indices(stream, mr);
    stream.synchronize();

    EXPECT_EQ(ambient_mr.get_bytes_counter().total, 0);
    EXPECT_GT(supplied_mr.get_bytes_counter().total, 0);
    EXPECT_GT(supplied_mr.get_bytes_counter().value, 0);
    EXPECT_EQ(semi_state->output_size(), 1);
    EXPECT_EQ(semi->size(), 1);
    EXPECT_EQ(anti_state->output_size(), 2);
    EXPECT_EQ(anti->size(), 2);
  }
  stream.synchronize();
  EXPECT_EQ(ambient_mr.get_bytes_counter().value, 0);
  EXPECT_EQ(supplied_mr.get_bytes_counter().value, 0);
}

TEST_F(FilteredJoinPreBuildReservationTest, RejectsInvalidInputs)
{
  column_wrapper<int32_t> col{1, 2, 3};
  auto const right  = cudf::table_view{{col}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  EXPECT_THROW((void)cudf::filtered_join::pre_build_reservation_size(
                 cudf::table_view{}, 0.5, stream, mr),
               std::invalid_argument);
  EXPECT_THROW(
    (void)cudf::filtered_join::pre_build_reservation_size(right, 0.0, stream, mr),
    std::invalid_argument);
  EXPECT_THROW(
    (void)cudf::filtered_join::pre_build_reservation_size(right, 1.5, stream, mr),
    std::invalid_argument);
}

TEST_F(FilteredJoinPreBuildReservationTest, MatchesRetainedStateForFlatInputs)
{
  column_wrapper<int32_t> col{1, 2, 3, 4, 5};
  expect_exact_pre_build_reservation(cudf::table_view{{col}}, 0.5);
  expect_exact_pre_build_reservation(cudf::table_view{{col}}, 1.0);

  column_wrapper<int32_t> empty_col{};
  expect_exact_pre_build_reservation(cudf::table_view{{empty_col}}, 0.5);

  column_wrapper<int32_t> nullable_col{{1, 2, 3, 4, 5}, {true, false, true, false, true}};
  expect_exact_pre_build_reservation(cudf::table_view{{nullable_col}}, 0.7);

  column_wrapper<int64_t> wide_col{1, 2, 3};
  column_wrapper<int32_t> second_col{4, 5, 6};
  expect_exact_pre_build_reservation(cudf::table_view{{wide_col, second_col}}, 0.5);

  strcol_wrapper strings{"alpha", "beta", "gamma"};
  expect_exact_pre_build_reservation(cudf::table_view{{strings}}, 0.5);

  strcol_wrapper nullable_strings{{"alpha", "beta", "gamma"}, {true, false, true}};
  expect_exact_pre_build_reservation(cudf::table_view{{nullable_strings}}, 0.5);

  auto const dirty_strings = make_dirty_null_string_column();
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_strings->view()}}, 0.5);
}

TEST_F(FilteredJoinPreBuildReservationTest, MatchesRetainedStateForNestedInputs)
{
  cudf::test::lists_column_wrapper<int32_t> lists{{1, 2}, {3}, {}, {4, 5, 6}};
  expect_exact_pre_build_reservation(cudf::table_view{{lists}}, 0.5);

  auto list_offsets = column_wrapper<int32_t>{0, 2, 3, 3, 6}.release();
  auto list_child   = column_wrapper<int32_t>{1, 2, 3, 4, 5, 6}.release();
  auto nullable_lists =
    assemble_list_column(std::move(list_offsets), std::move(list_child), 4, {true, false, true, true});
  expect_exact_pre_build_reservation(cudf::table_view{{nullable_lists->view()}}, 0.5);

  auto names = strcol_wrapper{{"sam", "vimes", "colon"}, {true, true, true}}.release();
  auto ages  = column_wrapper<int32_t>{{48, 27, 63}, {true, false, true}}.release();
  column_vector struct_children;
  struct_children.push_back(std::move(names));
  struct_children.push_back(std::move(ages));
  auto structs = assemble_struct_column(std::move(struct_children), 3, {true, false, true});
  expect_exact_pre_build_reservation(cudf::table_view{{structs->view()}}, 0.5);

  auto inner_offsets = column_wrapper<int32_t>{0, 2, 2, 3}.release();
  auto inner_child   = column_wrapper<int32_t>{1, 2, 3}.release();
  auto struct_list =
    assemble_list_column(std::move(inner_offsets), std::move(inner_child), 3);
  auto score = column_wrapper<int32_t>{10, 20, 30}.release();
  column_vector struct_list_children;
  struct_list_children.push_back(std::move(struct_list));
  struct_list_children.push_back(std::move(score));
  auto struct_with_list = assemble_struct_column(std::move(struct_list_children), 3);
  expect_exact_pre_build_reservation(cudf::table_view{{struct_with_list->view()}}, 0.5);

  auto member_a = column_wrapper<int32_t>{1, 2, 3, 4}.release();
  auto member_b = column_wrapper<int32_t>{10, 20, 30, 40}.release();
  column_vector inner_struct_children;
  inner_struct_children.push_back(std::move(member_a));
  inner_struct_children.push_back(std::move(member_b));
  auto inner_struct = assemble_struct_column(std::move(inner_struct_children), 4);
  auto outer_offsets = column_wrapper<int32_t>{0, 2, 2, 4}.release();
  auto list_of_structs =
    assemble_list_column(std::move(outer_offsets), std::move(inner_struct), 3);
  expect_exact_pre_build_reservation(cudf::table_view{{list_of_structs->view()}}, 0.5);

  // Nullable list rows with empty payloads: no sanitation work is retained.
  auto clean_offsets = column_wrapper<int32_t>{0, 2, 2, 4}.release();
  auto clean_child   = column_wrapper<int32_t>{1, 2, 3, 4}.release();
  auto clean_nullable_lists = assemble_list_column(
    std::move(clean_offsets), std::move(clean_child), 3, {true, false, true});
  expect_exact_pre_build_reservation(cudf::table_view{{clean_nullable_lists->view()}}, 0.5);

  // Non-nullable string child of a nullable struct adopts the parent mask with a zero null
  // count, so its non-empty payload under a null struct row never triggers sanitation.
  {
    auto adopt_offsets = column_wrapper<int32_t>{0, 2, 5, 6}.release();
    auto adopt_chars = column_wrapper<char>{'a', 'a', 'b', 'b', 'b', 'c'}.release()->release();
    column_vector adopt_string_children;
    adopt_string_children.push_back(std::move(adopt_offsets));
    auto adopt_strings = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                        3,
                                                        std::move(*adopt_chars.data),
                                                        rmm::device_buffer{},
                                                        0,
                                                        std::move(adopt_string_children));
    column_vector adopt_struct_children;
    adopt_struct_children.push_back(std::move(adopt_strings));
    auto adopt_struct =
      assemble_struct_column(std::move(adopt_struct_children), 3, {true, false, true});
    expect_exact_pre_build_reservation(cudf::table_view{{adopt_struct->view()}}, 0.5);
  }

  // Nullable string elements inside a list, with no dirty payloads.
  auto clean_strings = strcol_wrapper{{"a", "bb", "ccc"}, {true, false, true}}.release();
  auto string_list_offsets = column_wrapper<int32_t>{0, 1, 3}.release();
  auto string_lists =
    assemble_list_column(std::move(string_list_offsets), std::move(clean_strings), 2);
  expect_exact_pre_build_reservation(cudf::table_view{{string_lists->view()}}, 0.5);

  // Nested structs push null masks down two levels.
  auto deepest = column_wrapper<int32_t>{{1, 2, 3}, {true, false, true}}.release();
  column_vector mid_children;
  mid_children.push_back(std::move(deepest));
  auto mid_struct = assemble_struct_column(std::move(mid_children), 3, {true, true, false});
  column_vector outer_children;
  outer_children.push_back(std::move(mid_struct));
  auto outer_struct = assemble_struct_column(std::move(outer_children), 3, {true, false, true});
  expect_exact_pre_build_reservation(cudf::table_view{{outer_struct->view()}}, 0.5);

  auto deep_offsets = column_wrapper<int32_t>{0, 2, 4, 5}.release();
  auto deep_child   = column_wrapper<int32_t>{1, 2, 3, 4, 5}.release();
  auto inner_lists = assemble_list_column(std::move(deep_offsets), std::move(deep_child), 3);
  auto outer_list_offsets = column_wrapper<int32_t>{0, 2, 3}.release();
  auto list_of_lists =
    assemble_list_column(std::move(outer_list_offsets), std::move(inner_lists), 2);
  expect_exact_pre_build_reservation(cudf::table_view{{list_of_lists->view()}}, 0.5);
}

TEST_F(FilteredJoinPreBuildReservationTest, MatchesRetainedStateWithNonEmptyNulls)
{
  auto const dirty_strings = make_dirty_null_string_column();
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_strings->view()}}, 0.5);

  auto string_elements = strcol_wrapper{{"a", "bb", "ccc"}, {true, true, true}}.release();
  auto list_offsets    = column_wrapper<int32_t>{0, 1, 3, 3}.release();
  auto dirty_lists =
    assemble_list_column(std::move(list_offsets), std::move(string_elements), 3, {true, false, true});
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_lists->view()}}, 0.5);

  column_vector dirty_struct_children;
  dirty_struct_children.push_back(make_dirty_null_string_column());
  auto parent = assemble_struct_column(std::move(dirty_struct_children), 3);
  expect_exact_pre_build_reservation(cudf::table_view{{parent->view()}}, 0.5);

  auto const dirty_strings64 = make_dirty_null_string_column_int64();
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_strings64->view()}}, 0.5);

  // Dirty payload two list levels down: null inner list retains its element range.
  auto deep_strings = strcol_wrapper{{"a", "bb", "ccc"}, {true, true, true}}.release();
  auto inner_list_offsets = column_wrapper<int32_t>{0, 1, 3}.release();
  auto dirty_inner_lists =
    assemble_list_column(std::move(inner_list_offsets), std::move(deep_strings), 2, {true, false});
  auto outer_list_offsets = column_wrapper<int32_t>{0, 1, 2}.release();
  auto dirty_nested_lists =
    assemble_list_column(std::move(outer_list_offsets), std::move(dirty_inner_lists), 2);
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_nested_lists->view()}}, 0.5);

  // list<struct<string>> with a dirty payload under the struct: dirty detection must recurse
  // into list-borne struct children.
  column_vector string_member_children;
  string_member_children.push_back(make_dirty_null_string_column());
  auto string_structs = assemble_struct_column(std::move(string_member_children), 3);
  auto string_struct_offsets = column_wrapper<int32_t>{0, 2, 3}.release();
  auto dirty_string_struct_lists =
    assemble_list_column(std::move(string_struct_offsets), std::move(string_structs), 2);
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_string_struct_lists->view()}}, 0.5);

  // list<struct<list<int>>> with a dirty null outer row: the dropped row's null leaf element
  // still keeps the leaf mask alive under the list gather's full-span null gate.
  auto leaf_values = column_wrapper<int32_t>{{1, 2, 3, 4, 5}, {true, false, true, true, true}}.release();
  auto member_list_offsets = column_wrapper<int32_t>{0, 2, 3, 5}.release();
  auto member_lists =
    assemble_list_column(std::move(member_list_offsets), std::move(leaf_values), 3);
  column_vector member_children;
  member_children.push_back(std::move(member_lists));
  auto member_structs = assemble_struct_column(std::move(member_children), 3);
  auto outermost_offsets = column_wrapper<int32_t>{0, 1, 3}.release();
  auto dirty_struct_lists = assemble_list_column(
    std::move(outermost_offsets), std::move(member_structs), 2, {false, true});
  expect_exact_pre_build_reservation(cudf::table_view{{dirty_struct_lists->view()}}, 0.5);
}

TEST_F(FilteredJoinPreBuildReservationTest, MatchesRetainedStateForSlicedInputs)
{
  column_wrapper<int32_t> col{1, 2, 3, 4, 5, 6};
  auto const sliced = cudf::slice(col, {2, 5}).front();
  expect_exact_pre_build_reservation(cudf::table_view{{sliced}}, 0.5);

  strcol_wrapper strings{"a", "bb", "ccc", "dddd", "eeeee"};
  auto const sliced_strings = cudf::slice(strings, {1, 4}).front();
  expect_exact_pre_build_reservation(cudf::table_view{{sliced_strings}}, 0.5);

  auto list_offsets = column_wrapper<int32_t>{0, 2, 3, 5, 6}.release();
  auto list_child   = column_wrapper<int32_t>{1, 2, 3, 4, 5, 6}.release();
  auto lists = assemble_list_column(std::move(list_offsets), std::move(list_child), 4);
  auto const sliced_lists = cudf::slice(lists->view(), {1, 3}).front();
  expect_exact_pre_build_reservation(cudf::table_view{{sliced_lists}}, 0.5);

  auto const dirty = make_dirty_null_string_column();
  auto const sliced_dirty = cudf::slice(dirty->view(), {1, 3}).front();
  expect_exact_pre_build_reservation(cudf::table_view{{sliced_dirty}}, 0.5);
}

TEST_P(SemiAntiJoinTest, TestSimple)
{
  auto const implementation = GetParam();

  column_wrapper<int32_t> left_col0{0, 1, 2};
  column_wrapper<int32_t> right_col0{0, 1, 3};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result = left_semi_join(left, right, {0}, {0}, cudf::null_equality::EQUAL, implementation);
  auto sort_order = cudf::sorted_order(result->view());
  auto sorted     = cudf::gather(result->view(), *sort_order);

  column_wrapper<int32_t> expected_column{0, 1};
  auto expected = cudf::table_view{{expected_column}};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *sorted);
}

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> get_saj_tables(
  std::vector<bool> const& left_is_human_nulls, std::vector<bool> const& right_is_human_nulls)
{
  column_wrapper<int32_t> col0_0{{99, 1, 2, 0, 2}, {false, true, true, true, true}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, left_is_human_nulls.begin()};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {true, true, true, true, true}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, -99}, {true, true, true, true, false}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_names_col = strcol_wrapper{"Carrot Ironfoundersson",
                                       "Angua von Überwald",
                                       "Detritus",
                                       "Carrot Ironfoundersson",
                                       "Samuel Vimes"};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, right_is_human_nulls.begin()};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  column_vector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  return {std::make_unique<Table>(std::move(cols0)), std::make_unique<Table>(std::move(cols1))};
}

TEST_P(SemiAntiJoinTest, SemiJoinWithStructsAndNulls)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result            = left_semi_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::EQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 2}, {false, true}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, SemiJoinWithStructsAndNullsNotEqual)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result            = left_semi_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::UNEQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{2}, {true}};
  strcol_wrapper col_gold_1({"s0"}, {true});
  column_wrapper<int32_t> col_gold_2{{1}};
  auto col_gold_3_names_col = strcol_wrapper{"Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{false}, {true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNulls)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result            = left_anti_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::EQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{1, 2, 0}, {true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s0", "s4"}, {true, false, true});
  column_wrapper<int32_t> col_gold_2{{1, 2, 4}};
  auto col_gold_3_names_col = strcol_wrapper{"Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{27, 351, 31}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false, false}, {true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNullsNotEqual)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result            = left_anti_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::UNEQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 1, 2, 0}, {false, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "s0", "s4"}, {true, true, false, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4}};
  auto col_gold_3_names_col =
    strcol_wrapper{"Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31}};

  auto col_gold_3_is_human_col =
    column_wrapper<bool>{{true, true, false, false}, {true, true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNullsOnOneSide)
{
  auto const implementation = GetParam();
  auto constexpr null{0};
  auto left_col0 = [] {
    column_wrapper<int32_t> child1{{1, null}, cudf::test::iterators::null_at(1)};
    column_wrapper<int32_t> child2{11, 12};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();
  auto right_col0 = [] {
    column_wrapper<int32_t> child1{1, 2, 3, 4};
    column_wrapper<int32_t> child2{11, 12, 13, 14};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result = left_anti_join(left, right, {0}, {0}, cudf::null_equality::EQUAL, implementation);
  auto expected_indices_col = column_wrapper<cudf::size_type>{1};
  auto expected             = cudf::gather(left, expected_indices_col);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
}

TEST_P(SemiAntiJoinTest, AntiJoinEmptyTables)
{
  auto const implementation = GetParam();
  cudf::table empty_right_table{};
  cudf::table empty_left_table{};
  column_wrapper<int32_t> col{0, 1, 2};
  auto nonempty_table = cudf::table_view{{col}};
  // Empty left and right tables
  {
    auto result = left_anti_join(
      empty_left_table, empty_right_table, {}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty right table
  {
    auto result = left_anti_join(
      nonempty_table, empty_right_table, {0}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{0, 1, 2};
    auto expected             = cudf::gather(nonempty_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty left table
  {
    auto result = left_anti_join(
      empty_left_table, nonempty_table, {}, {0}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
}

TEST_P(SemiAntiJoinTest, SemiJoinEmptyTables)
{
  auto const implementation = GetParam();
  cudf::table empty_right_table{};
  cudf::table empty_left_table{};
  column_wrapper<int32_t> col{0, 1, 2};
  auto nonempty_table = cudf::table_view{{col}};
  // Empty left and right tables
  {
    auto result = left_semi_join(
      empty_left_table, empty_right_table, {}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty right table
  {
    auto result = left_semi_join(
      nonempty_table, empty_right_table, {0}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(nonempty_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty left table
  {
    auto result = left_semi_join(
      empty_left_table, nonempty_table, {}, {0}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
}

TEST_P(SemiAntiJoinTest, AntiSemiJoinLargeExtentOverflowPrevention)
{
  auto const implementation = GetParam();
  if (use_mark_join(implementation)) { GTEST_SKIP(); }
  // Test validates size_t extent can handle bucket storage sizes that would
  // overflow int32_t extent when compute_bucket_storage_size() uses low load factors
  constexpr cudf::size_type table_size = 10000000;  // 10M rows
  constexpr double load_factor         = 0.004;     // Hash table extent would be ~2.5B

  // Compile-time validation
  constexpr auto expected_bucket_size = static_cast<size_t>(table_size / load_factor);
  static_assert(expected_bucket_size > std::numeric_limits<cudf::size_type>::max(),
                "Bucket storage size should be significant");

  // Create test tables and validate semi join operations succeed
  auto const init = cudf::numeric_scalar<cudf::size_type>{0};
  auto right_col  = cudf::sequence(table_size, init, cudf::numeric_scalar<cudf::size_type>{1});

  auto right_table = cudf::table_view{{right_col->view()}};
  cudf::table empty_left_table{};

  // Test with load factors that would cause overflow in int32_t extent
  EXPECT_NO_THROW({
    cudf::filtered_join obj(
      right_table, cudf::null_equality::EQUAL, load_factor, cudf::get_default_stream());
    auto result = obj.semi_join(
      empty_left_table, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    result = obj.anti_join(
      empty_left_table, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  });
}

TEST_F(SemiAntiJoinTest, PrefilterNullableColumnsNullsEqual)
{
  column_wrapper<int32_t> left_col0{{1, 2, 3, 4, 5}, {true, false, true, true, false}};
  column_wrapper<int32_t> right_col0{{2, 3, 5, 6}, {false, true, true, true}};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto baseline = left_semi_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN);
  auto filtered = left_semi_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN_PREFILTER);

  auto baseline_order  = cudf::sorted_order(baseline->view());
  auto filtered_order  = cudf::sorted_order(filtered->view());
  auto sorted_baseline = cudf::gather(baseline->view(), *baseline_order);
  auto sorted_filtered = cudf::gather(filtered->view(), *filtered_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_baseline, *sorted_filtered);

  auto anti_baseline = left_anti_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN);
  auto anti_filtered = left_anti_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN_PREFILTER);

  auto anti_baseline_order  = cudf::sorted_order(anti_baseline->view());
  auto anti_filtered_order  = cudf::sorted_order(anti_filtered->view());
  auto anti_sorted_baseline = cudf::gather(anti_baseline->view(), *anti_baseline_order);
  auto anti_sorted_filtered = cudf::gather(anti_filtered->view(), *anti_filtered_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*anti_sorted_baseline, *anti_sorted_filtered);
}

TEST_F(SemiAntiJoinTest, MarkJoinPrefilterLoadFactorOverload)
{
  column_wrapper<int32_t> left_col0{0, 1, 1, 2, 3, 5};
  column_wrapper<int32_t> right_col0{1, 3, 3, 4};

  auto left_selected  = cudf::table_view{{left_col0}};
  auto right_selected = cudf::table_view{{right_col0}};

  cudf::mark_join obj(left_selected,
                      0.5,
                      cudf::null_equality::EQUAL,
                      cudf::join_prefilter::YES,
                      cudf::get_default_stream());

  auto const join_indices = obj.semi_join(
    right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto indices_col  = cudf::column_view{indices_span};
  auto result       = cudf::gather(left_selected, indices_col);

  column_wrapper<int32_t> expected_column{1, 1, 3};
  auto expected = cudf::table_view{{expected_column}};

  auto result_sort_order   = cudf::sorted_order(result->view());
  auto sorted_result       = cudf::gather(result->view(), *result_sort_order);
  auto expected_sort_order = cudf::sorted_order(expected);
  auto sorted_expected     = cudf::gather(expected, *expected_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_result);
}

std::string test_name(join_implementation implementation)
{
  switch (implementation) {
    case join_implementation::FILTERED_JOIN: return "FilteredJoin";
    case join_implementation::MARK_JOIN: return "MarkJoin";
    case join_implementation::MARK_JOIN_PREFILTER: return "MarkJoinPrefilter";
  }
  return "Unknown";
}

INSTANTIATE_TEST_SUITE_P(JoinImpl,
                         SemiAntiJoinTest,
                         ::testing::Values(join_implementation::FILTERED_JOIN,
                                           join_implementation::MARK_JOIN,
                                           join_implementation::MARK_JOIN_PREFILTER),
                         [](auto const& info) { return test_name(info.param); });
