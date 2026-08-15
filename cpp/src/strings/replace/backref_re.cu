/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backref_re.cuh"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <utility>

namespace cudf {
namespace strings {

struct replace_with_backrefs_prepared {
  using ProgPtr =
    std::unique_ptr<detail::reprog_device, std::function<void(detail::reprog_device*)>>;

  ProgPtr d_prog{};
  std::optional<rmm::device_uvector<detail::backref_type>> backrefs{};
  std::unique_ptr<string_scalar> repl_scalar{};
  std::unique_ptr<column> offsets{};
  column_view input_parent{};
  rmm::cuda_stream_view prepare_stream{};
  int64_t char_bytes{0};
  size_type strings_count{0};
  bool empty{true};
  std::size_t retained_output_bytes{0};
  std::size_t retained_state_bytes{0};
  std::size_t peak_execute_workspace_bytes{0};
  std::size_t execute_reservation_required_bytes{0};
  std::size_t strided_working_bytes{0};
  std::size_t program_bytes{0};
  std::size_t backrefs_bytes{0};
  std::size_t replacement_state_bytes{0};
  std::size_t input_device_view_bytes{0};
};

namespace detail {
namespace {

/**
 * @brief Return the capturing group index pattern to use with the given replacement string.
 *
 * Only two patterns are supported at this time `\d` and `${d}` where `d` is an integer in
 * the range 0-99. The `\d` pattern is returned by default unless no `\d` pattern is found in
 * the `repl` string,
 *
 * Reference: https://www.regular-expressions.info/refreplacebackref.html
 */
std::string get_backref_pattern(std::string_view repl)
{
  std::string const backslash_pattern = "\\\\(\\d+)";
  std::string const bracket_pattern   = "\\$\\{(\\d+)\\}";
  std::string const r{repl};
  std::smatch m;
  return std::regex_search(r, m, std::regex(backslash_pattern)) ? backslash_pattern
                                                                : bracket_pattern;
}
/**
 * @brief Parse the back-ref index and position values from a given replace format.
 *
 * The back-ref numbers are expected to be 1-based.
 *
 * Returns a modified string without back-ref indicators and a vector of back-ref
 * byte position pairs. These are used by the device code to build the output
 * string by placing the captured group elements into the replace format.
 *
 * For example, for input string 'hello \2 and \1' the returned `backref_type` vector
 * contains `[(2,6),(1,11)]` and the returned string is 'hello  and '.
 */
std::pair<std::string, std::vector<backref_type>> parse_backrefs(std::string_view repl,
                                                                 int const group_count)
{
  std::vector<backref_type> backrefs;
  std::string str{repl};  // make a modifiable copy
  std::smatch m;
  std::regex ex(get_backref_pattern(repl));
  std::string rtn;
  size_type byte_offset = 0;
  while (std::regex_search(str, m, ex) && !m.empty()) {
    // parse the back-ref index number
    size_type const index = static_cast<size_type>(std::atoi(std::string{m[1]}.c_str()));
    CUDF_EXPECTS(index >= 0 && index <= group_count,
                 "Group index numbers must be in the range 0 to group count");

    // store the new byte offset and index value
    size_type const position = static_cast<size_type>(m.position(0));
    byte_offset += position;
    backrefs.push_back({index, byte_offset});

    // update the output string
    rtn += str.substr(0, position);
    // remove the back-ref pattern to continue parsing
    str = str.substr(position + static_cast<size_type>(m.length(0)));
  }
  if (!str.empty())  // add the remainder
    rtn += str;      // of the string
  return {rtn, backrefs};
}

std::size_t checked_add_size(std::size_t left, std::size_t right)
{
  CUDF_EXPECTS(left <= std::numeric_limits<std::size_t>::max() - right,
               "replace_with_backrefs size preflight overflowed size_t",
               std::overflow_error);
  return left + right;
}

std::size_t checked_mul_size(std::size_t left, std::size_t right)
{
  CUDF_EXPECTS(left == 0 || right <= std::numeric_limits<std::size_t>::max() / left,
               "replace_with_backrefs size preflight overflowed size_t",
               std::overflow_error);
  return left * right;
}

std::size_t reservation_charge(std::size_t bytes, std::size_t alignment)
{
  return rmm::align_up(bytes, alignment);
}

std::size_t strings_column_retained_bytes(size_type rows,
                                          bool nullable,
                                          int64_t char_bytes,
                                          bool use_int64_offsets)
{
  CUDF_EXPECTS(char_bytes >= 0,
               "replace_with_backrefs produced a negative character size",
               std::overflow_error);
  auto const offsets_width = use_int64_offsets ? sizeof(int64_t) : sizeof(int32_t);
  auto const offsets_count = checked_add_size(static_cast<std::size_t>(rows), 1);
  auto const offsets_bytes = checked_mul_size(offsets_width, offsets_count);
  auto total               = checked_add_size(offsets_bytes, static_cast<std::size_t>(char_bytes));
  if (nullable) { total = checked_add_size(total, cudf::bitmask_allocation_size_bytes(rows)); }
  return total;
}

std::size_t column_device_view_allocation_size(column_view const& input)
{
  if (input.num_children() == 0) { return 0; }
  return checked_add_size(alignof(column_device_view) - 1,
                          column_device_view::extent(input) - sizeof(column_device_view));
}

struct replace_with_backrefs_size_result {
  std::size_t retained_output_bytes{0};
  std::size_t temporary_workspace_bytes{0};
};

std::size_t compute_execute_workspace_bytes(std::size_t program_bytes,
                                            std::size_t backrefs_bytes,
                                            std::size_t replacement_state_bytes,
                                            std::size_t input_device_view_bytes,
                                            std::size_t strided_working_bytes)
{
  auto temporary = std::size_t{0};
  temporary      = checked_add_size(temporary, program_bytes);
  temporary      = checked_add_size(temporary, backrefs_bytes);
  temporary      = checked_add_size(temporary, replacement_state_bytes);
  temporary      = checked_add_size(temporary, input_device_view_bytes);
  return checked_add_size(temporary, strided_working_bytes);
}

std::size_t compute_execute_reservation_bytes(std::size_t input_device_view_bytes,
                                              std::size_t char_bytes,
                                              std::size_t strided_working_bytes,
                                              std::size_t null_mask_bytes)
{
  auto const device_view_charge = reservation_charge(input_device_view_bytes, alignof(char));
  auto const chars_charge       = reservation_charge(char_bytes, alignof(char));
  auto const working_charge =
    reservation_charge(strided_working_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const null_mask_charge = reservation_charge(null_mask_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  return checked_add_size(checked_add_size(device_view_charge, chars_charge),
                          std::max(working_charge, null_mask_charge));
}

void launch_size_or_write_kernel(backrefs_fn<backref_type*> size_and_exec_fn,
                                 reprog_device& d_prog,
                                 size_type strings_count,
                                 std::size_t buffer_size,
                                 int32_t thread_count,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto d_buffer = rmm::device_buffer(buffer_size, stream, mr);
  d_prog.set_working_memory(d_buffer.data(), thread_count);
  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, 256};
  for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
    size_and_exec_fn, d_prog, strings_count);
}

std::unique_ptr<replace_with_backrefs_prepared> prepare_replace_with_backrefs_impl(
  strings_column_view const& input,
  regex_program const& prog,
  std::string_view replacement,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto prepared            = std::make_unique<replace_with_backrefs_prepared>();
  prepared->strings_count  = input.size();
  prepared->input_parent   = input.parent();
  prepared->prepare_stream = stream;

  if (input.is_empty()) { return prepared; }

  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(!replacement.empty(), "Parameter replacement must not be empty");

  prepared->empty         = false;
  prepared->d_prog        = regex_device_builder::create_prog_device(prog, stream, mr);
  prepared->program_bytes = prepared->d_prog->allocation_size();

  auto const group_count  = std::min(99, prepared->d_prog->group_counts());
  auto const parse_result = parse_backrefs(replacement, group_count);
  prepared->backrefs.emplace(cudf::detail::make_device_uvector(parse_result.second, stream, mr));
  prepared->repl_scalar    = std::make_unique<string_scalar>(parse_result.first, true, stream, mr);
  prepared->backrefs_bytes = checked_mul_size(sizeof(backref_type), parse_result.second.size());
  prepared->replacement_state_bytes = checked_add_size(parse_result.first.size(), sizeof(bool));
  string_view const d_repl_template = prepared->repl_scalar->value(stream);

  prepared->input_device_view_bytes = column_device_view_allocation_size(input.parent());
  auto const d_strings = column_device_view::create_with_mr(input.parent(), stream, mr);

  auto size_and_exec_fn = backrefs_fn<backref_type*>{
    *d_strings, d_repl_template, prepared->backrefs->begin(), prepared->backrefs->end()};
  auto const strings_count = input.size();
  auto output_sizes        = rmm::device_uvector<size_type>(strings_count, stream, mr);
  size_and_exec_fn.d_sizes = output_sizes.data();

  auto [buffer_size, thread_count] =
    prepared->d_prog->compute_strided_working_memory(strings_count);
  prepared->strided_working_bytes = buffer_size;
  launch_size_or_write_kernel(
    size_and_exec_fn, *prepared->d_prog, strings_count, buffer_size, thread_count, stream, mr);

  auto [offsets, char_bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  prepared->offsets            = std::move(offsets);
  prepared->char_bytes         = char_bytes;
  auto const use_int64_offsets = prepared->offsets->type().id() == type_id::INT64;

  prepared->retained_output_bytes = strings_column_retained_bytes(
    strings_count, input.parent().nullable(), char_bytes, use_int64_offsets);

  auto retained_state = std::size_t{0};
  retained_state      = checked_add_size(retained_state, prepared->offsets->alloc_size());
  retained_state      = checked_add_size(retained_state, prepared->program_bytes);
  retained_state      = checked_add_size(retained_state, prepared->backrefs_bytes);
  prepared->retained_state_bytes =
    checked_add_size(retained_state, prepared->replacement_state_bytes);

  prepared->peak_execute_workspace_bytes =
    compute_execute_workspace_bytes(prepared->program_bytes,
                                    prepared->backrefs_bytes,
                                    prepared->replacement_state_bytes,
                                    prepared->input_device_view_bytes,
                                    prepared->strided_working_bytes);
  prepared->execute_reservation_required_bytes = compute_execute_reservation_bytes(
    prepared->input_device_view_bytes,
    static_cast<std::size_t>(prepared->char_bytes),
    prepared->strided_working_bytes,
    input.parent().nullable() ? cudf::bitmask_allocation_size_bytes(strings_count) : 0);

  return prepared;
}

std::unique_ptr<column> execute_replace_with_backrefs_impl(
  strings_column_view const& input,
  replace_with_backrefs_prepared* prepared_raw,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(prepared_raw != nullptr, "prepared replace handle is null");
  std::unique_ptr<replace_with_backrefs_prepared> prepared{prepared_raw};

  CUDF_EXPECTS(stream == prepared->prepare_stream,
               "prepared replace state must execute on the stream used by prepare");
  CUDF_EXPECTS(cudf::detail::is_shallow_equivalent(prepared->input_parent, input.parent()),
               "prepared replace state must execute with the input used by prepare");

  if (prepared->empty) { return make_empty_column(type_id::STRING); }

  string_view const d_repl_template = prepared->repl_scalar->value(stream);
  auto const d_strings  = column_device_view::create_with_mr(input.parent(), stream, mr);
  auto size_and_exec_fn = backrefs_fn<backref_type*>{
    *d_strings, d_repl_template, prepared->backrefs->begin(), prepared->backrefs->end()};
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(prepared->offsets->view());

  auto chars =
    rmm::device_uvector<char>(static_cast<std::size_t>(prepared->char_bytes), stream, mr);
  if (prepared->char_bytes > 0) {
    size_and_exec_fn.d_chars = chars.data();
    auto const [buffer_size, thread_count] =
      prepared->d_prog->compute_strided_working_memory(prepared->strings_count);
    launch_size_or_write_kernel(size_and_exec_fn,
                                *prepared->d_prog,
                                prepared->strings_count,
                                buffer_size,
                                thread_count,
                                stream,
                                mr);
  }

  auto offsets_column = std::move(prepared->offsets);
  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

replace_with_backrefs_size_result replace_with_backrefs_size_impl(strings_column_view const& input,
                                                                  regex_program const& prog,
                                                                  std::string_view replacement,
                                                                  rmm::cuda_stream_view stream,
                                                                  rmm::device_async_resource_ref mr)
{
  replace_with_backrefs_size_result result{};
  auto sizing_mr = rmm::mr::statistics_resource_adaptor(mr);
  {
    auto prepared = prepare_replace_with_backrefs_impl(
      input, prog, replacement, stream, rmm::device_async_resource_ref{sizing_mr});
    result.retained_output_bytes = prepared->retained_output_bytes;
    auto const measured_size_pass_peak =
      static_cast<std::size_t>(sizing_mr.get_bytes_counter().peak);
    result.temporary_workspace_bytes =
      std::max(measured_size_pass_peak, prepared->peak_execute_workspace_bytes);
  }
  return result;
}

}  // namespace

//
std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                              regex_program const& prog,
                                              std::string_view replacement,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  auto prepared = prepare_replace_with_backrefs_impl(input, prog, replacement, stream, mr);
  return execute_replace_with_backrefs_impl(input, prepared.release(), stream, mr);
}

void replace_with_backrefs_output_size(strings_column_view const& input,
                                       regex_program const& prog,
                                       std::string_view replacement,
                                       std::size_t& retained_output_bytes,
                                       std::size_t& temporary_workspace_bytes,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const sized          = replace_with_backrefs_size_impl(input, prog, replacement, stream, mr);
  retained_output_bytes     = sized.retained_output_bytes;
  temporary_workspace_bytes = sized.temporary_workspace_bytes;
}

replace_with_backrefs_prepared* prepare_replace_with_backrefs(strings_column_view const& input,
                                                              regex_program const& prog,
                                                              std::string_view replacement,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  return prepare_replace_with_backrefs_impl(input, prog, replacement, stream, mr).release();
}

replace_with_backrefs_prepared_output_size_facts
get_replace_with_backrefs_prepared_output_size_facts(replace_with_backrefs_prepared const& prepared)
{
  return {prepared.retained_output_bytes,
          prepared.retained_state_bytes,
          prepared.peak_execute_workspace_bytes,
          prepared.execute_reservation_required_bytes};
}

std::unique_ptr<column> execute_replace_with_backrefs(strings_column_view const& input,
                                                      replace_with_backrefs_prepared* prepared,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return execute_replace_with_backrefs_impl(input, prepared, stream, mr);
}

void destroy_replace_with_backrefs_prepared(replace_with_backrefs_prepared* prepared)
{
  delete prepared;
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& strings,
                                              regex_program const& prog,
                                              std::string_view replacement,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_with_backrefs(strings, prog, replacement, stream, mr);
}

void replace_with_backrefs_output_size(strings_column_view const& strings,
                                       regex_program const& prog,
                                       std::string_view replacement,
                                       std::size_t& retained_output_bytes,
                                       std::size_t& temporary_workspace_bytes,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  detail::replace_with_backrefs_output_size(
    strings, prog, replacement, retained_output_bytes, temporary_workspace_bytes, stream, mr);
}

replace_with_backrefs_prepared* prepare_replace_with_backrefs(strings_column_view const& strings,
                                                              regex_program const& prog,
                                                              std::string_view replacement,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::prepare_replace_with_backrefs(strings, prog, replacement, stream, mr);
}

replace_with_backrefs_prepared_output_size_facts
get_replace_with_backrefs_prepared_output_size_facts(replace_with_backrefs_prepared const& prepared)
{
  return detail::get_replace_with_backrefs_prepared_output_size_facts(prepared);
}

std::unique_ptr<column> execute_replace_with_backrefs(strings_column_view const& strings,
                                                      replace_with_backrefs_prepared* prepared,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::execute_replace_with_backrefs(strings, prepared, stream, mr);
}

void destroy_replace_with_backrefs_prepared(replace_with_backrefs_prepared* prepared)
{
  detail::destroy_replace_with_backrefs_prepared(prepared);
}

}  // namespace strings
}  // namespace cudf
