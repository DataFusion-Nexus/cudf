# cmake-format: off
# Counts non-overlapping occurrences of a literal needle. `string(REGEX MATCHALL)`
# cannot be used here: matches that contain `;` are split into extra list
# elements, so `list(LENGTH)` does not report the occurrence count.
function(_cudf_count_literal haystack needle out_var)
  string(LENGTH "${needle}" needle_length)
  if(needle_length EQUAL 0)
    message(FATAL_ERROR "empty needle passed to _cudf_count_literal")
  endif()
  string(LENGTH "${haystack}" before_length)
  string(REPLACE "${needle}" "" stripped "${haystack}")
  string(LENGTH "${stripped}" after_length)
  math(EXPR count "(${before_length} - ${after_length}) / ${needle_length}")
  set(${out_var} "${count}" PARENT_SCOPE)
endfunction()

function(cudf_patch_cuco_exception_safety cuco_source_dir)
  set(impl "${cuco_source_dir}/include/cuco/detail/open_addressing/open_addressing_impl.cuh")
  file(READ "${impl}" contents)
  set(original_contents "${contents}")

  set(count_noexcept [[cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);]])
  set(count_throwing [[cuda::stream_ref stream) const
  {
    auto const num_keys = cuco::detail::distance(first, last);]])
  string(FIND "${contents}" "${count_noexcept}" count_noexcept_position)
  string(FIND "${contents}" "${count_throwing}" count_throwing_position)
  if(NOT count_noexcept_position EQUAL -1)
    string(REPLACE "${count_noexcept}" "${count_throwing}" contents "${contents}")
  elseif(count_throwing_position EQUAL -1)
    message(FATAL_ERROR "cuco count allocation boundary changed; re-audit its exception specification")
  endif()

  string(FIND "${contents}" "#include <memory>" memory_include_position)
  if(memory_include_position EQUAL -1)
    string(REPLACE "#include <cstdint>\n" "#include <cstdint>\n#include <memory>\n#include <utility>\n" contents "${contents}")
  endif()

  set(raw_num_out [[    auto d_num_out =
      reinterpret_cast<size_type*>(temp_allocator.allocate(sizeof(size_type), stream));]])
  set(owned_num_out [[    auto num_out_deleter = [temp_allocator, stream](char* ptr) mutable noexcept {
      if (ptr != nullptr) { temp_allocator.deallocate(ptr, sizeof(size_type), stream); }
    };
    auto d_num_out_owner = std::unique_ptr<char, decltype(num_out_deleter)>{
      temp_allocator.allocate(sizeof(size_type), stream), std::move(num_out_deleter)};
    auto* d_num_out = reinterpret_cast<size_type*>(d_num_out_owner.get());]])
  string(FIND "${contents}" "${raw_num_out}" raw_num_out_position)
  string(FIND "${contents}" "${owned_num_out}" owned_num_out_position)
  if(NOT raw_num_out_position EQUAL -1)
    string(REPLACE "${raw_num_out}" "${owned_num_out}" contents "${contents}")
  elseif(owned_num_out_position EQUAL -1)
    message(FATAL_ERROR "cuco d_num_out ownership changed; re-audit retrieve_all")
  endif()

  set(raw_temp_storage [[    auto d_temp_storage = temp_allocator.allocate(temp_storage_bytes, stream);]])
  set(owned_temp_storage [[    auto temp_storage_deleter =
      [temp_allocator, stream, temp_storage_bytes](char* ptr) mutable noexcept {
        if (ptr != nullptr) { temp_allocator.deallocate(ptr, temp_storage_bytes, stream); }
      };
    auto d_temp_storage_owner = std::unique_ptr<char, decltype(temp_storage_deleter)>{
      temp_allocator.allocate(temp_storage_bytes, stream), std::move(temp_storage_deleter)};
    auto* d_temp_storage = d_temp_storage_owner.get();]])
  string(FIND "${contents}" "${raw_temp_storage}" raw_temp_storage_position)
  string(FIND "${contents}" "${owned_temp_storage}" owned_temp_storage_position)
  if(NOT raw_temp_storage_position EQUAL -1)
    _cudf_count_literal("${contents}" "${raw_temp_storage}" raw_temp_storage_count)
    if(NOT raw_temp_storage_count EQUAL 2)
      message(FATAL_ERROR
        "cuco temporary-storage allocation count changed; re-audit size and retrieve_all")
    endif()
    string(REPLACE "${raw_temp_storage}" "${owned_temp_storage}" contents "${contents}")
  else()
    _cudf_count_literal("${contents}" "auto d_temp_storage_owner = std::unique_ptr<char"
      owned_temp_storage_count)
    if(NOT owned_temp_storage_count EQUAL 2)
      message(FATAL_ERROR
        "cuco temporary-storage ownership changed; re-audit size and retrieve_all")
    endif()
  endif()

  set(raw_size_release [[    temp_allocator.deallocate(d_temp_storage, temp_storage_bytes, stream);
    temp_allocator.deallocate(reinterpret_cast<char*>(d_count), sizeof(size_type), stream);]])
  set(owned_size_release [[    temp_allocator.deallocate(reinterpret_cast<char*>(d_count), sizeof(size_type), stream);]])
  string(FIND "${contents}" "${raw_size_release}" raw_size_release_position)
  string(FIND "${contents}" "${owned_size_release}" owned_size_release_position)
  if(NOT raw_size_release_position EQUAL -1)
    string(REPLACE "${raw_size_release}" "${owned_size_release}" contents "${contents}")
  elseif(owned_size_release_position EQUAL -1)
    message(FATAL_ERROR "cuco size release boundary changed; re-audit temporary-storage ownership")
  endif()

  set(raw_retrieve_releases [[    temp_allocator.deallocate(d_temp_storage, temp_storage_bytes, stream);
    temp_allocator.deallocate(reinterpret_cast<char*>(d_num_out), sizeof(size_type), stream);

]])
  string(FIND "${contents}" "${raw_retrieve_releases}" raw_retrieve_release_position)
  if(NOT raw_retrieve_release_position EQUAL -1)
    string(REPLACE "${raw_retrieve_releases}" "" contents "${contents}")
  else()
    string(FIND "${contents}" "temp_allocator.deallocate(reinterpret_cast<char*>(d_num_out)"
      stale_num_out_release_position)
    if(NOT stale_num_out_release_position EQUAL -1)
      message(FATAL_ERROR
        "cuco retrieve_all release boundary changed; re-audit temporary-storage ownership")
    endif()
  endif()
  # Post-condition: `size()` and `retrieve_all()` must each release their
  # temporary storage through exactly one path, the unique_ptr owner. Any
  # surviving explicit deallocate of the owned pointer is a double free that
  # the Nexus adaptor reports as an unknown-pointer invariant violation.
  _cudf_count_literal("${contents}" "auto d_temp_storage_owner = std::unique_ptr<char"
    final_owner_count)
  if(NOT final_owner_count EQUAL 2)
    message(FATAL_ERROR
      "cuco temporary-storage ownership incomplete after patching; expected 2 owners, found ${final_owner_count}")
  endif()
  _cudf_count_literal("${contents}" "temp_allocator.deallocate(d_temp_storage" final_raw_release_count)
  if(NOT final_raw_release_count EQUAL 0)
    message(FATAL_ERROR
      "cuco temporary storage still has ${final_raw_release_count} explicit deallocate(s) after patching; owner would double free")
  endif()

  if(NOT contents STREQUAL original_contents)
    file(WRITE "${impl}" "${contents}")
  endif()
endfunction()
# cmake-format: on
