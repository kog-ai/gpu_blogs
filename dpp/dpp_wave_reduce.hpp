#include <bit>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <hip/hip_runtime.h>

template <int ctrl>
__device__ __forceinline__
auto dpp(auto v) requires(sizeof(decltype(v)) == 4) {
  return std::bit_cast<decltype(v)>(
      __builtin_amdgcn_mov_dpp(std::bit_cast<int>(v), ctrl, 0xf, 0xf, true));
}

// Row-wide inclusive scan using Data Parallel Primitives, result is in lane 0
// of each row (ie. lanes 0/16/32/48)
template <typename T, typename Op>
__device__ __forceinline__ T dpp_row_scan(T val, Op op) {
  val = op(val, dpp<0x101>(val)); // row_shl:1
  val = op(val, dpp<0x102>(val)); // row_shl:2
  val = op(val, dpp<0x104>(val)); // row_shl:4
  val = op(val, dpp<0x108>(val)); // row_shl:8
  return val;
}

// Wavefront-wide inclusive scan using Data Parallel Primitives, result is in
// lane 63 Note that for correct inclusive scan output some banks/rows would
// need to be masked out, but this prevents LLVM from merging the DPP modifier
// into the ALU instruction.
template <typename T, typename Op>
__device__ __forceinline__ T dpp_wave_scan(T val, Op op) {
  val = op(val, dpp<0x111>(val)); // row_shr:1
  val = op(val, dpp<0x112>(val)); // row_shr:2
  val = op(val, dpp<0x114>(val)); // row_shr:4
  val = op(val, dpp<0x118>(val)); // row_shr:8
  val = op(val, dpp<0x142>(val)); // row_bcast:15
  val = op(val, dpp<0x143>(val)); // row_bcast:31
  return val;
}

// Broadcast lane 63 to the wavefront
template <typename T>
__device__ __forceinline__
T dpp_wave_broadcast63(T val) {
  return std::bit_cast<T>(
      __builtin_amdgcn_readlane(std::bit_cast<int>(val), 63));
}
// Wavefront-wide inclusive sum
template <typename T>
__device__ __forceinline__
T dpp_wave_scan_sum(T val) {
  return dpp_wave_scan(val, std::plus<T>{});
}

// Wavefront-wide sum reduction
template <typename T>
__device__ __forceinline__
T dpp_wave_sum(T val) {
  return dpp_wave_broadcast63(dpp_wave_scan_sum(val));
}
