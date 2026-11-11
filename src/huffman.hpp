#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace kzip {

// Build code lengths with the same heap, tie, scaling and equal-frequency
// length sorting behaviour as kzip_build_huffman_tree at 0x403130.
//
// `frequencies` must contain nonnegative values. `max_bits` is an exclusive
// depth limit: the build succeeds only when the maximum code length is strictly
// less than max_bits. On failure the frequencies are repeatedly shifted with
// w_i = (f_i + add) >> shift, where add = 2^shift - 1.
//
// `empty_tree_flag` corresponds to the -zl121 global at 0x1fb1290. When it is
// set and no symbol is active, lengths[0] is forced to one.
std::vector<std::uint8_t> build_huffman_lengths(
   std::span<const std::int32_t> frequencies,
   std::uint32_t max_bits,
   bool empty_tree_flag = false);

// Canonical code assignment at 0x402b10. Codes are bit-reversed for the
// target's LSB-first bit writer. Zero-length symbols leave their output slot
// untouched.
void assign_canonical_codes(std::span<const std::uint8_t> lengths,
                            std::span<std::uint32_t> codes);

} // namespace kzip
