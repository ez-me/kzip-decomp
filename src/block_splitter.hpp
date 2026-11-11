#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kzip {

// Block-selection options mirroring the target's CLI state:
//
//   threshold   <- -b# (default 256.0); <= 0 disables automatic splitting
//   block_count <- -n# (0 = automatic threshold mode, 1 = single block,
//                  >= 2 = explicit greedy refinement, clamped to 0x800)
struct SplitOptions {
   double threshold{256.0};
   std::size_t block_count{0};
};

// Reproduces the block-context selection at 0x400b87-0x401ea9.
//
// The returned offsets are ascending exclusive block ends and always include
// `input.size()`. With no accepted split the result is `{input.size()}`.
[[nodiscard]] std::vector<std::size_t> choose_block_ends(
   std::span<const std::uint8_t> input, const SplitOptions& options);

} // namespace kzip
