#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace kzip {

struct SplitScoreResult {
   std::size_t offset{};
   double score{};
   double minimum_statistic{};
};

// Static transcription of the byte-histogram scorer at 0x405060. Runtime
// captures match on repeated-byte, full-alphabet, and biased-random fixtures.
// The target's rational log-increment approximation is retained; libm log()
// can differ at edge cases.
[[nodiscard]] SplitScoreResult score_split(std::span<const std::uint8_t> bytes);

} // namespace kzip
