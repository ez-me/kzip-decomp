#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kzip {

constexpr std::size_t kLiteralLengthSymbols = 286;
constexpr std::size_t kParserDistanceSymbols = 30;
constexpr std::size_t kTokenCostSymbols = 59;
constexpr std::int32_t kUnusableSymbolCost = 0x1000;
constexpr std::int32_t kInitialParseCost = std::numeric_limits<std::int32_t>::min();
constexpr std::int32_t kUnreachableParseCost = std::numeric_limits<std::int32_t>::max();

struct DeflateCodeParam {
   std::uint32_t extra_bits;
   std::uint32_t base_value;
};

struct ParserCostTable {
   // [0, 29] are distance symbols; [30, 58] are length symbols 257..285.
   std::array<DeflateCodeParam, kTokenCostSymbols> params{};
   std::array<std::uint16_t, kTokenCostSymbols> token_cost{};
   std::array<std::uint8_t, 256> length_to_symbol{};
   std::array<std::uint8_t, 32768> distance_to_symbol{};
};

// Build the same combined table layout used by KZIP's 0xD2BCE0 and
// 0xD235E0 tables. Code lengths are the active context's current Huffman
// lengths, not frequencies.
ParserCostTable make_parser_cost_table(
   const std::array<std::uint8_t, kLiteralLengthSymbols>& literal_length_bits,
   const std::array<std::uint8_t, kParserDistanceSymbols>& distance_bits);

// KZIP's parser uses signed 32-bit accumulated costs and updates only when
// the candidate is strictly cheaper. Equal-cost candidates do not replace
// the existing decision record.
std::int32_t literal_transition_cost(std::int32_t from_cost,
                                     std::uint8_t literal,
                                     const std::array<std::uint8_t, kLiteralLengthSymbols>& literal_length_bits);

std::int32_t match_transition_cost(std::int32_t from_cost,
                                   std::uint16_t match_length,
                                   std::uint16_t distance,
                                   const ParserCostTable& table);

constexpr bool should_replace_literal_cost(std::int32_t candidate, std::int32_t incumbent) noexcept {
   return candidate <= incumbent;
}

constexpr bool should_replace_match_cost(std::int32_t candidate, std::int32_t incumbent) noexcept {
   return candidate < incumbent;
}

} // namespace kzip
