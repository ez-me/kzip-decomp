#include "parser_cost.hpp"

#include <algorithm>
#include <stdexcept>

namespace kzip {
namespace {

constexpr std::array<std::uint16_t, 29> kLengthBase = {
   3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27,
   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};
constexpr std::array<std::uint8_t, 29> kLengthExtra = {
   0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};
constexpr std::array<std::uint16_t, 30> kDistanceBase = {
   1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
   193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
   6145, 8193, 12289, 16385, 24577
};
constexpr std::array<std::uint8_t, 30> kDistanceExtra = {
   0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
   6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};

// The length and distance symbol maps depend only on the DEFLATE base tables,
// not on the active code lengths, so the target builds them once at startup.
// Build them at compile time here; recomputing the 32,768-entry distance map
// with a rescanning loop on every optimization pass dominated the runtime for
// archives with many small members.
constexpr std::array<std::uint8_t, 256> make_length_to_symbol() {
   std::array<std::uint8_t, 256> out{};
   for (std::size_t length = 3; length <= 258; ++length) {
      std::size_t code = 0;
      while (code + 1 < kLengthBase.size() &&
             length >= kLengthBase[code + 1]) {
         ++code;
      }
      out[length - 3] = static_cast<std::uint8_t>(30 + code);
   }
   return out;
}

constexpr std::array<std::uint8_t, 32768> make_distance_to_symbol() {
   std::array<std::uint8_t, 32768> out{};
   for (std::size_t distance = 1; distance <= 32768; ++distance) {
      std::size_t code = 0;
      while (code + 1 < kDistanceBase.size() &&
             distance >= kDistanceBase[code + 1]) {
         ++code;
      }
      out[distance - 1] = static_cast<std::uint8_t>(code);
   }
   return out;
}

constexpr auto kLengthToSymbol = make_length_to_symbol();
constexpr auto kDistanceToSymbol = make_distance_to_symbol();

} // namespace

ParserCostTable make_parser_cost_table(
   const std::array<std::uint8_t, kLiteralLengthSymbols>& literal_length_bits,
   const std::array<std::uint8_t, kParserDistanceSymbols>& distance_bits) {
   ParserCostTable out{};

   for (std::size_t i = 0; i < kParserDistanceSymbols; ++i) {
      out.params[i] = {kDistanceExtra[i], kDistanceBase[i]};
      out.token_cost[i] = distance_bits[i] == 0
         ? static_cast<std::uint16_t>(kUnusableSymbolCost)
         : static_cast<std::uint16_t>(distance_bits[i] + kDistanceExtra[i]);
   }

   for (std::size_t i = 0; i < kLengthBase.size(); ++i) {
      const std::size_t table_index = 30 + i;
      const std::size_t symbol = 257 + i;
      out.params[table_index] = {kLengthExtra[i], kLengthBase[i]};
      out.token_cost[table_index] = literal_length_bits[symbol] == 0
         ? static_cast<std::uint16_t>(kUnusableSymbolCost)
         : static_cast<std::uint16_t>(literal_length_bits[symbol] + kLengthExtra[i]);
   }

   out.length_to_symbol = kLengthToSymbol;
   out.distance_to_symbol = kDistanceToSymbol;

   return out;
}

std::int32_t literal_transition_cost(
   std::int32_t from_cost,
   std::uint8_t literal,
   const std::array<std::uint8_t, kLiteralLengthSymbols>& literal_length_bits) {
   // The binary performs a direct signed 32-bit addition. A valid active
   // literal has a nonzero code length; zero is retained as supplied.
   return static_cast<std::int32_t>(from_cost + literal_length_bits[literal]);
}

std::int32_t match_transition_cost(std::int32_t from_cost,
                                   std::uint16_t match_length,
                                   std::uint16_t distance,
                                   const ParserCostTable& table) {
   if (match_length < 3 || match_length > 258 || distance == 0 || distance > 32768) {
      throw std::out_of_range("DEFLATE match outside KZIP parser limits");
   }

   const auto length_symbol = table.length_to_symbol[match_length - 3];
   const auto distance_symbol = table.distance_to_symbol[distance - 1];
   const auto length_bits = table.token_cost[length_symbol];
   const auto distance_bits = table.token_cost[distance_symbol];
   return static_cast<std::int32_t>(from_cost + length_bits + distance_bits);
}

} // namespace kzip
