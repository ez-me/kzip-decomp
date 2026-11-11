#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace kzip {

constexpr std::uint32_t kWindowMask = 0x7fff;
constexpr std::uint32_t kCostMask = 0x1ff;
constexpr std::uint32_t kPredMask = 0x3ffff;
constexpr std::uint32_t kMaxDistance = 32768;
constexpr std::uint32_t kMaxMatch = 258;
constexpr std::uint32_t kMinMatch = 3;
constexpr std::uint32_t kDistanceArenaEntries = 0x800000;
constexpr std::uint32_t kDistanceArenaMaxBase = 0x7fff00;

[[nodiscard]] constexpr std::uint32_t cache_row_base_from_end(
   std::uint32_t row_end, std::uint32_t max_len_minus2) noexcept {
   return row_end - max_len_minus2;
}

struct RunCacheBuilder {
   std::uint8_t previous_byte{};
   std::uint32_t run_count{};
   bool initialized{};

   // Returns the count of identical bytes immediately preceding this byte.
   std::uint16_t push(std::uint8_t byte) noexcept;
};

class PackedDistanceArena {
public:
   // Store a row of per-length next-candidate distances. The original compares
   // the new row with a suffix of the previous row; on equality it reuses that
   // suffix, otherwise it appends the row at the current arena end.
   [[nodiscard]] std::uint32_t intern_row(std::span<const std::uint16_t> row);
   [[nodiscard]] std::span<const std::uint16_t> words() const noexcept {
      return words_;
   }
   void clear() noexcept { words_.clear(); }

private:
   std::vector<std::uint16_t> words_;
   std::uint32_t previous_row_end_{};
   std::uint32_t previous_row_length_{};
};

struct MatchCacheView {
   // The original binary uses a packed arena. row_base[pos] is an index in
   // distance_rows, not a byte offset. Rows are allowed to share storage.
   std::span<const std::uint16_t> distance_rows;
   std::span<const std::uint32_t> row_base;       // 32768 entries
   std::span<const std::uint32_t> max_len_minus2; // 32768 entries
   std::span<const std::uint16_t> run_length;     // 32768 entries

   [[nodiscard]] std::optional<std::uint32_t>
   next_candidate_distance(std::uint32_t current_pos,
                           std::uint32_t current_distance,
                           std::uint32_t required_length) const;

   [[nodiscard]] std::optional<std::uint32_t>
   next_distance_after_run(std::uint32_t current_pos,
                           std::uint32_t current_distance) const;
};

// Build one source-position row while candidates are visited nearest to farthest.
// Only the newly covered suffix is written: earlier candidates remain the
// nearest candidate for shorter match lengths.
std::uint32_t record_strategy0_candidate_row(std::span<std::uint16_t> row,
                                            std::uint32_t distance,
                                            std::uint32_t maximum_match_length,
                                            std::uint32_t& max_match_seen);

struct DeflateCostTables {
   std::array<std::uint8_t, 256> length_symbol{}; // indexed by length - 3
   std::array<std::uint8_t, kMaxDistance> distance_symbol{}; // indexed by distance - 1
   std::array<std::uint32_t, 59> token_cost{}; // 0..29 distance, 30..58 length
   std::array<std::uint8_t, 30> next_cheaper_distance_symbol{};
   // First (1-based) distance covered by each DEFLATE distance symbol. The
   // strategy-0 search restarts at a run boundary using these base distances.
   std::array<std::uint32_t, 30> distance_base{};
};

struct Predecessor {
   std::uint16_t length{};
   // Literal byte when length == 1; match distance when length >= 3.
   std::uint16_t value{};
};

struct ParseWorkspace {
   std::array<std::uint32_t, 512> costs{};
   std::array<Predecessor, 1u << 18> predecessors{};
   std::array<std::uint32_t, 256> best_distance_cost{}; // reset per input position

   ParseWorkspace();
   void begin_position() noexcept;
};

struct RelaxResult {
   std::uint32_t lengths_considered{};
   std::uint32_t lengths_improved_distance_cost{};
   std::uint32_t dp_updates{};
};

// The original's 6d4c60 array is a per-length best distance-token-cost
// frontier. A later candidate is useful for length L only if its distance
// token cost is strictly lower than the best already seen for that L.
// This routine uses a straightforward scan; the binary uses range-skipping
// branches to avoid scanning lengths whose frontier cannot improve.
RelaxResult relax_strategy0_candidate(
   ParseWorkspace& workspace,
   const DeflateCostTables& tables,
   std::uint32_t position,
   std::uint32_t source_cost,
   std::uint32_t block_end,
   std::uint32_t distance,
   std::uint32_t maximum_match_length);

// Literal transitions are deliberately <= in the binary. Match transitions
// are <. The distinction determines equal-cost tokenization.
bool relax_literal(ParseWorkspace& workspace,
                   const DeflateCostTables& tables,
                   std::uint32_t position,
                   std::uint32_t source_cost,
                   std::uint32_t block_end,
                   std::uint8_t literal);

// Build the 30-entry successor table used by 0x402c20. For each distance
// symbol s, return the first later symbol with strictly lower complete token
// cost; 30 is the no-successor sentinel.
void build_next_cheaper_distance_symbols(DeflateCostTables& tables);

} // namespace kzip
