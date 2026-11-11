#include "strategy0_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace kzip {

namespace {

inline std::int32_t signed_cost(std::uint32_t value) noexcept {
   return std::bit_cast<std::int32_t>(value);
}

inline std::uint32_t add_cost(std::uint32_t a, std::uint32_t b) noexcept {
   // The original uses 32-bit register arithmetic, including wraparound.
   return a + b;
}

} // namespace

std::uint32_t PackedDistanceArena::intern_row(
   std::span<const std::uint16_t> row) {
   if (row.size() > kMaxMatch - 2) {
      throw std::out_of_range("strategy-0 row exceeds 256 entries");
   }

   // The binary compares the new row against a suffix of the preceding row.
   // The row is represented by [end - length, end), so a shorter matching row
   // can reuse the tail of the previous row without advancing the arena end.
   if (row.size() <= previous_row_length_) {
      const std::uint32_t base = previous_row_end_ -
         static_cast<std::uint32_t>(row.size());
      const auto begin = words_.begin() + static_cast<std::ptrdiff_t>(base);
      if (std::equal(row.begin(), row.end(), begin)) {
         previous_row_length_ = static_cast<std::uint32_t>(row.size());
         return base;
      }
   }

   if (words_.size() > kDistanceArenaMaxBase ||
       words_.size() + row.size() > kDistanceArenaEntries) {
      words_.clear();
      previous_row_end_ = 0;
      previous_row_length_ = 0;
   }
   if (words_.size() > kDistanceArenaMaxBase) {
      throw std::length_error("strategy-0 distance arena cursor exceeds limit");
   }

   const auto base = static_cast<std::uint32_t>(words_.size());
   words_.insert(words_.end(), row.begin(), row.end());
   previous_row_end_ = static_cast<std::uint32_t>(words_.size());
   previous_row_length_ = static_cast<std::uint32_t>(row.size());
   return base;
}

std::uint16_t RunCacheBuilder::push(std::uint8_t byte) noexcept {
   if (!initialized || byte != previous_byte) {
      run_count = 0;
      initialized = true;
   } else if (run_count < 0x7fff) {
      ++run_count;
   }
   previous_byte = byte;
   return static_cast<std::uint16_t>(run_count);
}

std::optional<std::uint32_t> MatchCacheView::next_candidate_distance(
   std::uint32_t current_pos,
   std::uint32_t current_distance,
   std::uint32_t required_length) const {
   if (required_length < kMinMatch || required_length > kMaxMatch ||
       current_distance == 0 || current_distance > kMaxDistance ||
       row_base.size() < 32768 || max_len_minus2.size() < 32768) {
      return std::nullopt;
   }

   const std::uint32_t source = (current_pos - current_distance) & kWindowMask;
   const std::uint32_t required_minus2 = required_length - 2;
   if (required_minus2 > max_len_minus2[source]) {
      return std::nullopt;
   }

   const std::uint64_t row_index =
      static_cast<std::uint64_t>(row_base[source]) + required_length - 3;
   if (row_index >= distance_rows.size()) {
      return std::nullopt;
   }

   const std::uint32_t following = distance_rows[static_cast<std::size_t>(row_index)];
   if (following == 0 || following > kMaxDistance ||
       current_distance + following > kMaxDistance) {
      return std::nullopt;
   }

   return current_distance + following;
}

std::uint32_t record_strategy0_candidate_row(
   std::span<std::uint16_t> row,
   std::uint32_t distance,
   std::uint32_t maximum_match_length,
   std::uint32_t& max_match_seen) {
   if (distance == 0 || distance > kMaxDistance ||
       maximum_match_length < kMinMatch) {
      return 0;
   }
   maximum_match_length = std::min(maximum_match_length, kMaxMatch);
   const std::uint32_t first_length = std::max(kMinMatch, max_match_seen + 1);
   if (first_length > maximum_match_length) {
      return 0;
   }
   const std::uint32_t last_index = maximum_match_length - 3;
   if (last_index >= row.size()) {
      throw std::out_of_range("strategy-0 cache row is too short");
   }
   for (std::uint32_t length = first_length; length <= maximum_match_length; ++length) {
      row[length - 3] = static_cast<std::uint16_t>(distance);
   }
   max_match_seen = maximum_match_length;
   return maximum_match_length - first_length + 1;
}

std::optional<std::uint32_t> MatchCacheView::next_distance_after_run(
   std::uint32_t current_pos,
   std::uint32_t current_distance) const {
   if (run_length.size() < 32768 || current_distance == 0 ||
       current_distance > kMaxDistance) {
      return std::nullopt;
   }
   const std::uint32_t source = (current_pos - current_distance) & kWindowMask;
   const std::uint32_t run = run_length[source]; // identical bytes immediately preceding source
   if (run == 0 || current_distance + run > kMaxDistance) {
      return std::nullopt;
   }
   return current_distance + run;
}

ParseWorkspace::ParseWorkspace() {
   costs.fill(std::numeric_limits<std::uint32_t>::max());
   costs[0] = 0x80000000u; // original parser's initial reachable state
   begin_position();
}

void ParseWorkspace::begin_position() noexcept {
   best_distance_cost.fill(std::numeric_limits<std::uint32_t>::max());
}

RelaxResult relax_strategy0_candidate(
   ParseWorkspace& workspace,
   const DeflateCostTables& tables,
   std::uint32_t position,
   std::uint32_t source_cost,
   std::uint32_t block_end,
   std::uint32_t distance,
   std::uint32_t maximum_match_length) {
   if (distance == 0 || distance > kMaxDistance ||
       maximum_match_length < kMinMatch) {
      return {};
   }

   maximum_match_length = std::min(maximum_match_length, kMaxMatch);
   RelaxResult result{};
   const std::uint32_t distance_symbol = tables.distance_symbol[distance - 1];
   if (distance_symbol >= 30) {
      throw std::out_of_range("distance symbol outside DEFLATE alphabet");
   }
   const std::uint32_t distance_cost = tables.token_cost[distance_symbol];

   for (std::uint32_t length = kMinMatch;
        length <= maximum_match_length;
        ++length) {
      ++result.lengths_considered;
      const std::uint32_t length_index = length - 3;
      const std::uint32_t old_distance_cost =
         workspace.best_distance_cost[length_index];

      // 0x404c90 uses JGE: equal-cost or more expensive distances are
      // rejected, preserving the first candidate in hash-chain order.
      if (distance_cost >= old_distance_cost) {
         continue;
      }
      workspace.best_distance_cost[length_index] = distance_cost;
      ++result.lengths_improved_distance_cost;

      const std::uint32_t length_symbol = tables.length_symbol[length_index];
      if (length_symbol < 30 || length_symbol >= 59) {
         throw std::out_of_range("length symbol outside DEFLATE length alphabet");
      }

      const std::uint32_t destination = position + length;
      if (destination > block_end) {
         continue;
      }

      const std::uint32_t candidate_cost = add_cost(
         add_cost(source_cost, distance_cost),
         tables.token_cost[length_symbol]);
      const std::uint32_t cost_slot = destination & kCostMask;

      // 0x404688 uses signed SETL: match transitions require strict <.
      if (signed_cost(candidate_cost) < signed_cost(workspace.costs[cost_slot])) {
         workspace.costs[cost_slot] = candidate_cost;
         workspace.predecessors[destination & kPredMask] = {
            static_cast<std::uint16_t>(length),
            static_cast<std::uint16_t>(distance)
         };
         ++result.dp_updates;
      }
   }

   return result;
}

bool relax_literal(ParseWorkspace& workspace,
                   const DeflateCostTables& tables,
                   std::uint32_t position,
                   std::uint32_t source_cost,
                   std::uint32_t block_end,
                   std::uint8_t literal) {
   const std::uint32_t destination = position + 1;
   if (destination > block_end) {
      return false;
   }

   const std::uint32_t candidate_cost =
      add_cost(source_cost, tables.token_cost[literal]);
   const std::uint32_t slot = destination & kCostMask;

   // 0x403f82 uses JG to reject only strictly greater candidates. Literals
   // therefore replace equal-cost predecessor choices.
   if (signed_cost(candidate_cost) <= signed_cost(workspace.costs[slot])) {
      workspace.costs[slot] = candidate_cost;
      workspace.predecessors[destination & kPredMask] = {1, literal};
      return true;
   }
   return false;
}

void build_next_cheaper_distance_symbols(DeflateCostTables& tables) {
   for (std::uint32_t symbol = 0; symbol < 30; ++symbol) {
      std::uint32_t next = symbol + 1;
      while (next < 30 &&
             tables.token_cost[next] >= tables.token_cost[symbol]) {
         ++next;
      }
      tables.next_cheaper_distance_symbol[symbol] =
         static_cast<std::uint8_t>(next);
   }
}

} // namespace kzip
