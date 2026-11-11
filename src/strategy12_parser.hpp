#pragma once

#include "strategy0_encoder.hpp"
#include "strategy0_model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kzip {

// One block's parsed token stream plus the target's raw frequency array. The
// high bit of a frequency entry is the parser's "seen" marker and is cleared
// later by finalize_frequencies().
struct Strategy12BlockResult {
   std::vector<EncoderToken> tokens;
   std::array<std::int32_t, kTokenSlots> frequencies{};
};

// Result of the strategy-0 arena-based match search: the number of packed row
// entries written for this source position and the arena cursor after the row.
struct Strategy0SearchResult {
   std::int32_t row_len{0};
   std::int32_t arena_end{0};
};

// Streaming reconstruction of the strategy-1/2 parser at 0x403DC0. The target
// keeps a rolling cost ring, a predecessor ring, hash chains, and the packed
// match-distance row arena alive for a whole optimization pass; only the cost
// ring is reset at each deflate block boundary. Instantiate once per pass and
// call parse_block() for each block in order.
class Strategy12Parser {
public:
   Strategy12Parser(std::span<const std::uint8_t> input, int strategy,
                    std::int32_t run_cap = 0x7fff,
                    std::int32_t arena_base_limit = 0x7fff00);

   // Reset the per-pass state exactly like the target's optimization setup.
   // The cost ring, hash chains and rolling row state are reinitialised; the
   // predecessor ring, row arena and row-base table persist across passes,
   // matching the target's process-wide globals.
   void begin_pass();

   // Full reset used when the parser is first constructed.
   void reset_pass();

   // Parse positions [pos, block_end) and emit their tokens. `lengths` are the
   // block's start-of-pass code lengths (literal costs) and `tables` supply the
   // length/distance symbol and token-cost mappings used by the target.
   Strategy12BlockResult parse_block(
      std::size_t block_end,
      const DeflateCostTables& tables,
      const std::array<std::uint8_t, kTokenSlots>& lengths);

private:
   std::uint8_t byte_at(std::int32_t index) const noexcept;
   std::uint32_t word_at(std::int32_t index) const noexcept;
   void ensure_arena(std::size_t index);
   void append_window();
   void fast_forward(std::int32_t start, std::int32_t end);
   void commit(Strategy12BlockResult& result,
               const DeflateCostTables& tables);
   Strategy0SearchResult search_strategy0(
      std::int32_t position, std::int32_t source_cost,
      std::int32_t run_ahead, std::int32_t initial_distance,
      const DeflateCostTables& tables, std::int32_t block_limit);
   void intern_row(std::int32_t position, std::int32_t row_base,
                   std::int32_t row_len, std::int32_t arena_index);

   std::span<const std::uint8_t> input_;
   int strategy_{0};
   std::int32_t run_cap_{0x7fff};
   std::int32_t arena_base_limit_{0x7fff00};

   // Process-wide scratch, matching the target's globals. These are all
   // reinitialised by begin_pass()/reset_pass() before use, so sharing them
   // across parser instances is observationally identical to per-instance
   // storage and avoids zeroing several hundred KiB per file.
   static std::array<std::int32_t, 512> cost_ring_;
   static std::vector<Predecessor> predecessor_ring_;

   // The target's input window (0x1E9E720) is a process-wide 64 KiB ring that
   // is only appended to, never cleared. Each pass rewrites the ring from the
   // start, so positions that have not been appended yet still hold bytes from
   // the previous pass (or zero at process start); those stale bytes are
   // visible to match searches whose distance exceeds the current position.
   // The four bytes at 0x1EAE720 mirror the first dword of the ring so that
   // the target's non-wrapping 32-bit word reads across the ring end can be
   // reproduced.
   static std::array<std::uint8_t, 65536 + 8> window_;
   std::int32_t window_len_{0};
   std::int32_t feed_cursor_{0};
   std::int32_t piece_offset_{0};
   static std::array<std::uint16_t, 4096> hash_heads_;
   static std::array<std::uint16_t, 32768> hash_predecessors_;
   std::array<std::uint16_t, 65536> run_cache_{};
   static std::array<std::int32_t, 32768> max_match_minus2_;
   static std::array<std::int32_t, 32768> row_base_;
   std::vector<std::uint16_t> packed_rows_;

   std::uint8_t prev_byte_{0};
   std::int32_t run_count_{-1};
   std::int32_t scan_{0};
   std::int32_t prev_row_length_{-1};
   std::int32_t prev_row_end_{0};
   std::int32_t arena_cursor_{0};
   std::int32_t pos_{0};
   std::int32_t farthest_{0};
   std::int32_t last_committed_{0};

   // The target's strategy-0 "best distance cost frontier" (0x6D4C60) is a
   // process-wide .bss array that is never cleared: stale entries from earlier
   // positions and earlier optimization passes are visible to later searches.
   // It is shared by every parser instance so its lifetime matches the target.
   static std::array<std::uint32_t, 256> best_frontier_;
};

} // namespace kzip
