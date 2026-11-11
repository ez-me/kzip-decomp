#include "strategy12_parser.hpp"

#include "parser_cost.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace kzip {
namespace {

constexpr std::int32_t kInitialCost =
   std::numeric_limits<std::int32_t>::min(); // 0x80000000
constexpr std::int32_t kUnreachable = std::numeric_limits<std::int32_t>::max();
constexpr std::uint32_t kHashMask = 0xfff;
constexpr std::int32_t kArenaMaxDistance = 0x8000;

inline std::uint32_t hash_word(std::uint32_t word) noexcept {
   return ((word >> 12) - word) & kHashMask;
}

} // namespace

std::array<std::uint32_t, 256> Strategy12Parser::best_frontier_{};
std::array<std::uint8_t, 65536 + 8> Strategy12Parser::window_{};
std::array<std::int32_t, 512> Strategy12Parser::cost_ring_{};
std::vector<Predecessor> Strategy12Parser::predecessor_ring_(1u << 18);
std::array<std::uint16_t, 4096> Strategy12Parser::hash_heads_{};
std::array<std::uint16_t, 32768> Strategy12Parser::hash_predecessors_{};
std::array<std::int32_t, 32768> Strategy12Parser::max_match_minus2_{};
std::array<std::int32_t, 32768> Strategy12Parser::row_base_{};

Strategy12Parser::Strategy12Parser(std::span<const std::uint8_t> input,
                                   int strategy, std::int32_t run_cap,
                                   std::int32_t arena_base_limit)
   : input_(input), strategy_(strategy), run_cap_(run_cap),
     arena_base_limit_(arena_base_limit) {
   reset_pass();
}

void Strategy12Parser::begin_pass() {
   cost_ring_.fill(kUnreachable);
   cost_ring_[0] = kInitialCost;
   hash_heads_.fill(0);
   hash_predecessors_.fill(0);
   max_match_minus2_.fill(0);
   prev_byte_ = 0;
   run_count_ = -1;
   scan_ = 0;
   prev_row_length_ = -1;
   prev_row_end_ = 0;
   arena_cursor_ = 0;
   pos_ = 0;
   farthest_ = 0;
   last_committed_ = 0;
   window_len_ = 0;
   feed_cursor_ = 0;
   piece_offset_ = 0;
}

void Strategy12Parser::reset_pass() {
   // The target's predecessor ring and row-base table are process-wide globals
   // that are never cleared per file (only the cost ring and hash chains are
   // reset at each pass). The reconstruction still starts every file from
   // zeros, which is equivalent for every entry a file can read: destinations
   // only range over [0, input size]. Clearing just that prefix keeps the
   // per-file cost proportional to the file size instead of 1 MiB, which
   // matters for archives with tens of thousands of tiny members.
   const std::size_t touched =
      std::min<std::size_t>(predecessor_ring_.size(),
                            input_.size() + 1);
   std::fill_n(predecessor_ring_.begin(), touched, Predecessor{});
   const std::size_t row_touched =
      std::min<std::size_t>(row_base_.size(), input_.size() + 1);
   std::fill_n(row_base_.begin(), row_touched, 0);
   packed_rows_.clear();
   begin_pass();
}

std::uint8_t Strategy12Parser::byte_at(std::int32_t index) const noexcept {
   return window_[static_cast<std::uint32_t>(index) & 0xffffu];
}

std::uint32_t Strategy12Parser::word_at(std::int32_t index) const noexcept {
   const std::size_t base = static_cast<std::uint32_t>(index) & 0xffffu;
   return static_cast<std::uint32_t>(window_[base]) |
      (static_cast<std::uint32_t>(window_[base + 1]) << 8) |
      (static_cast<std::uint32_t>(window_[base + 2]) << 16);
}

void Strategy12Parser::ensure_arena(std::size_t index) {
   if (index >= packed_rows_.size()) {
      packed_rows_.resize(index + 1);
   }
}

void Strategy12Parser::append_window() {
   std::int32_t room = 0x10000 - piece_offset_;
   std::int32_t chunk = room < 0x7e00 ? room : 0x7e00;
   const std::int32_t available =
      static_cast<std::int32_t>(input_.size()) - feed_cursor_;
   if (chunk > available) {
      chunk = available;
   }
   if (chunk <= 0) {
      return;
   }
   for (std::int32_t i = 0; i < chunk; ++i) {
      window_[static_cast<std::uint16_t>(window_len_)] =
         input_[static_cast<std::size_t>(feed_cursor_ + i)];
      ++window_len_;
   }
   feed_cursor_ += chunk;
   piece_offset_ += chunk;
   if (piece_offset_ >= 0x10000) {
      piece_offset_ = 0;
   }
   // The target copies the ring's first dword into 0x1EAE720 right after each
   // append, so a 32-bit word read crossing the ring end sees these bytes.
   for (int i = 0; i < 4; ++i) {
      window_[65536 + i] = window_[i];
   }
}

void Strategy12Parser::fast_forward(std::int32_t start, std::int32_t end) {
   for (std::int32_t p = start; p < end; ++p) {
      cost_ring_[static_cast<std::size_t>(p) & 0x1ff] = kUnreachable;
      const std::uint8_t byte = byte_at(p);
      if (byte == prev_byte_) {
         if (run_count_ < run_cap_) {
            ++run_count_;
         } else {
            run_count_ = run_cap_;
         }
      } else {
         run_count_ = 0;
      }
      run_cache_[static_cast<std::size_t>(p) & 0xffff] =
         static_cast<std::uint16_t>(run_count_);
      max_match_minus2_[static_cast<std::size_t>(p) & 0x7fff] = 0;
      prev_row_length_ = 0;
      prev_byte_ = byte;

      const std::uint32_t word = word_at(p);
      const std::uint32_t hash = hash_word(word);
      const std::uint16_t old_head = hash_heads_[hash];
      hash_predecessors_[static_cast<std::size_t>(p) & 0x7fff] = old_head;
      hash_heads_[hash] = static_cast<std::uint16_t>(p & 0x7fff);
   }
   pos_ = end;
}

void Strategy12Parser::commit(Strategy12BlockResult& result,
                              const DeflateCostTables& tables) {
   const std::int32_t end = farthest_;
   std::int32_t token_start =
      end - static_cast<std::int32_t>(predecessor_ring_[end & kPredMask].length);
   std::int32_t emit_start = end;
   if (token_start > last_committed_) {
      std::int32_t cursor = token_start;
      while (last_committed_ < cursor) {
         --emit_start;
         predecessor_ring_[emit_start & kPredMask] =
            predecessor_ring_[cursor & kPredMask];
         cursor -= static_cast<std::int32_t>(
            predecessor_ring_[cursor & kPredMask].length);
      }
   }

   for (std::int32_t position = emit_start; position <= end; ++position) {
      const Predecessor predecessor = predecessor_ring_[position & kPredMask];
      if (predecessor.length == 1) {
         EncoderToken token;
         token.kind = EncoderToken::Kind::Literal;
         token.value = predecessor.value;
         result.tokens.push_back(token);
         ++result.frequencies[predecessor.value];
      } else {
         EncoderToken token;
         token.kind = EncoderToken::Kind::Match;
         token.value = predecessor.length;
         token.distance = predecessor.value;
         result.tokens.push_back(token);
         const std::uint32_t length_symbol =
            tables.length_symbol[predecessor.length - 3];
         const std::uint32_t distance_symbol =
            tables.distance_symbol[predecessor.value - 1];
         ++result.frequencies[length_symbol + 227];
         ++result.frequencies[288 + distance_symbol];
      }
   }
}

Strategy0SearchResult Strategy12Parser::search_strategy0(
   std::int32_t position, std::int32_t source_cost, std::int32_t run_ahead,
   std::int32_t initial_distance, const DeflateCostTables& tables,
   std::int32_t block_limit) {
   const std::int32_t row_base_value = row_base_[position & 0x7fff];

   // The target's best-distance-cost frontier (0x6D4C60) is a process-wide
   // array that is never cleared, so stale entries from earlier positions and
   // passes are visible here.
   std::array<std::uint32_t, 256>& best = best_frontier_;

   std::int32_t distance = initial_distance;
   std::int32_t sym = tables.distance_symbol[distance - 1];
   std::int32_t length = 3;
   std::int32_t dest = position + 3;
   std::int32_t index = 0;
   std::int32_t arena_write = row_base_value;
   std::int32_t row_len = 0;
   std::int32_t next_sym = 0;
   std::int32_t threshold = 0;
   std::int32_t sym_choice = 0;
   std::int32_t candidate = 0;
   std::int32_t cost = 0;
   std::int32_t sym2 = 0;
   std::int32_t found_length = 0;

record:
   row_len = arena_write - row_base_value;
   if (row_len < length - 2) {
      ensure_arena(static_cast<std::size_t>(arena_write));
      packed_rows_[static_cast<std::size_t>(arena_write)] =
         static_cast<std::uint16_t>(distance);
      ++arena_write;
      row_len = arena_write - row_base_value;
      cost = static_cast<std::int32_t>(tables.token_cost[sym]);
      best[static_cast<std::size_t>(index)] = static_cast<std::uint32_t>(cost);
   } else {
      cost = static_cast<std::int32_t>(tables.token_cost[sym]);
      if (static_cast<std::uint32_t>(cost) <
          best[static_cast<std::size_t>(index)]) {
         best[static_cast<std::size_t>(index)] =
            static_cast<std::uint32_t>(cost);
      }
   }
   candidate = cost +
      static_cast<std::int32_t>(
         tables.token_cost[tables.length_symbol[static_cast<std::size_t>(index)]]) +
      source_cost;
   if (candidate < cost_ring_[static_cast<std::size_t>(dest) & 0x1ff] &&
       dest <= block_limit) {
      if (dest > farthest_) {
         farthest_ = dest;
      }
      predecessor_ring_[static_cast<std::size_t>(dest) & kPredMask] = {
         static_cast<std::uint16_t>(length),
         static_cast<std::uint16_t>(distance)};
      cost_ring_[static_cast<std::size_t>(dest) & 0x1ff] = candidate;
   }
   if (length == 0x102) {
      goto max_length;
   }

   next_sym = tables.next_cheaper_distance_symbol[static_cast<std::size_t>(sym)];
   {
      const std::uint8_t dest_byte = byte_at(dest);
      std::int32_t scan_distance = distance;
      for (;;) {
         if (byte_at(dest - scan_distance) == dest_byte) {
            distance = scan_distance;
            goto extends;
         }
         if (length == run_ahead) {
            scan_distance += static_cast<std::int32_t>(
               run_cache_[static_cast<std::size_t>(position - scan_distance) & 0xffff]);
         }
         {
            const std::int32_t source = (position - scan_distance) & 0x7fff;
            if (length - 2 >
                max_match_minus2_[static_cast<std::size_t>(source)]) {
               goto fail;
            }
            scan_distance += static_cast<std::int32_t>(packed_rows_[
               static_cast<std::size_t>(
                  row_base_[static_cast<std::size_t>(source)]) +
               static_cast<std::size_t>(length) - 3]);
            if (scan_distance > 0x8000) {
               goto fail;
            }
         }
         sym2 = tables.distance_symbol[static_cast<std::size_t>(scan_distance) - 1];
         if (next_sym <= sym2) {
            break;
         }
      }

      if (sym2 <
          tables.next_cheaper_distance_symbol[static_cast<std::size_t>(next_sym)]) {
         sym_choice = next_sym;
      } else {
         sym_choice = next_sym;
         std::int32_t next =
            tables.next_cheaper_distance_symbol[static_cast<std::size_t>(sym_choice)];
         while (sym2 >= next) {
            sym_choice = next;
            next =
               tables.next_cheaper_distance_symbol[static_cast<std::size_t>(sym_choice)];
         }
      }
      threshold = static_cast<std::int32_t>(tables.token_cost[
         static_cast<std::size_t>(sym_choice)]);
      goto search_length;
   }

extends:
   {
      const std::int32_t old_length = length;
      length = old_length + 1;
      index = old_length - 2;
      sym = tables.distance_symbol[static_cast<std::size_t>(distance) - 1];
      dest = old_length + 1 + position;
      goto record;
   }

fail:
   if (next_sym > 29) {
      goto end_row;
   }
   threshold = static_cast<std::int32_t>(
      tables.token_cost[static_cast<std::size_t>(next_sym)]);
   sym_choice = next_sym;
   goto search_length;

search_length:
   if (length <= 5) {
      found_length = length;
      if (found_length != 3) {
         goto refine;
      }
      goto after_search;
   }
   if (threshold >=
       static_cast<std::int32_t>(best[static_cast<std::size_t>(length) - 6])) {
      found_length = length;
      goto refine;
   }
   {
      std::int32_t cursor = index; // length - 3
      for (;;) {
         found_length = cursor;
         if (cursor <= 5) {
            break;
         }
         cursor -= 3;
         if (threshold <
             static_cast<std::int32_t>(best[static_cast<std::size_t>(cursor) - 3])) {
            continue;
         }
         goto refine;
      }
      if (found_length == 3) {
         goto after_search;
      }
      goto refine;
   }

refine:
   if (threshold <
       static_cast<std::int32_t>(best[static_cast<std::size_t>(found_length) - 4])) {
      std::int32_t cursor = found_length - 1;
      for (;;) {
         found_length = cursor;
         if (cursor == 3) {
            found_length = 3;
            goto after_search;
         }
         cursor -= 1;
         if (threshold >=
             static_cast<std::int32_t>(best[static_cast<std::size_t>(cursor) - 3])) {
            found_length = cursor + 1;
            goto after_search;
         }
      }
   }

after_search:
   length = found_length;
   for (;;) {
      if (run_ahead == length) {
         distance += static_cast<std::int32_t>(
            run_cache_[static_cast<std::size_t>(position - distance) & 0xffff]);
         if (distance > 0x8000) {
            goto end_row;
         }
         if (next_sym <=
             static_cast<std::int32_t>(
                tables.distance_symbol[static_cast<std::size_t>(distance) - 1])) {
            const std::int32_t base = static_cast<std::int32_t>(
               tables.distance_base[static_cast<std::size_t>(sym_choice)]);
            length = run_ahead;
            index = run_ahead - 3;
            sym = tables.distance_symbol[static_cast<std::size_t>(base) - 1];
            distance = base;
            dest = run_ahead + position;
            goto record;
         }
      }
      {
         const std::int32_t source = (position - distance) & 0x7fff;
         if (length - 2 >
             max_match_minus2_[static_cast<std::size_t>(source)]) {
            goto end_row;
         }
         distance += static_cast<std::int32_t>(packed_rows_[
            static_cast<std::size_t>(
               row_base_[static_cast<std::size_t>(source)]) +
            static_cast<std::size_t>(length) - 3]);
         if (distance > 0x8000) {
            goto end_row;
         }
      }
      sym = tables.distance_symbol[static_cast<std::size_t>(distance) - 1];
      if (next_sym <= sym) {
         index = length - 3;
         dest = position + length;
         goto record;
      }
   }

max_length:
   if (tables.next_cheaper_distance_symbol[static_cast<std::size_t>(sym)] > 29) {
      goto end_row;
   }
   if (run_ahead == 0x102) {
      distance += static_cast<std::int32_t>(
         run_cache_[static_cast<std::size_t>(position - distance) & 0xffff]);
   }
   {
      const std::int32_t source = (position - distance) & 0x7fff;
      if (max_match_minus2_[static_cast<std::size_t>(source)] <= 0xff) {
         goto end_row;
      }
      distance += static_cast<std::int32_t>(packed_rows_[
         static_cast<std::size_t>(row_base_[static_cast<std::size_t>(source)]) +
         0xff]);
      if (distance > 0x8000) {
         goto end_row;
      }
   }
   sym = tables.distance_symbol[static_cast<std::size_t>(distance) - 1];
   goto record;

end_row:
   return {row_len, arena_write};
}

void Strategy12Parser::intern_row(std::int32_t position, std::int32_t row_base,
                                  std::int32_t row_len,
                                  std::int32_t arena_index) {
   max_match_minus2_[static_cast<std::size_t>(position) & 0x7fff] = row_len;

   // Intern the row when its tail is a suffix of the previous row.
   bool reuse = false;
   if (prev_row_length_ >= row_len) {
      bool prefix_ok = true;
      if ((row_len & 1) != 0) {
         prefix_ok =
            packed_rows_[static_cast<std::size_t>(row_base)] ==
            packed_rows_[static_cast<std::size_t>(prev_row_end_ - row_len)];
      }
      if (prefix_ok) {
         const std::int32_t count = row_len & ~1;
         if (count == 0) {
            reuse = true;
         } else {
            reuse = true;
            const std::int32_t new_begin = arena_index - count;
            const std::int32_t old_begin = prev_row_end_ - count;
            for (std::int32_t offset = 0; offset < count; ++offset) {
               if (packed_rows_[static_cast<std::size_t>(new_begin + offset)] !=
                   packed_rows_[static_cast<std::size_t>(old_begin + offset)]) {
                  reuse = false;
                  break;
               }
            }
         }
      }
   }

   if (reuse) {
      row_base_[static_cast<std::size_t>(position) & 0x7fff] =
         prev_row_end_ - row_len;
      arena_cursor_ = prev_row_end_;
   } else {
      prev_row_end_ = arena_index;
      arena_cursor_ = arena_index;
   }

   prev_row_length_ = row_len;
}

Strategy12BlockResult Strategy12Parser::parse_block(
   std::size_t block_end,
   const DeflateCostTables& tables,
   const std::array<std::uint8_t, kTokenSlots>& lengths) {
   Strategy12BlockResult result;
   if (block_end <= static_cast<std::size_t>(pos_)) {
      return result;
   }

   const std::int32_t limit = static_cast<std::int32_t>(block_end);
   const std::int32_t input_size = static_cast<std::int32_t>(input_.size());

   // The target streams the whole file through the parser's 64 KiB ring in
   // 0x10000-byte read pieces split into 0x7E00-byte chunks. Block boundaries
   // only cap token destinations (local_6c); they do not bound the read
   // window, and the chunk alignment carries across blocks.
   while (true) {
      if (limit <= pos_) {
         break;
      }
      if (feed_cursor_ < input_size && pos_ >= window_len_ - 0x104) {
         append_window();
         continue;
      }

      const std::int32_t source_cost =
         cost_ring_[static_cast<std::size_t>(pos_) & 0x1ff];
      cost_ring_[static_cast<std::size_t>(pos_) & 0x1ff] = kUnreachable;

      const std::uint8_t current_byte = byte_at(pos_);

      // Run-length cache for this position.
      if (current_byte == prev_byte_) {
         if (run_count_ < run_cap_) {
            ++run_count_;
         } else {
            run_count_ = run_cap_;
         }
      } else {
         run_count_ = 0;
      }
      run_cache_[static_cast<std::size_t>(pos_) & 0xffff] =
         static_cast<std::uint16_t>(run_count_);

      // Scan ahead to the end of the run that starts at this position.
      if (scan_ < pos_ + 1) {
         scan_ = pos_ + 1;
      }
      if (scan_ < window_len_) {
         while (scan_ < window_len_ && byte_at(scan_) == current_byte) {
            ++scan_;
         }
      }

      const std::uint32_t word = word_at(pos_);
      const std::uint32_t hash = hash_word(word);

      // The target zeroes the row base once the arena cursor passes the cap.
      const std::int32_t row_base =
         (arena_cursor_ <= arena_base_limit_) ? arena_cursor_ : 0;
      row_base_[static_cast<std::size_t>(pos_) & 0x7fff] = row_base;

      const std::uint16_t original_head = hash_heads_[hash];
      std::int32_t slot = static_cast<std::int16_t>(original_head);
      std::uint32_t distance =
         ((static_cast<std::uint32_t>(pos_) - static_cast<std::uint32_t>(slot) - 1u) &
          0x7fffu) +
         1u;

      bool candidate_found = false;
      for (;;) {
         if (((word_at(static_cast<std::int32_t>(
                  static_cast<std::uint32_t>(pos_) - distance)) ^
               word) &
              0xffffffu) == 0) {
            if (pos_ >= static_cast<std::int32_t>(distance)) {
               candidate_found = true;
            }
            break;
         }
         slot = static_cast<std::int16_t>(
            hash_predecessors_[static_cast<std::size_t>(slot) & 0x7fff]);
         const std::uint32_t next_distance =
            ((static_cast<std::uint32_t>(pos_) -
              static_cast<std::uint32_t>(slot) - 1u) &
             0x7fffu) +
            1u;
         if (next_distance <= distance) {
            break;
         }
         distance = next_distance;
      }

      std::int32_t row_len = 0;
      std::int32_t arena_index = 0;
      if (!candidate_found) {
         // No three-byte match: the row is empty and reuses the previous
         // suffix (or the fresh arena base on the first position).
         max_match_minus2_[static_cast<std::size_t>(pos_) & 0x7fff] = 0;
         row_len = 0;
         if (prev_row_length_ < 0) {
            arena_cursor_ = row_base;
            prev_row_end_ = row_base;
         } else {
            row_base_[static_cast<std::size_t>(pos_) & 0x7fff] = prev_row_end_;
            arena_cursor_ = prev_row_end_;
         }
      } else if (strategy_ == 0) {
         const Strategy0SearchResult searched = search_strategy0(
            pos_, source_cost, scan_ - pos_,
            static_cast<std::int32_t>(distance), tables, limit);
         row_len = searched.row_len;
         arena_index = searched.arena_end;
         intern_row(pos_, row_base, row_len, arena_index);
      } else {
         const std::int32_t run_ahead = scan_ - pos_;
         std::int32_t length = 3;
         std::int32_t destination = pos_ + 3;
         arena_index = row_base + 1;
         std::int32_t current_distance = static_cast<std::int32_t>(distance);

         for (;;) {
            ensure_arena(static_cast<std::size_t>(arena_index));
            packed_rows_[static_cast<std::size_t>(arena_index) - 1] =
               static_cast<std::uint16_t>(current_distance);

            const std::uint32_t distance_symbol =
               tables.distance_symbol[static_cast<std::size_t>(current_distance) - 1];
            const std::uint32_t length_symbol =
               tables.length_symbol[static_cast<std::size_t>(length) - 3];
            const std::uint32_t candidate =
               static_cast<std::uint32_t>(source_cost) +
               tables.token_cost[length_symbol] +
               tables.token_cost[distance_symbol];

            if (destination <= limit &&
                std::bit_cast<std::int32_t>(candidate) <
                   cost_ring_[static_cast<std::size_t>(destination) & 0x1ff]) {
               if (destination > farthest_) {
                  farthest_ = destination;
               }
               predecessor_ring_[static_cast<std::size_t>(destination) & kPredMask] = {
                  static_cast<std::uint16_t>(length),
                  static_cast<std::uint16_t>(current_distance)};
               cost_ring_[static_cast<std::size_t>(destination) & 0x1ff] =
                  std::bit_cast<std::int32_t>(candidate);
            }

            if (length == 0x102) {
               row_len = arena_index - row_base;
               break;
            }

            bool matched = true;
            while (byte_at(destination - current_distance) !=
                   byte_at(destination)) {
               if (length == run_ahead) {
                  current_distance += static_cast<std::int32_t>(
                     run_cache_[static_cast<std::size_t>(pos_ - current_distance) &
                                0xffff]);
               }
               const std::int32_t source = (pos_ - current_distance) & 0x7fff;
               if (max_match_minus2_[static_cast<std::size_t>(source)] <
                   length - 2) {
                  matched = false;
                  break;
               }
               current_distance +=
                  packed_rows_[static_cast<std::size_t>(
                     row_base_[static_cast<std::size_t>(source)] + length - 3)];
               if (current_distance > kArenaMaxDistance) {
                  matched = false;
                  break;
               }
            }
            if (!matched) {
               row_len = arena_index - row_base;
               break;
            }

            ++length;
            ++destination;
            ++arena_index;
         }

         intern_row(pos_, row_base, row_len, arena_index);
      }

      prev_row_length_ = row_len;

      // Insert this position into the hash chain.
      hash_predecessors_[static_cast<std::size_t>(pos_) & 0x7fff] =
         original_head;
      hash_heads_[hash] = static_cast<std::uint16_t>(pos_ & 0x7fff);
      prev_byte_ = current_byte;

      // Literal transition. The target marks the literal as "seen" before the
      // cost comparison and uses <=, so equal-cost literals replace matches.
      result.frequencies[current_byte] |= static_cast<std::int32_t>(0x80000000u);
      const std::int32_t literal_destination = pos_ + 1;
      const std::int32_t literal_cost =
         source_cost + static_cast<std::int32_t>(lengths[current_byte]);
      if (literal_cost <=
          cost_ring_[static_cast<std::size_t>(literal_destination) & 0x1ff]) {
         if (literal_destination > farthest_) {
            farthest_ = literal_destination;
         }
         predecessor_ring_[static_cast<std::size_t>(literal_destination) & kPredMask] = {
            1, current_byte};
         cost_ring_[static_cast<std::size_t>(literal_destination) & 0x1ff] =
            literal_cost;
      }

      // Forward-processing gate. Strategies 1 and 2 only differ here.
      bool normal;
      if (strategy_ == 2) {
         normal = (pos_ + 1 >= farthest_);
      } else {
         const bool near =
            (strategy_ == 0) || (pos_ + 0x102 >= farthest_);
         const bool within_ring =
            (pos_ + 0x105 - last_committed_ <= 0x3ffff);
         normal = (near && within_ring) || (pos_ + 1 >= farthest_);
      }

      if (normal) {
         ++pos_;
         if (farthest_ != pos_) {
            continue;
         }
      } else {
         fast_forward(pos_ + 1, farthest_);
      }

      commit(result, tables);
      last_committed_ = pos_;
      if (pos_ >= limit) {
         break;
      }
   }

   // The target resets the rolling cost ring at every deflate block boundary
   // and seeds the new block's start position as reachable.
   cost_ring_.fill(kUnreachable);
   cost_ring_[static_cast<std::size_t>(pos_) & 0x1ff] = kInitialCost;
   return result;
}

} // namespace kzip
