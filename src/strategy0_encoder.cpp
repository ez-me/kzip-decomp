#include "strategy0_encoder.hpp"

#include "deflate_writer.hpp"
#include "huffman.hpp"
#include "parser_cost.hpp"
#include "strategy12_parser.hpp"
#include "strategy0_model.hpp"

#include <algorithm>
#include <array>
#include <ctime>
#include <limits>
#include <stdexcept>

namespace kzip {
namespace {

constexpr std::uint32_t kInitialCost = 0x80000000u;

// DEFLATE length and distance mappings. These mirror the tables built by
// kzip_init_context_tables (0x402c20) through parser_cost.cpp.
struct EmissionTables {
   ParserCostTable parser{};
   std::array<std::uint8_t, kLitLenAlphabet> litlen_bits{};
   std::array<std::uint8_t, kDistanceAlphabet> distance_bits{};
   std::array<std::uint32_t, kLitLenAlphabet> litlen_codes{};
   std::array<std::uint32_t, kDistanceAlphabet> distance_codes{};
};

EmissionTables make_emission_tables(
   const std::array<std::uint8_t, kTokenSlots>& lengths) {
   EmissionTables tables{};
   for (std::size_t i = 0; i < kLitLenAlphabet; ++i) {
      tables.litlen_bits[i] = lengths[i];
   }
   for (std::size_t i = 0; i < kDistanceAlphabet; ++i) {
      tables.distance_bits[i] = lengths[288 + i];
   }
   tables.parser = make_parser_cost_table(tables.litlen_bits, tables.distance_bits);
   assign_canonical_codes(tables.litlen_bits, tables.litlen_codes);
   assign_canonical_codes(tables.distance_bits, tables.distance_codes);
   return tables;
}

DeflateCostTables make_deflate_cost_tables(const EmissionTables& emission) {
   DeflateCostTables tables{};
   for (std::size_t i = 0; i < 256; ++i) {
      tables.length_symbol[i] = emission.parser.length_to_symbol[i];
   }
   for (std::size_t i = 0; i < kMaxDistance; ++i) {
      tables.distance_symbol[i] = emission.parser.distance_to_symbol[i];
   }
   for (std::size_t i = 0; i < kTokenCostSymbols; ++i) {
      tables.token_cost[i] = emission.parser.token_cost[i];
   }
   for (std::size_t i = 0; i < 30; ++i) {
      tables.distance_base[i] = emission.parser.params[i].base_value;
   }
   build_next_cheaper_distance_symbols(tables);
   return tables;
}

inline std::int32_t signed_cost(std::uint32_t value) noexcept {
   return std::bit_cast<std::int32_t>(value);
}

inline std::uint32_t add_cost(std::uint32_t a, std::uint32_t b) noexcept {
   return a + b;
}

// The target keeps reading bytes past the current input length from its 64 KiB
// ring. For the fixtures used here the tail is zero-initialised, so padding the
// match search with zeros reproduces the observed candidate lengths.
inline std::uint8_t byte_at(std::span<const std::uint8_t> input,
                            std::size_t index) noexcept {
   return index < input.size() ? input[index] : 0;
}

inline std::uint32_t word_at(std::span<const std::uint8_t> input,
                             std::size_t index) noexcept {
   return static_cast<std::uint32_t>(byte_at(input, index)) |
      (static_cast<std::uint32_t>(byte_at(input, index + 1)) << 8) |
      (static_cast<std::uint32_t>(byte_at(input, index + 2)) << 16);
}

inline std::uint32_t hash_word(std::uint32_t word) noexcept {
   return ((word >> 12) - word) & 0xfff;
}

bool first_three_match(std::span<const std::uint8_t> input,
                       std::size_t position,
                       std::uint32_t distance) noexcept {
   const std::size_t source = position - distance;
   return byte_at(input, source) == byte_at(input, position) &&
      byte_at(input, source + 1) == byte_at(input, position + 1) &&
      byte_at(input, source + 2) == byte_at(input, position + 2);
}

std::uint32_t match_length(std::span<const std::uint8_t> input,
                           std::size_t position,
                           std::uint32_t distance) noexcept {
   std::uint32_t length = kMinMatch;
   while (length < kMaxMatch &&
          byte_at(input, position + length) ==
             byte_at(input, position + length - distance)) {
      ++length;
   }
   return length;
}

struct BlockParseResult {
   std::vector<EncoderToken> tokens;
   std::array<std::int32_t, kTokenSlots> frequencies{};
};

// Parse one block with the per-block DP and replay the selected path.
BlockParseResult parse_block(std::span<const std::uint8_t> input,
                             std::size_t block_start,
                             std::size_t block_end,
                             std::span<std::uint16_t> hash_heads,
                             std::span<std::uint16_t> hash_predecessors,
                             const DeflateCostTables& tables,
                             const std::array<std::uint8_t, kTokenSlots>& lengths,
                             bool literal_only) {
   const std::size_t block_length = block_end - block_start;
   if (block_length == 0) {
      return {};
   }

   std::vector<std::uint32_t> costs(block_length + 1,
                                    std::numeric_limits<std::uint32_t>::max());
   std::vector<Predecessor> predecessors(block_length + 1);
   costs[0] = kInitialCost;

   BlockParseResult result;
   std::array<std::uint32_t, 256> best_distance_cost{};

   for (std::size_t position = block_start; position < block_end; ++position) {
      const std::size_t relative = position - block_start;
      const std::uint32_t source_cost = costs[relative];
      costs[relative] = std::numeric_limits<std::uint32_t>::max();

      // Every processed position marks its literal as "seen" in the target's
      // frequency array. Selected literals add their count later.
      result.frequencies[byte_at(input, position)] |= 0x80000000u;

      if (!literal_only) {
         best_distance_cost.fill(std::numeric_limits<std::uint32_t>::max());

         const std::uint32_t word = word_at(input, position);
         const std::uint32_t hash = hash_word(word);
         std::uint16_t slot = hash_heads[hash];
         std::uint32_t previous_distance = 0;

         for (;;) {
            const std::uint32_t distance =
               ((static_cast<std::uint32_t>(position) - slot - 1u) & 0x7fffu) +
               1u;
            if (distance <= previous_distance) {
               break;
            }
            previous_distance = distance;
            if (position < distance) {
               break;
            }

            if (first_three_match(input, position, distance)) {
               const std::uint32_t maximum_length =
                  match_length(input, position, distance);

               const std::uint32_t distance_symbol =
                  tables.distance_symbol[distance - 1];
               const std::uint32_t distance_cost =
                  tables.token_cost[distance_symbol];

               // A candidate only improves a length when its complete distance
               // token cost is strictly below the best cost seen so far. The
               // target visits candidates nearest-first, so equal-cost farther
               // candidates are discarded.
               for (std::uint32_t length = kMinMatch;
                    length <= maximum_length;
                    ++length) {
                  const std::uint32_t index = length - 3;
                  if (distance_cost >= best_distance_cost[index]) {
                     continue;
                  }
                  best_distance_cost[index] = distance_cost;

                  if (length > block_length - relative) {
                     continue;
                  }
                  const std::uint32_t length_symbol =
                     tables.length_symbol[index];
                  const std::uint32_t candidate = add_cost(
                     add_cost(source_cost, distance_cost),
                     tables.token_cost[length_symbol]);
                  const std::size_t destination = relative + length;
                  if (signed_cost(candidate) <
                      signed_cost(costs[destination])) {
                     costs[destination] = candidate;
                     predecessors[destination] = {
                        static_cast<std::uint16_t>(length),
                        static_cast<std::uint16_t>(distance)};
                  }
               }

               // Once a candidate reaches the maximum match length, only a
               // later distance symbol with a strictly cheaper code can still
               // improve a length. The target stops the hash walk when no such
               // symbol exists.
               if (maximum_length == kMaxMatch &&
                   tables.next_cheaper_distance_symbol[distance_symbol] >= 30) {
                  break;
               }
            }

            slot = hash_predecessors[slot];
         }

         // Insert this position into the hash chain after the candidate search.
         const std::uint16_t new_slot =
            static_cast<std::uint16_t>(position & 0x7fffu);
         hash_predecessors[new_slot] = hash_heads[hash];
         hash_heads[hash] = new_slot;
      }

      // Literal transition. The target uses <=, so a literal replaces an
      // equal-cost match predecessor.
      const std::size_t literal_destination = relative + 1;
      if (literal_destination <= block_length) {
         const std::uint8_t literal = byte_at(input, position);
         const std::uint32_t candidate =
            add_cost(source_cost, lengths[literal]);
         if (signed_cost(candidate) <= signed_cost(costs[literal_destination])) {
            costs[literal_destination] = candidate;
            predecessors[literal_destination] = {1, literal};
         }
      }
   }

   // Replay the selected path backwards, then reverse it into forward order.
   std::size_t cursor = block_length;
   while (cursor != 0) {
      const Predecessor& predecessor = predecessors[cursor];
      if (predecessor.length == 0 || predecessor.length > cursor) {
         throw std::runtime_error("strategy-0 backtracking lost the parse path");
      }
      EncoderToken token;
      if (predecessor.length == 1) {
         token.kind = EncoderToken::Kind::Literal;
         token.value = predecessor.value;
      } else {
         token.kind = EncoderToken::Kind::Match;
         token.value = predecessor.length;
         token.distance = predecessor.value;
      }
      result.tokens.push_back(token);
      cursor -= predecessor.length;
   }
   std::reverse(result.tokens.begin(), result.tokens.end());

   for (const EncoderToken& token : result.tokens) {
      if (token.kind == EncoderToken::Kind::Literal) {
         ++result.frequencies[token.value];
      } else {
         const std::uint32_t length_symbol =
            tables.length_symbol[token.value - 3];
         const std::uint32_t distance_symbol =
            tables.distance_symbol[token.distance - 1];
         ++result.frequencies[length_symbol + 227]; // 257 + (symbol - 30)
         ++result.frequencies[288 + distance_symbol];
      }
   }

   return result;
}

// kzip_finalize_deflate_block adds one EOB count and then clears the
// high-bit "seen" markers before both the tree build and the weighted
// code-length delta. The delta therefore includes the EOB frequency.
std::array<std::int32_t, kTokenSlots> finalize_frequencies(
   const std::array<std::int32_t, kTokenSlots>& frequencies) {
   std::array<std::int32_t, kTokenSlots> finalized = frequencies;
   ++finalized[256];

   for (std::int32_t& value : finalized) {
      if (value == static_cast<std::int32_t>(0x80000000u)) {
         value = 1;
      } else {
         value &= 0x7fffffff;
      }
   }
   return finalized;
}

std::array<std::uint8_t, kTokenSlots> build_new_lengths(
   const std::array<std::int32_t, kTokenSlots>& finalized,
   bool empty_tree_flag) {
   std::array<std::uint8_t, kTokenSlots> lengths{};

   std::array<std::int32_t, kLitLenAlphabet> litlen_frequencies{};
   for (std::size_t i = 0; i < kLitLenAlphabet; ++i) {
      litlen_frequencies[i] = finalized[i];
   }
   const auto litlen_lengths =
      build_huffman_lengths(litlen_frequencies, 16, empty_tree_flag);
   for (std::size_t i = 0; i < kLitLenAlphabet; ++i) {
      lengths[i] = litlen_lengths[i];
   }

   std::array<std::int32_t, kDistanceAlphabet> distance_frequencies{};
   for (std::size_t i = 0; i < kDistanceAlphabet; ++i) {
      distance_frequencies[i] = finalized[288 + i];
   }
   const auto distance_lengths =
      build_huffman_lengths(distance_frequencies, 16, empty_tree_flag);
   for (std::size_t i = 0; i < kDistanceAlphabet; ++i) {
      lengths[288 + i] = distance_lengths[i];
   }

   return lengths;
}

std::int64_t length_delta(
   const std::array<std::uint8_t, kTokenSlots>& old_lengths,
   const std::array<std::uint8_t, kTokenSlots>& new_lengths,
   const std::array<std::int32_t, kTokenSlots>& finalized_frequencies) {
   std::int64_t delta = 0;
   for (std::size_t i = 0; i < kTokenSlots; ++i) {
      delta += static_cast<std::int64_t>(
                  static_cast<std::int32_t>(old_lengths[i]) -
                  static_cast<std::int32_t>(new_lengths[i])) *
         finalized_frequencies[i];
   }
   return delta;
}

void emit_block(BitWriter& writer,
                const EmissionTables& emission,
                std::span<const EncoderToken> tokens,
                bool final_block) {
   writer.write_bits(final_block ? 1u : 0u, 1);
   writer.write_bits(2, 2);

   std::array<std::uint8_t, kTokenSlots> lengths{};
   for (std::size_t i = 0; i < kLitLenAlphabet; ++i) {
      lengths[i] = emission.litlen_bits[i];
   }
   for (std::size_t i = 0; i < kDistanceAlphabet; ++i) {
      lengths[288 + i] = emission.distance_bits[i];
   }
   write_dynamic_header(writer, lengths);

   for (const EncoderToken& token : tokens) {
      if (token.kind == EncoderToken::Kind::Literal) {
         writer.write_bits(emission.litlen_codes[token.value],
                           emission.litlen_bits[token.value]);
         continue;
      }

      const std::uint32_t length_index =
         static_cast<std::uint32_t>(emission.parser.length_to_symbol[token.value - 3]) -
         30u;
      const std::uint32_t length_symbol = 257u + length_index;
      writer.write_bits(emission.litlen_codes[length_symbol],
                        emission.litlen_bits[length_symbol]);
      const auto& length_param = emission.parser.params[30 + length_index];
      writer.write_bits(static_cast<std::uint32_t>(token.value) -
                           length_param.base_value,
                        length_param.extra_bits);

      const std::uint32_t distance_symbol =
         emission.parser.distance_to_symbol[token.distance - 1];
      writer.write_bits(emission.distance_codes[distance_symbol],
                        emission.distance_bits[distance_symbol]);
      const auto& distance_param = emission.parser.params[distance_symbol];
      writer.write_bits(static_cast<std::uint32_t>(token.distance) -
                           distance_param.base_value,
                        distance_param.extra_bits);
   }

   writer.write_bits(emission.litlen_codes[256], emission.litlen_bits[256]);
}

std::vector<std::size_t> normalized_block_ends(
   std::span<const std::uint8_t> input,
   std::span<const std::size_t> block_ends) {
   if (block_ends.empty()) {
      return {input.size()};
   }
   std::vector<std::size_t> ends(block_ends.begin(), block_ends.end());
   std::size_t previous = 0;
   for (std::size_t end : ends) {
      if (end < previous || end > input.size()) {
         throw std::invalid_argument("strategy-0 block ends are out of range");
      }
      previous = end;
   }
   if (ends.back() != input.size()) {
      throw std::invalid_argument("strategy-0 block ends must cover the input");
   }
   return ends;
}

} // namespace

std::array<std::uint8_t, kTokenSlots> initial_code_lengths() {
   std::array<std::uint8_t, kTokenSlots> lengths{};
   for (std::size_t i = 0; i <= 143; ++i) {
      lengths[i] = 8;
   }
   for (std::size_t i = 144; i <= 255; ++i) {
      lengths[i] = 9;
   }
   for (std::size_t i = 256; i <= 281; ++i) {
      lengths[i] = 7;
   }
   for (std::size_t i = 282; i <= 285; ++i) {
      lengths[i] = 8;
   }
   lengths[288] = 4;
   lengths[289] = 4;
   for (std::size_t i = 290; i <= 317; ++i) {
      lengths[i] = 5;
   }
   return lengths;
}

std::array<std::uint8_t, kTokenSlots> initial_literal_code_lengths() {
   std::array<std::uint8_t, kTokenSlots> lengths{};
   for (std::size_t i = 0; i <= 0x7f; ++i) {
      lengths[i] = 8;
   }
   lengths[0x80] = 9;
   for (std::size_t i = 0x81; i <= 0xff; ++i) {
      lengths[i] = 8;
   }
   lengths[0x100] = 9;
   return lengths;
}

namespace {

// glibc `random()`: 31-word additive feedback with Park-Miller seeding,
// a 310-output warm-up starting at state[3], and `(*f + *r) >> 1` output.
class BsdRandom {
 public:
   explicit BsdRandom(std::uint32_t seed) {
      if (seed == 0) {
         seed = 1;
      }
      state_[0] = static_cast<std::int32_t>(seed);
      for (std::size_t i = 1; i < state_.size(); ++i) {
         // Schrage decomposition, matching glibc's __srandom_r for both
         // positive and high-bit seeds.
         const std::int32_t word = state_[i - 1];
         const std::int32_t hi = word / 127773;
         const std::int32_t lo = word % 127773;
         std::int64_t next = 16807LL * lo - 2836LL * hi;
         if (next < 0) {
            next += 2147483647;
         }
         state_[i] = static_cast<std::int32_t>(next);
      }
      front_ = 3;
      rear_ = 0;
      for (int i = 0; i < 310; ++i) {
         next();
      }
   }

   std::uint32_t next() {
      const std::uint32_t value =
         static_cast<std::uint32_t>(state_[front_]) +
         static_cast<std::uint32_t>(state_[rear_]);
      state_[front_] = static_cast<std::int32_t>(value);
      const std::uint32_t result = (value >> 1) & 0x7fffffffu;
      if (++front_ >= state_.size()) {
         front_ = 0;
         ++rear_;
      } else if (++rear_ >= state_.size()) {
         rear_ = 0;
      }
      return result;
   }

 private:
   std::array<std::int32_t, 31> state_{};
   std::size_t front_{};
   std::size_t rear_{};
};

} // namespace

std::array<std::uint8_t, kTokenSlots> random_initial_code_lengths(
   std::uint32_t seed) {
   BsdRandom random(seed);
   std::array<std::int32_t, kLitLenAlphabet> litlen_frequencies{};
   std::array<std::int32_t, kDistanceAlphabet> distance_frequencies{};
   for (std::int32_t& value : litlen_frequencies) {
      value = static_cast<std::int32_t>((random.next() & 0x7fffu) + 0x4000u);
   }
   for (std::int32_t& value : distance_frequencies) {
      value = static_cast<std::int32_t>((random.next() & 0x7fffu) + 0x4000u);
   }

   std::array<std::uint8_t, kTokenSlots> lengths{};
   const auto litlen_lengths =
      build_huffman_lengths(litlen_frequencies, 16, false);
   for (std::size_t i = 0; i < kLitLenAlphabet; ++i) {
      lengths[i] = litlen_lengths[i];
   }
   const auto distance_lengths =
      build_huffman_lengths(distance_frequencies, 16, false);
   for (std::size_t i = 0; i < kDistanceAlphabet; ++i) {
      lengths[288 + i] = distance_lengths[i];
   }
   return lengths;
}

std::uint32_t length_fingerprint(
   std::span<const std::array<std::uint8_t, kTokenSlots>> start_lengths) {
   static const std::array<std::uint32_t, 256> table = [] {
      std::array<std::uint32_t, 256> generated{};
      for (std::uint32_t i = 0; i < 256; ++i) {
         std::uint32_t value = i;
         for (int bit = 0; bit < 8; ++bit) {
            value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
         }
         generated[i] = value;
      }
      return generated;
   }();

   std::vector<std::uint8_t> bytes;
   bytes.reserve(start_lengths.size() * kTokenSlots);
   for (const auto& block : start_lengths) {
      bytes.insert(bytes.end(), block.begin(), block.end());
   }

   std::uint32_t crc = 0xffffffffu;
   // Walk the byte string from the end in two-byte steps. Each step folds the
   // high byte first, matching the target's reversed table walk.
   for (std::size_t i = bytes.size(); i >= 2; i -= 2) {
      const std::uint32_t index =
         ((static_cast<std::uint32_t>(bytes[i - 1]) * 16u +
           static_cast<std::uint32_t>(bytes[i - 2])) ^
          (crc & 0xffu)) &
         0xffu;
      crc = (crc >> 8) ^ table[index];
   }
   return crc;
}

EncoderPassResult run_strategy0_pass(
   std::span<const std::uint8_t> input,
   std::span<const std::size_t> block_ends,
   std::span<const std::array<std::uint8_t, kTokenSlots>> start_lengths,
   const EncoderOptions& options,
   Strategy12Parser* shared_parser) {
   const std::vector<std::size_t> ends = normalized_block_ends(input, block_ends);
   if (start_lengths.size() != ends.size()) {
      throw std::invalid_argument("strategy-0 start lengths do not match blocks");
   }

   std::vector<std::uint16_t> hash_heads(4096, 0);
   std::vector<std::uint16_t> hash_predecessors(32768, 0);

   const bool use_strategy12 =
      options.strategy == 0 || options.strategy == 1 || options.strategy == 2;
   std::optional<Strategy12Parser> strategy12;
   Strategy12Parser* parser = shared_parser;
   if (use_strategy12 && parser == nullptr) {
      // The target keeps the parser's rolling globals alive across the whole
      // optimization loop; only the per-pass state is reinitialised.
      strategy12.emplace(input, options.strategy, options.run_cap,
                         options.arena_base_limit);
      parser = &*strategy12;
   }
   if (parser != nullptr) {
      parser->begin_pass();
   }

   EncoderPassResult result;
   BitWriter writer;
   std::size_t block_start = 0;

   for (std::size_t block_index = 0; block_index < ends.size(); ++block_index) {
      const std::size_t block_end = ends[block_index];
      const auto& lengths = start_lengths[block_index];
      const EmissionTables emission = make_emission_tables(lengths);
      const DeflateCostTables cost_tables = make_deflate_cost_tables(emission);

      std::vector<EncoderToken> tokens;
      std::array<std::int32_t, kTokenSlots> frequencies{};
      if (use_strategy12) {
         Strategy12BlockResult parsed =
            parser->parse_block(block_end, cost_tables, lengths);
         tokens = std::move(parsed.tokens);
         frequencies = parsed.frequencies;
      } else {
         BlockParseResult parsed = parse_block(
            input, block_start, block_end, hash_heads, hash_predecessors,
            cost_tables, lengths, options.literal_only);
         tokens = std::move(parsed.tokens);
         frequencies = parsed.frequencies;
      }

      EncoderBlockResult block;
      block.tokens = std::move(tokens);
      block.lengths = lengths;

      const auto finalized = finalize_frequencies(frequencies);
      const auto new_lengths =
         build_new_lengths(finalized, options.empty_tree_flag);

      result.delta += length_delta(lengths, new_lengths, finalized);
      result.new_lengths.push_back(new_lengths);
      result.blocks.push_back(std::move(block));

      emit_block(writer, emission, result.blocks.back().tokens,
                 block_index + 1 == ends.size());

      block_start = block_end;
   }

   writer.align_to_byte();
   result.bytes = writer.bytes();
   result.size = result.bytes.size();
   return result;
}

EncoderResult encode_strategy0(
   std::span<const std::uint8_t> input,
   std::span<const std::size_t> block_ends,
   const EncoderOptions& options) {
   const std::vector<std::size_t> ends = normalized_block_ends(input, block_ends);

   std::vector<std::array<std::uint8_t, kTokenSlots>> lengths;
   lengths.reserve(ends.size());
   for (std::size_t i = 0; i < ends.size(); ++i) {
      // The target's -rn branch is gated on strategy < 3, so strategy 3 keeps
      // its literal-only initial table even when -rn is present.
      if (options.randomize_tables && !options.literal_only) {
         const std::uint32_t seed =
            options.random_seed.has_value()
               ? *options.random_seed
               : static_cast<std::uint32_t>(std::time(nullptr));
         lengths.push_back(random_initial_code_lengths(seed));
      } else if (options.literal_only) {
         lengths.push_back(initial_literal_code_lengths());
      } else {
         lengths.push_back(initial_code_lengths());
      }
   }
   std::vector<std::array<std::uint8_t, kTokenSlots>> best_lengths;

   std::optional<Strategy12Parser> shared_parser;
   if (options.strategy == 0 || options.strategy == 1 ||
       options.strategy == 2) {
      // The target's parser state lives in process-wide globals, so it
      // survives the whole optimization loop and is only reinitialised at the
      // start of each pass.
      shared_parser.emplace(input, options.strategy, options.run_cap,
                            options.arena_base_limit);
   }
   Strategy12Parser* parser = shared_parser ? &*shared_parser : nullptr;

   EncoderResult result;
   result.block_ends = ends;

   std::size_t best_size = std::numeric_limits<std::size_t>::max();
   std::vector<std::uint32_t> fingerprints;
   std::vector<std::size_t> recorded_sizes;

   EncoderPassResult last_pass;
   bool stop = false;
   bool restore_best = false;

   while (!stop) {
      const auto start_lengths = lengths;
      EncoderPassResult pass = run_strategy0_pass(input, ends, start_lengths, options, parser);
      result.pass_sizes.push_back(pass.size);
      result.pass_log.push_back(
         EncoderPassLog{pass.size, pass.delta, false, false});
      last_pass = std::move(pass);
      ++result.pass_count;

      if (last_pass.size < best_size) {
         best_size = last_pass.size;
         best_lengths = start_lengths;
      }

      if (last_pass.delta > 0) {
         // The target stops after 1,024 improving passes (0x3ff index cap).
         // This pass is emitted but its size is not recorded by the target, so
         // keep the output while skipping the remaining bookkeeping.
         if (result.pass_count > options.max_passes) {
            stop = true;
            continue;
         }

         const std::uint32_t fingerprint = length_fingerprint(start_lengths);
         bool repeated = false;
         for (std::size_t i = 0; i < fingerprints.size(); ++i) {
            if (fingerprints[i] == fingerprint &&
                recorded_sizes[i] == last_pass.size) {
               repeated = true;
               break;
            }
         }

         if (repeated) {
            if (last_pass.size <= best_size) {
               stop = true;
            } else {
               restore_best = true;
               stop = true;
            }
         } else {
            fingerprints.push_back(fingerprint);
            recorded_sizes.push_back(last_pass.size);
            lengths = last_pass.new_lengths;
         }
      } else {
         // The target checks the stored-versus-deflated decision on a
         // non-improving pass before deciding whether to restore the best
         // tables. When the input is smaller than the best deflate size it
         // writes a stored member and stops without re-emitting.
         const std::size_t best_candidate = std::min(last_pass.size, best_size);
         if (!options.suppress_stored_fallback &&
             input.size() < best_candidate) {
            result.stored = true;
            result.pass_log.back().stored = true;
            stop = true;
         } else if (last_pass.size <= best_size) {
            stop = true;
         } else {
            restore_best = true;
            stop = true;
         }
      }
   }

   if (restore_best) {
      // Re-emit once with the best start-of-pass tables, exactly as the target
      // does after a larger repeated-table or converged pass.
      EncoderPassResult restored =
         run_strategy0_pass(input, ends, best_lengths, options, parser);
      result.pass_sizes.push_back(restored.size);
      result.pass_log.push_back(
         EncoderPassLog{restored.size, restored.delta, true, false});
      ++result.pass_count;
      last_pass = std::move(restored);
   }

   result.deflate = std::move(last_pass.bytes);
   result.blocks = std::move(last_pass.blocks);
   result.final_lengths = std::move(best_lengths);
   return result;
}

} // namespace kzip
