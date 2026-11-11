#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace kzip {

class Strategy12Parser;

constexpr std::size_t kLengthSymbols = 29;
constexpr std::size_t kDistanceCodes = 30;
constexpr std::size_t kLitLenAlphabet = 286;
constexpr std::size_t kDistanceAlphabet = 30;
constexpr std::size_t kTokenSlots = 320;

struct EncoderToken {
   enum class Kind : std::uint8_t { Literal, Match };
   Kind kind{Kind::Literal};
   std::uint16_t value{};    // literal byte or match length
   std::uint16_t distance{}; // zero for literals
};

struct EncoderOptions {
   // Parser strategy selected by -sN. Strategies 0, 1, and 2 share the match
   // search; strategy 3 is literal-only.
   int strategy{0};
   bool empty_tree_flag{false};
   // Strategy 3 skips match search and hash insertion, leaving literal
   // transitions only.
   bool literal_only{false};
   // Seed each block's first Huffman tables from the target's additive-feedback
   // generator, reseeded from `time()` per block.
   bool randomize_tables{false};
   // Optional fixed seed for reproducible differential runs.
   std::optional<std::uint32_t> random_seed;
   // -nS: keep the deflate result even when the size-based stored fallback
   // would normally replace it.
   bool suppress_stored_fallback{false};
   // Defensive convergence cap. The target stops after 1,024 improving passes
   // (`DAT_006d5084 > 0x3ff`); the value here is the equivalent post-increment
   // count. Exposed so guard probes can lower it in lockstep with a patched
   // target binary.
   std::size_t max_passes{1024};
   // Test-only hooks mirroring the strategy-1/2 parser's run-cache saturation
   // cap (`cmp r12d,0x7fff`) and packed-arena base reset threshold
   // (`cmp eax,0x7fff00`). Lowered in lockstep with a patched target so the
   // stress paths can be exercised on small inputs.
   std::int32_t run_cap{0x7fff};
   std::int32_t arena_base_limit{0x7fff00};
};

struct EncoderBlockResult {
   std::vector<EncoderToken> tokens;
   std::array<std::uint8_t, kTokenSlots> lengths{};
};

struct EncoderPassResult {
   std::vector<std::uint8_t> bytes;
   std::vector<EncoderBlockResult> blocks;
   std::vector<std::array<std::uint8_t, kTokenSlots>> new_lengths;
   std::int64_t delta{};
   std::size_t size{};
};

// One entry per emitted optimization pass, used to reproduce the target's
// verbose/normal console output. `size` is the pass's compressed payload size
// (the value behind "Fileleng" and the progress line) and `delta` is the
// accumulated bit savings reported by the Huffman rebuild. `saved_before` is
// the target's DAT_006d5080 flag at the start of the pass: it is true only for
// the extra re-emit pass after restoring the best tables. `stored` marks the
// pass that decided to write the member stored instead of deflated.
struct EncoderPassLog {
   std::size_t size{};
   std::int64_t delta{};
   bool saved_before{};
   bool stored{};
};

struct EncoderResult {
   std::vector<std::uint8_t> deflate;
   std::vector<EncoderBlockResult> blocks;
   std::vector<std::size_t> block_ends;
   std::vector<std::size_t> pass_sizes;
   std::vector<EncoderPassLog> pass_log;
   std::size_t pass_count{};
   std::vector<std::array<std::uint8_t, kTokenSlots>> final_lengths;
   // The target stops the optimization loop early when it decides to store the
   // member instead of using the deflate stream. `deflate` still holds the last
   // reconstructed pass for inspection.
   bool stored{false};
};

// Initial literal/length and distance code lengths used by the target for
// strategies 0..2 before the first optimization pass.
std::array<std::uint8_t, kTokenSlots> initial_code_lengths();

// Initial code lengths used by strategy 3: literal codes are mostly 8 bits,
// symbol 0x80 and EOB are 9 bits, and every length/distance slot is zero.
std::array<std::uint8_t, kTokenSlots> initial_literal_code_lengths();

// Randomized first-pass code lengths used by -rn. The target seeds a 31-word
// additive-feedback generator with `time()`, warms it for 310 outputs, then
// builds both trees from `(value & 0x7fff) + 0x4000` frequencies.
std::array<std::uint8_t, kTokenSlots> random_initial_code_lengths(
   std::uint32_t seed);

// Compress `input` with the reconstructed strategy-0 iterative encoder.
// `block_ends` are exclusive offsets. An empty span means one block covering
// the whole input.
EncoderResult encode_strategy0(std::span<const std::uint8_t> input,
                               std::span<const std::size_t> block_ends,
                               const EncoderOptions& options = {});

// Run one optimization pass with the supplied per-block starting lengths.
EncoderPassResult run_strategy0_pass(
   std::span<const std::uint8_t> input,
   std::span<const std::size_t> block_ends,
   std::span<const std::array<std::uint8_t, kTokenSlots>> start_lengths,
   const EncoderOptions& options = {},
   Strategy12Parser* shared_parser = nullptr);

// Convergence fingerprint used by the target over the start-of-pass lengths.
std::uint32_t length_fingerprint(
   std::span<const std::array<std::uint8_t, kTokenSlots>> start_lengths);

} // namespace kzip
