#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kzip {

constexpr std::size_t kLitLenSymbols = 286;
constexpr std::size_t kDistanceSymbols = 30;
constexpr std::size_t kContextLengthSlots = 320;

// LSB-first bit writer matching 0x4030d0. Bytes are appended through the same
// pending-accumulator rule as the target: append the low byte whenever at
// least eight bits are pending.
class BitWriter {
public:
   void write_bits(std::uint32_t value, std::uint32_t bit_count) noexcept;
   void align_to_byte() noexcept;
   void reset() noexcept;

   [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept {
      return bytes_;
   }
   [[nodiscard]] std::vector<std::uint8_t>& bytes() noexcept { return bytes_; }
   [[nodiscard]] std::uint32_t pending_bits() const noexcept {
      return bit_count_;
   }

private:
   std::vector<std::uint8_t> bytes_;
   std::uint32_t accumulator_{};
   std::uint32_t bit_count_{};
};

// RLE item used by the dynamic-header code-length encoder.
struct CodeLengthRleItem {
   std::uint8_t symbol{};
   std::uint8_t extra_bits{};
   std::uint16_t extra_value{};
};

// Encode the 286 literal/length lengths followed by the 30 distance lengths
// (distance lengths live at indices 288..317 of `lengths`) exactly as
// kzip_write_dynamic_header at 0x4035a0.
void write_dynamic_header(BitWriter& writer,
                          std::span<const std::uint8_t> lengths);

// Exposed for tests and for the differential harness.
std::vector<CodeLengthRleItem> encode_code_length_rle(
   std::span<const std::uint8_t> combined_lengths);

} // namespace kzip
