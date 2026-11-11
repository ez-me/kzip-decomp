#include "deflate_writer.hpp"

#include "huffman.hpp"

#include <algorithm>
#include <array>

namespace kzip {
namespace {

constexpr std::array<std::uint8_t, 19> kCodeLengthOrder = {
   16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

} // namespace

void BitWriter::write_bits(std::uint32_t value, std::uint32_t bit_count) noexcept {
   accumulator_ |= value << bit_count_;
   bit_count_ += bit_count;
   while (bit_count_ > 7) {
      bytes_.push_back(static_cast<std::uint8_t>(accumulator_ & 0xff));
      accumulator_ >>= 8;
      bit_count_ -= 8;
   }
}

void BitWriter::align_to_byte() noexcept {
   write_bits(0, (0u - bit_count_) & 7u);
}

void BitWriter::reset() noexcept {
   bytes_.clear();
   accumulator_ = 0;
   bit_count_ = 0;
}

std::vector<CodeLengthRleItem> encode_code_length_rle(
   std::span<const std::uint8_t> combined_lengths) {
   std::vector<CodeLengthRleItem> items;
   items.reserve(combined_lengths.size());

   std::size_t position = 0;
   const std::size_t total = combined_lengths.size();
   while (position < total) {
      const std::uint8_t value = combined_lengths[position];
      std::size_t run = 1;
      if (position + 1 < total && combined_lengths[position + 1] == value) {
         run = 2;
         while (position + run < total && combined_lengths[position + run] == value) {
            ++run;
         }
      }

      const bool previous_is_value =
         position > 0 && combined_lengths[position - 1] == value;

      if (value == 0 && run > 2) {
         if (run < 0x8b) {
            if (run > 10) {
               items.push_back({18, 7, static_cast<std::uint16_t>(run - 11)});
               position += run;
            } else {
               if (!previous_is_value || run > 6) {
                  items.push_back({17, 3, static_cast<std::uint16_t>(run - 3)});
               } else {
                  items.push_back({16, 2, static_cast<std::uint16_t>(run - 3)});
               }
               position += run;
            }
         } else {
            items.push_back({18, 7, 127});
            position += 138;
         }
         continue;
      }

      if (!previous_is_value) {
         if (run < 4) {
            items.push_back({value, 0, 0});
            ++position;
         } else {
            items.push_back({value, 0, 0});
            const std::size_t repeats = run - 1;
            const std::size_t chunk =
               (repeats % 6 == 2) ? 5 : std::min<std::size_t>(repeats, 6);
            items.push_back({16, 2, static_cast<std::uint16_t>(chunk - 3)});
            position += 1 + chunk;
         }
      } else {
         if (run < 3) {
            items.push_back({value, 0, 0});
            ++position;
         } else {
            const std::size_t chunk =
               (run % 6 == 2) ? 5 : std::min<std::size_t>(run, 6);
            items.push_back({16, 2, static_cast<std::uint16_t>(chunk - 3)});
            position += chunk;
         }
      }
   }

   return items;
}

void write_dynamic_header(BitWriter& writer,
                          std::span<const std::uint8_t> lengths) {
   if (lengths.size() < kContextLengthSlots) {
      return;
   }

   std::size_t last_litlen = 0;
   bool has_litlen = false;
   for (std::size_t index = kLitLenSymbols; index-- > 257;) {
      if (lengths[index] != 0) {
         last_litlen = index;
         has_litlen = true;
         break;
      }
   }

   std::size_t hlit_count = has_litlen ? last_litlen + 1 : 257;
   const std::uint32_t hlit_field =
      static_cast<std::uint32_t>(hlit_count - 257);

   std::size_t last_distance = 0;
   bool has_distance = false;
   for (std::size_t index = kDistanceSymbols; index-- > 0;) {
      if (lengths[288 + index] != 0) {
         last_distance = index;
         has_distance = true;
         break;
      }
   }

   const std::size_t hdist_count = has_distance ? last_distance + 1 : 1;
   const std::uint32_t hdist_field =
      static_cast<std::uint32_t>(hdist_count - 1);

   writer.write_bits(hlit_field, 5);
   writer.write_bits(hdist_field, 5);

   std::vector<std::uint8_t> combined(hlit_count + hdist_count);
   std::copy_n(lengths.begin(), hlit_count, combined.begin());
   for (std::size_t index = 0; index < hdist_count; ++index) {
      combined[hlit_count + index] = lengths[288 + index];
   }

   const auto items = encode_code_length_rle(combined);

   std::array<std::int32_t, 19> frequencies{};
   for (const auto& item : items) {
      ++frequencies[item.symbol];
   }

   const auto code_lengths =
      build_huffman_lengths(frequencies, 8, false);
   std::array<std::uint32_t, 19> codes{};
   assign_canonical_codes(code_lengths, codes);

   std::size_t hclen_position = 19;
   while (hclen_position > 4 &&
          code_lengths[kCodeLengthOrder[hclen_position - 1]] == 0) {
      --hclen_position;
   }
   const std::uint32_t hclen_field =
      static_cast<std::uint32_t>(hclen_position - 4);
   writer.write_bits(hclen_field, 4);

   for (std::size_t index = 0; index < hclen_position; ++index) {
      writer.write_bits(code_lengths[kCodeLengthOrder[index]], 3);
   }

   for (const auto& item : items) {
      writer.write_bits(codes[item.symbol], code_lengths[item.symbol]);
      if (item.extra_bits != 0) {
         writer.write_bits(item.extra_value, item.extra_bits);
      }
   }
}

} // namespace kzip
