#include "huffman.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace kzip {
namespace {

constexpr std::uint32_t kSentinelNode = 0x23f;
constexpr std::size_t kWeightSlots = 0x240;

class HuffmanHeap {
public:
   explicit HuffmanHeap(std::size_t leaf_count)
      : heap_(std::max<std::size_t>(leaf_count + 2, 4), kSentinelNode) {
      heap_[0] = kSentinelNode;
      heap_[1] = kSentinelNode;
   }

   std::size_t size() const noexcept { return size_; }

   void insert(std::span<const std::int32_t> weights, std::int32_t node) {
      std::size_t position = ++size_;
      const std::int32_t weight = weights[static_cast<std::size_t>(node)];
      while (position > 1) {
         const std::size_t parent = position >> 1;
         const std::int32_t parent_weight =
            weights[static_cast<std::size_t>(heap_[parent])];
         if (!(weight < parent_weight)) {
            break;
         }
         heap_[position] = heap_[parent];
         position = parent;
      }
      heap_[position] = node;
      heap_[size_ + 1] = kSentinelNode;
   }

   std::int32_t remove(std::span<const std::int32_t> weights) {
      const std::int32_t result = heap_[1];
      const std::int32_t last = heap_[size_];
      --size_;
      if (size_ <= 1) {
         heap_[1] = last;
         return result;
      }

      const std::int32_t last_weight = weights[static_cast<std::size_t>(last)];
      // The target unconditionally inspects heap[2] and heap[3] here. When the
      // right slot is outside the remaining heap it still contains the last
      // element, whose weight equals last_weight, so the stop test below
      // rejects it without changing the result.
      std::size_t child = 2;
      if (weights[static_cast<std::size_t>(heap_[3])] <
          weights[static_cast<std::size_t>(heap_[2])]) {
         child = 3;
      }
      if (!(weights[static_cast<std::size_t>(heap_[child])] < last_weight)) {
         heap_[1] = last;
         return result;
      }

      std::size_t position = 1;
      for (;;) {
         heap_[position] = heap_[child];
         const std::size_t left = child << 1;
         position = child;
         if (size_ < left) {
            break;
         }
         std::size_t next = left;
         if (weights[static_cast<std::size_t>(heap_[left + 1])] <
             weights[static_cast<std::size_t>(heap_[left])]) {
            next = left + 1;
         }
         if (!(weights[static_cast<std::size_t>(heap_[next])] < last_weight)) {
            break;
         }
         child = next;
      }
      heap_[position] = last;
      return result;
   }

private:
   std::vector<std::int32_t> heap_;
   std::size_t size_{};
};

std::int32_t adjusted_weight(std::int32_t frequency,
                             std::int32_t add,
                             std::uint32_t shift) noexcept {
   // The target uses a signed 32-bit add followed by an arithmetic shift.
   return static_cast<std::int32_t>(
      (static_cast<std::uint32_t>(frequency) + static_cast<std::uint32_t>(add)) >>
      (shift & 31));
}

void sort_equal_frequency_runs(std::span<const std::int32_t> frequencies,
                               std::int32_t add,
                               std::uint32_t shift,
                               std::span<std::uint8_t> lengths) {
   const std::size_t n = frequencies.size();
   for (std::size_t i = 1; i < n; ++i) {
      if (frequencies[i - 1] <= 0) {
         continue;
      }
      const std::int32_t run_value = adjusted_weight(frequencies[i], add, shift);
      if (adjusted_weight(frequencies[i - 1], add, shift) != run_value) {
         continue;
      }

      std::size_t end = i + 1;
      while (end < n &&
             adjusted_weight(frequencies[end], add, shift) == run_value) {
         ++end;
      }

      // Stable insertion sort over [i - 1, end).
      for (std::size_t current = i; current < end; ++current) {
         const std::uint8_t key = lengths[current];
         std::size_t position = current;
         while (position > i - 1 && lengths[position - 1] > key) {
            lengths[position] = lengths[position - 1];
            --position;
         }
         lengths[position] = key;
      }
   }
}

} // namespace

std::vector<std::uint8_t> build_huffman_lengths(
   std::span<const std::int32_t> frequencies,
   std::uint32_t max_bits,
   bool empty_tree_flag) {
   const std::size_t n = frequencies.size();
   std::vector<std::uint8_t> lengths(n, 0);
   std::vector<std::int32_t> weights(kWeightSlots, 0);
   std::vector<std::int32_t> tree(std::max<std::size_t>(n * 2 + 2, 2), 0);
   weights[kSentinelNode] = std::numeric_limits<std::int32_t>::min();

   std::int32_t add = 0;
   std::uint32_t shift = 0;

   for (;;) {
      std::fill(weights.begin(), weights.end(), 0);
      weights[kSentinelNode] = std::numeric_limits<std::int32_t>::min();
      HuffmanHeap heap(n);

      std::size_t active = 0;
      for (std::size_t symbol = 0; symbol < n; ++symbol) {
         const std::int32_t frequency = frequencies[symbol];
         std::int32_t weight;
         if (frequency > 0) {
            weight = adjusted_weight(frequency, add, shift);
            ++active;
         } else {
            weight = std::numeric_limits<std::int32_t>::max();
         }
         weights[symbol] = weight;
         heap.insert(weights, static_cast<std::int32_t>(symbol));
      }

      if (active < 2) {
         std::fill(lengths.begin(), lengths.end(), std::uint8_t{0});
         const std::int32_t leaf = heap.remove(weights);
         lengths[static_cast<std::size_t>(leaf)] =
            static_cast<std::uint8_t>(active);
         if (empty_tree_flag && active == 0) {
            lengths[0] = 1;
         }
         return lengths;
      }

      const std::size_t first_internal = n;
      const std::size_t root = n + active - 2;
      for (std::size_t node = first_internal; node <= root; ++node) {
         const std::int32_t left = heap.remove(weights);
         const std::int32_t right = heap.remove(weights);
         tree[static_cast<std::size_t>(left)] = static_cast<std::int32_t>(node);
         tree[static_cast<std::size_t>(right)] = static_cast<std::int32_t>(node);
         weights[node] = weights[static_cast<std::size_t>(left)] +
                         weights[static_cast<std::size_t>(right)];
         heap.insert(weights, static_cast<std::int32_t>(node));
      }

      weights[root] = 0;
      if (active >= 3) {
         for (std::size_t node = n + active - 3;; --node) {
            weights[node] = weights[static_cast<std::size_t>(tree[node])] + 1;
            if (node == n) {
               break;
            }
         }
      }

      std::uint32_t maximum_depth = 0;
      for (std::size_t symbol = n; symbol-- > 0;) {
         if (frequencies[symbol] > 0) {
            const std::uint32_t length = static_cast<std::uint32_t>(
               weights[static_cast<std::size_t>(tree[symbol])] + 1);
            lengths[symbol] = static_cast<std::uint8_t>(length);
            maximum_depth = std::max(maximum_depth, length);
         } else {
            lengths[symbol] = 0;
         }
      }

      if (maximum_depth < max_bits) {
         sort_equal_frequency_runs(frequencies, add, shift, lengths);
         return lengths;
      }

      add += static_cast<std::int32_t>(1u << shift);
      ++shift;
   }
}

void assign_canonical_codes(std::span<const std::uint8_t> lengths,
                            std::span<std::uint32_t> codes) {
   std::array<std::uint32_t, 32> counts{};
   for (std::size_t index = lengths.size(); index-- > 0;) {
      --counts[lengths[index]];
   }

   std::uint32_t code = 0;
   for (std::size_t length = 1; length < counts.size(); ++length) {
      const std::uint32_t count = counts[length];
      counts[length] = code;
      code = (code - count) * 2;
   }

   for (std::size_t index = 0; index < lengths.size(); ++index) {
      const std::uint32_t length = lengths[index];
      if (length == 0) {
         continue;
      }

      const std::uint32_t value = counts[length];
      counts[length] = value + 1;

      std::uint32_t reversed = 0;
      const std::uint32_t limit = 1u << length;
      if (limit >= 2) {
         std::uint32_t mask = 1;
         do {
            reversed = reversed * 2 + ((value & mask) != 0 ? 1u : 0u);
            mask *= 2;
         } while (mask < limit);
      }
      codes[index] = reversed;
   }
}

} // namespace kzip
