#include "block_splitter.hpp"

#include "split_score_model.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace kzip {
namespace {

constexpr std::size_t kMaxContexts = 0x800;

struct BlockContext {
   std::size_t start{};
   std::size_t end{};
   std::ptrdiff_t parent{};
   double score{};
   std::size_t cursor{};
};

// One scorer pass over a context. Returns true when the target would accept a
// split and updates the context in place.
bool try_split(std::span<const std::uint8_t> input,
               std::vector<BlockContext>& contexts,
               std::size_t index,
               double threshold) {
   BlockContext& context = contexts[index];
   const std::size_t length = context.end - context.start;
   if (length <= 1) {
      return false;
   }

   const SplitScoreResult result =
      score_split(input.subspan(context.start, length));
   if (result.offset == 0 || !(threshold < result.score)) {
      return false;
   }

   const std::size_t split_at = context.start + result.offset;
   const std::size_t old_end = context.end;
   context.end = split_at;
   contexts.push_back(BlockContext{split_at, old_end,
                                   static_cast<std::ptrdiff_t>(index), 0.0, 0});
   return true;
}

std::vector<std::size_t> sorted_ends(const std::vector<BlockContext>& contexts) {
   std::vector<std::size_t> ends;
   ends.reserve(contexts.size());
   for (const BlockContext& context : contexts) {
      ends.push_back(context.end);
   }
   std::sort(ends.begin(), ends.end());
   return ends;
}

std::vector<std::size_t> choose_automatic(std::span<const std::uint8_t> input,
                                          double threshold) {
   std::vector<BlockContext> contexts;
   contexts.reserve(kMaxContexts);
   contexts.push_back(BlockContext{0, input.size(), -1, 0.0, 0});

   std::size_t count = 1;
   for (std::size_t outer = 0; outer < count; ++outer) {
      std::size_t current = outer;

      // The target processes the current context, then its parent when the
      // parent link is valid. A two-step parity counter (r8 = 2, 1, 0) stops
      // the walk after the parent. The counter is decremented before the
      // parity test, matching `sub r8d,1` at 0x401ddc.
      for (int level = 2;;) {
         const bool accepted = try_split(input, contexts, current, threshold);
         if (accepted) {
            ++count;
            if (count >= kMaxContexts) {
               return sorted_ends(contexts);
            }
         }

         --level;
         const std::ptrdiff_t parent = contexts[current].parent;
         if (parent < 0 || (level & 1) == 0) {
            break;
         }
         current = static_cast<std::size_t>(parent);
      }
   }

   return sorted_ends(contexts);
}

std::vector<std::size_t> choose_explicit(std::span<const std::uint8_t> input,
                                         std::size_t requested) {
   if (requested < 2 || input.size() < 2) {
      return {input.size()};
   }
   if (requested > kMaxContexts) {
      requested = kMaxContexts;
   }

   std::vector<BlockContext> contexts;
   contexts.reserve(requested);
   contexts.push_back(BlockContext{0, input.size(), 0, 0.0, 0});

   {
      const SplitScoreResult result = score_split(input);
      contexts[0].score = result.score;
      contexts[0].cursor = result.offset;
   }

   // Each iteration subdivides the pending context with the largest stored
   // score. The scan starts with context zero as the incumbent and walks from
   // the newest context down to zero with a strict `>` update.
   for (std::size_t created = 0; created != requested - 1; ++created) {
      std::size_t selected = 0;
      double best_score = contexts[0].score;
      for (std::size_t i = created + 1; i-- > 0;) {
         if (best_score < contexts[i].score) {
            best_score = contexts[i].score;
            selected = i;
         }
      }

      if (contexts[selected].end - contexts[selected].start < 2) {
         break;
      }

      const std::size_t split_at = contexts[selected].cursor;
      const std::size_t old_end = contexts[selected].end;
      const std::size_t new_index = created + 1;
      contexts.push_back(BlockContext{split_at, old_end, 0, 0.0, 0});

      // Prepare the right child's pending split and score.
      if (contexts[new_index].end - contexts[new_index].start > 1) {
         const SplitScoreResult result = score_split(
            input.subspan(contexts[new_index].start,
                          contexts[new_index].end - contexts[new_index].start));
         contexts[new_index].score = result.score;
         contexts[new_index].cursor =
            contexts[new_index].start + result.offset;
      } else {
         contexts[new_index].cursor = contexts[new_index].start;
      }

      // The selected context becomes the left child.
      contexts[selected].end = split_at;
      if (split_at - contexts[selected].start > 1) {
         const SplitScoreResult result = score_split(
            input.subspan(contexts[selected].start,
                          split_at - contexts[selected].start));
         contexts[selected].score = result.score;
         contexts[selected].cursor =
            contexts[selected].start + result.offset;
      } else {
         contexts[selected].cursor = contexts[selected].start;
      }
   }

   return sorted_ends(contexts);
}

} // namespace

std::vector<std::size_t> choose_block_ends(std::span<const std::uint8_t> input,
                                           const SplitOptions& options) {
   // The target checks the threshold before the requested block count, so a
   // non-positive -b disables splitting even when -n is present.
   if (options.threshold <= 0.0 || options.block_count == 1) {
      return {input.size()};
   }
   if (options.block_count < 1) {
      return choose_automatic(input, options.threshold);
   }
   return choose_explicit(input, options.block_count);
}

} // namespace kzip
