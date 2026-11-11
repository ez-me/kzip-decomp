#include "split_score_model.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace kzip {
namespace {

constexpr double kRationalA = 0.03160502036941208;
constexpr double kRationalB = 0.19823983692603997;
constexpr double kOutputScale = -0.18033688011112042;
constexpr double kInitialBest = 1.0e32;
constexpr double kShortInputFallback = 0.5;

// This is the instruction-level shape of the repeated SSE2 recurrence.
// Keeping the same sequence of operations avoids simplifying away small
// differences near split-score ties.
double log_increment(std::size_t m) noexcept {
   const double md = static_cast<double>(m);
   const double product = md * (md + 1.0);
   double numerator = (md + 0.5) * (product + kRationalA);
   const double denominator = product * (product + kRationalB);
   numerator /= denominator;
   return numerator;
}

} // namespace

SplitScoreResult score_split(std::span<const std::uint8_t> bytes) {
   const std::size_t n = bytes.size();
   if (n <= 1) {
      return {0, kShortInputFallback * kOutputScale, kShortInputFallback};
   }

   std::array<std::uint32_t, 256> total{};
   for (const std::uint8_t byte : bytes) {
      ++total[byte];
   }

   std::array<std::uint32_t, 256> prefix_count{};
   std::array<double, 256> prefix_ratio_sum{};
   std::array<double, 256> suffix_log{};
   std::array<double, 256> symbol_state{};
   for (std::size_t s = 0; s != total.size(); ++s) {
      if (total[s] != 0) {
         suffix_log[s] = std::log(static_cast<double>(total[s]));
         symbol_state[s] = static_cast<double>(total[s]) * suffix_log[s];
      }
   }

   double prefix_log = 0.0;
   double remaining_log = std::log(static_cast<double>(n - 1));
   double old_position_term = static_cast<double>(n) * std::log(static_cast<double>(n));
   double statistic = 0.0;
   double best_statistic = kInitialBest;
   std::size_t best_offset = 0;

   for (std::size_t j = 1; j < n; ++j) {
      const std::size_t s = bytes[j - 1];
      const std::size_t c = static_cast<std::size_t>(prefix_count[s]) + 1;
      const std::size_t q = static_cast<std::size_t>(total[s]) - c;

      statistic += symbol_state[s] - old_position_term;

      // Prefix ratio update is unconditional, including the final occurrence.
      prefix_ratio_sum[s] += log_increment(c);
      if (q != 0) {
         suffix_log[s] -= log_increment(q);
      }
      symbol_state[s] = static_cast<double>(c) * prefix_ratio_sum[s]
                      + static_cast<double>(q) * suffix_log[s];
      prefix_count[s] = static_cast<std::uint32_t>(c);

      const double new_position_term =
         static_cast<double>(j) * prefix_log
         + static_cast<double>(n - j) * remaining_log;
      statistic -= symbol_state[s] - new_position_term;
      old_position_term = new_position_term;

      // The binary uses UCOMISD + JBE: equal scores retain the earlier offset.
      if (best_statistic > statistic) {
         best_statistic = statistic;
         best_offset = j;
      }

      // These scalar log states are advanced after scoring the current prefix.
      prefix_log += log_increment(j);
      remaining_log -= log_increment(n - j);
   }

   return {best_offset, best_statistic * kOutputScale, best_statistic};
}

} // namespace kzip
