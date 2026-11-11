// Reconstructed KZIP command-line compressor.
//
// This is a behavioral reconstruction of the statically linked `kzip-static`
// binary (SHA-256
// e7bd2055a45df7cd2785d564474916be07205364985924bc2808d6048c0063cd).
// It mirrors the target's option parsing, file-specification expansion,
// per-file block selection, strategy dispatch, ZIP container emission, and
// console output. All five target strategies are reconstructed.
//
// Build:
//   make            # produces build/kzip
//   make clean      # removes build artifacts
//
// Usage:
//   kzip [options] [.ZIP file] [@file or filespec...]
//
//   -b#    split threshold (strtod; default 256.0; <=0 disables splitting)
//   -n#    exact block count (decimal; <1 selects automatic splitting)
//   -nS    disable the stored-entry fallback
//   -q     quiet: suppress all console output
//   -r     recurse into directories
//   -rN    randomized initial Huffman tables (also -rn)
//   -s#    strategy: <1 -> 0, 1-3 -> itself, >=4 -> 4
//   -v     verbose pass/block reporting
//   -y     answer yes to the overwrite prompt
//   -zl121 hidden empty-Huffman-tree switch
//
// Every line the target writes goes through `__fprintf_chk(stdout, ...)` behind
// a single global gate, so `-q` suppresses *all* console output: errors,
// prompts, progress, and totals included. The helpers below reproduce that
// gate and the exact format strings/order decoded from the target.
//
// Module map:
//   strategy0_encoder.*   iterative pass driver, DEFLATE emission, CRC
//   strategy12_parser.*   shared parser for strategies 0/1/2
//   strategy0_model.*     strategy-0 cache rows and relaxation rules
//   parser_cost.*         token-cost and length/distance symbol tables
//   huffman.*             heap, length limiting, canonical codes
//   deflate_writer.*      LSB-first bit writer and dynamic header RLE
//   block_splitter.*      automatic -b and explicit -n boundary selection
//   split_score_model.*   256-bin statistical split scorer
//   zip_writer.*          ZIP container and target-exact output ring
//
// Known intentional divergence: a member whose discovery-time size lies in
// (0x7FFF0000, 0x7FFFFFFF] overflows the target's read-loop accumulator and
// makes strategies 0 and 4 loop forever; this reconstruction terminates.

#include "block_splitter.hpp"
#include "strategy0_encoder.hpp"
#include "zip_writer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {

bool g_quiet = false;

// Reproduces FUN_00405420 (`__fprintf_chk(stdout, 1, fmt, ...)`), including the
// global quiet gate DAT_01fb0220.
void emitf(const char* format, ...) {
   if (g_quiet) {
      return;
   }
   va_list args;
   va_start(args, format);
   std::vfprintf(stdout, format, args);
   va_end(args);
}

// Test-only allocation-failure hook. The target is statically linked, so the
// differential harness forces its malloc/realloc sites to fail by replacing the
// call instructions with `xor eax,eax`. This hook makes the reconstructed CLI
// fail at the same observable point:
//
//   startup-records / startup-buffer
//      one of the two startup allocations at 0x4008c9 / 0x4008f7. The target
//      prints `Error: malloc failed` and exits before discovery, so no archive
//      is created.
//   record-array / name-scan / name-append
//      the three discovery-time reallocations at 0x405745 / 0x40592a /
//      0x405809. They are observationally identical: the target has not
//      printed anything yet, so the run is just the error and exit status.
//   file-buffer
//      the per-file splitter buffer at 0x400bb1, allocated after `  Adding:`
//      when the strategy is not 4 and the -b threshold is positive. Failure
//      prints the short `malloc failed` message and leaves the freshly opened
//      output file empty.
bool alloc_fail_requested(std::string_view site) {
   const char* text = std::getenv("KZIP_ALLOC_FAIL");
   return text != nullptr && site == std::string_view(text);
}

// Models the target's 0x1fb0260 scratch buffer. The target formats its
// non-verbose progress and stored lines into that buffer, and the local-header
// writer leaves bytes 14..25 of the buffer untouched. Those twelve bytes are
// emitted as the local header's CRC/compressed-size/uncompressed-size
// placeholders and are only replaced by the later patch. Above 2 GiB the patch
// can be dropped, so the placeholder becomes observable. Formatting into this
// model at the same points reproduces it.
class ScratchModel {
 public:
   [[nodiscard]] std::array<std::uint8_t, 12> placeholder() const {
      std::array<std::uint8_t, 12> bytes{};
      for (std::size_t index = 0; index < bytes.size(); ++index) {
         bytes[index] = static_cast<std::uint8_t>(scratch_[14 + index]);
      }
      return bytes;
   }

   void format(const char* format, ...) {
      va_list args;
      va_start(args, format);
      std::vsnprintf(scratch_.data(), scratch_.size(), format, args);
      va_end(args);
   }

 private:
   std::array<char, 0x101e> scratch_{};
};

struct Options {
   double threshold{256.0};
   std::size_t block_count{0};
   int strategy{0};
   bool verbose{false};
   bool quiet{false};
   bool recurse{false};
   bool randomize_tables{false};
   bool assume_yes{false};
   bool no_stored_fallback{false};
   bool lz121{false};
};

struct FileRecord {
   std::string prefix;
   std::string name;
   bool directory{false};
   std::int64_t mtime{};
   // Size captured by the discovery-time stat. The target stores the 64-bit
   // st_size into a 32-bit int field, so >= 2 GiB sizes truncate (and can
   // become negative). It then compresses exactly that many bytes.
   std::int32_t size{};
};

void print_usage() {
   static const char kUsage[] =
      "KZIP [options] [.ZIP file] [@file or filespec...]"
      "              by Ken Silverman\n"
      "   KZIP creates .ZIP files that are roughly 1-3% smaller than "
      "\"pkzip -ex ...\"\n\n"
      "Available options:\n"
      "   -r   Recurse sub-directories\n"
      "   -v,q Verbose,Quiet mode\n"
      "   -y   Assume Yes at the 'overwrite file?' prompt\n"
      "   -s#  Select strategy. 0:Xtreme(default), 1:Intense, "
      "2:Longest Match,\n"
      "        3:Huffman Only, 4:Uncompressed\n"
      "   -b#  Block split threshold (lower=more blocks, 0=1 block/file, "
      "default=256)\n"
      "        Good values to try (other than default) are: /b128, /b512, "
      "/b0, /b1024\n"
      "   -n#  Exact number of Huffman blocks (overrides /b#)\n"
      "   -rn  Randomized initial tables (good for many trials with same "
      "options)\n\n"
      "Example: \"kzip save *\" will compress the current directory into "
      "save.zip\n"
      "Ken's official web site: http://advsys.net/ken            "
      "Compiled: Jan 15 2020\n";
   std::fputs(kUsage, stdout);
}

std::string normalize_path(std::string_view text) {
   std::string result(text);
   std::replace(result.begin(), result.end(), '\\', '/');
   return result;
}

std::string join_path(const std::string& prefix, const std::string& name) {
   if (prefix.empty()) {
      return name;
   }
   return prefix + "/" + name;
}

std::string member_name(const FileRecord& record, bool recurse) {
   return recurse ? join_path(record.prefix, record.name) : record.name;
}

char fold_ascii(char value) {
   if (value >= 'a' && value <= 'z') {
      return static_cast<char>(value - 'a' + 'A');
   }
   return value;
}

bool is_separator(char value) {
   return value == '/' || value == '\\';
}

bool chars_match(char name, char pattern) {
   name = fold_ascii(name);
   pattern = fold_ascii(pattern);
   if (is_separator(name) || is_separator(pattern)) {
      return is_separator(name) && is_separator(pattern);
   }
   return name == pattern;
}

// Wildcard matcher at 0x402DB0: ASCII case-insensitive, `?` matches one
// character, `*` backtracks recursively, and slash/backslash are equivalent.
bool wildcard_match(std::string_view name, std::string_view pattern) {
   std::size_t name_index = 0;
   std::size_t pattern_index = 0;
   std::size_t star_name = std::string_view::npos;
   std::size_t star_pattern = std::string_view::npos;

   while (name_index < name.size()) {
      if (pattern_index < pattern.size() && pattern[pattern_index] == '*') {
         star_pattern = pattern_index++;
         star_name = name_index;
      } else if (pattern_index < pattern.size() &&
                 (pattern[pattern_index] == '?' ||
                  chars_match(name[name_index], pattern[pattern_index]))) {
         ++name_index;
         ++pattern_index;
      } else if (star_pattern != std::string_view::npos) {
         pattern_index = star_pattern + 1;
         name_index = ++star_name;
      } else {
         return false;
      }
   }
   while (pattern_index < pattern.size() && pattern[pattern_index] == '*') {
      ++pattern_index;
   }
   return pattern_index == pattern.size();
}

// The target reads each member twice: once with `fread(buffer, size, 1)` into a
// per-file malloc'd buffer used only by the block splitter, and again from the
// start of the file in 64 KiB chunks (`__fread_chk`) while it parses and CRCs
// the data. The 64 KiB chunk buffer is a single static global
// (`DAT_01fb12c0`) that is never cleared, so a short read leaves the previous
// chunk's bytes (or the previous member's bytes) in the tail.
//
// `split_input` models the malloc buffer as the bytes actually read followed by
// zeros. `encode_input` models the persistent chunk buffer exactly. Both have
// the target's truncated 32-bit size.
constexpr std::size_t kChunkSize = 0x10000;

bool read_file(const std::string& path, std::int32_t size,
               std::array<std::uint8_t, kChunkSize>& chunk_buffer,
               std::vector<std::uint8_t>& raw_input,
               std::vector<std::uint8_t>& encode_input) {
   std::FILE* file = std::fopen(path.c_str(), "rb");
   if (file == nullptr) {
      return false;
   }
   raw_input.clear();
   encode_input.clear();
   if (size > 0) {
      const std::size_t count = static_cast<std::size_t>(size);
      raw_input.resize(count);
      // The target issues one fread(buffer, size, 1) and ignores short reads.
      const std::size_t got = std::fread(raw_input.data(), 1, count, file);
      raw_input.resize(got);
      // Encoder buffer: replay the target's 64 KiB chunk loop, including the
      // stale tail left by a short read.
      encode_input.reserve(count);
      std::size_t position = 0;
      std::size_t offset = 0;
      while (offset < count) {
         const std::size_t chunk = std::min(kChunkSize, count - offset);
         const std::size_t available =
            position < got ? std::min(chunk, got - position) : 0;
         if (available != 0) {
            std::memcpy(chunk_buffer.data(), raw_input.data() + position,
                        available);
            position += available;
         }
         encode_input.insert(encode_input.end(), chunk_buffer.begin(),
                             chunk_buffer.begin() +
                                static_cast<std::ptrdiff_t>(chunk));
         offset += chunk;
      }
   }
   std::fclose(file);
   return true;
}

// The target mallocs a per-file splitter buffer, reads into it, and frees it
// after choosing block ends. glibc reuses freed chunks, so a later short read
// can leave the previous member's bytes in the uninitialized tail. This pool
// mirrors that LIFO same-size reuse; larger mmap-backed allocations are treated
// as fresh (zeroed) pages.
class SplitterPool {
public:
   std::vector<std::uint8_t> acquire(std::size_t size) {
      auto found = free_.find(size);
      if (found != free_.end() && !found->second.empty()) {
         std::vector<std::uint8_t> buffer =
            std::move(found->second.back());
         found->second.pop_back();
         return buffer;
      }
      return std::vector<std::uint8_t>(size, 0);
   }

   void release(std::vector<std::uint8_t> buffer) {
      if (buffer.size() > kMmapThreshold) {
         return;
      }
      free_[buffer.size()].push_back(std::move(buffer));
   }

private:
   static constexpr std::size_t kMmapThreshold = 0x20000;
   std::unordered_map<std::size_t,
                      std::vector<std::vector<std::uint8_t>>> free_;
};

void discover(const std::string& spec, bool recurse,
              std::vector<FileRecord>& records) {
   // Discovery-time realloc failures all abort before the output archive is
   // opened, so they only differ in where the target happened to need a larger
   // buffer. The hook is paired with a target capacity patch that guarantees
   // the corresponding site is reached by the harness fixture.
   if (alloc_fail_requested("record-array") ||
       alloc_fail_requested("name-scan") ||
       alloc_fail_requested("name-append")) {
      throw std::bad_alloc();
   }
   const std::string normalized = normalize_path(spec);
   const std::size_t separator = normalized.find_last_of('/');
   const std::string prefix = separator == std::string::npos
                                 ? std::string()
                                 : normalized.substr(0, separator);
   std::string pattern = separator == std::string::npos
                            ? normalized
                            : normalized.substr(separator + 1);
   // The target treats an empty leaf pattern (an empty filespec or a trailing
   // slash such as "sub/") as "*", so it archives everything in that directory.
   if (pattern.empty()) {
      pattern = "*";
   }

   const std::string directory =
      prefix.empty() ? std::string(".") : std::string("./") + prefix;
   DIR* handle = ::opendir(directory.c_str());
   if (handle == nullptr) {
      return;
   }

   while (const dirent* entry = ::readdir(handle)) {
      const std::string name = entry->d_name;
      if (!wildcard_match(name, pattern)) {
         continue;
      }

      const std::string path = join_path(prefix, name);
      struct stat info {};
      const bool stat_ok = ::stat(path.c_str(), &info) == 0;
      // The target tests `st_mode & 0x4000` rather than S_ISDIR. S_IFSOCK is
      // 0xC000, so a Unix socket also matches and is handled as a directory:
      // skipped unless recursion is enabled, and never archived.
      const bool directory_entry = stat_ok && (info.st_mode & 0x4000) != 0;

      if (directory_entry) {
         if (!recurse || name == "." || name == "..") {
            continue;
         }
      }

      FileRecord record;
      record.prefix = prefix;
      record.name = name;
      record.directory = directory_entry;
      record.mtime = stat_ok ? static_cast<std::int64_t>(info.st_mtime) : 0;
      record.size = stat_ok
                       ? std::bit_cast<std::int32_t>(
                            static_cast<std::uint32_t>(info.st_size))
                       : 0;
      records.push_back(std::move(record));
   }

   ::closedir(handle);
}

std::vector<std::string> read_response_file(
   const std::string& path, std::array<std::uint8_t, kChunkSize>& buffer) {
   std::ifstream input(path, std::ios::binary);
   if (!input) {
      return {};
   }
   input.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
   const std::size_t count = static_cast<std::size_t>(input.gcount());

   std::vector<std::string> tokens;
   std::size_t index = 0;
   while (index < count) {
      while (index < count && buffer[index] <= 0x20) {
         ++index;
      }
      const std::size_t start = index;
      while (index < count && buffer[index] > 0x20) {
         ++index;
      }
      if (index > start) {
         tokens.emplace_back(
            reinterpret_cast<const char*>(buffer.data() + start),
            index - start);
         // The target tokenizes the response file in place: the delimiter that
         // ends each token becomes NUL. That same 64 KiB buffer later doubles
         // as the encoder's stream buffer, so this content is observable when
         // a member shrinks below a full chunk.
         if (index < buffer.size()) {
            buffer[index] = 0;
         }
      }
   }
   return tokens;
}

// Reproduces the output-name extension logic at 0x400800. The target looks for
// a dot in the basename; without one it appends ".zip" when the last letter in
// the *whole* name is lowercase, and ".ZIP" otherwise (including names with no
// letters at all).
std::string output_archive_name(std::string name) {
   const std::size_t separator = name.find_last_of("/\\");
   const std::string_view basename =
      separator == std::string::npos
         ? std::string_view(name)
         : std::string_view(name).substr(separator + 1);
   if (basename.find('.') != std::string_view::npos) {
      return name;
   }
   bool lowercase = false;
   for (char value : name) {
      if (value >= 'A' && value <= 'Z') {
         lowercase = false;
      } else if (value >= 'a' && value <= 'z') {
         lowercase = true;
      }
   }
   name += lowercase ? ".zip" : ".ZIP";
   return name;
}

bool parse_double(const char* text, double& value) {
   char* end = nullptr;
   errno = 0;
   value = std::strtod(text, &end);
   return end != text;
}

bool parse_long(const char* text, long& value) {
   char* end = nullptr;
   errno = 0;
   value = std::strtol(text, &end, 10);
   return end != text;
}

kzip::EncoderOptions make_encoder_options(const Options& options) {
   kzip::EncoderOptions encoder_options;
   encoder_options.strategy = options.strategy;
   encoder_options.literal_only = options.strategy == 3;
   encoder_options.randomize_tables = options.randomize_tables;
   encoder_options.suppress_stored_fallback = options.no_stored_fallback;
   // -zl121 sets the Huffman builder's empty-tree global (0x1fb1290).
   encoder_options.empty_tree_flag = options.lz121;
   if (const char* seed_text = std::getenv("KZIP_RANDOM_SEED")) {
      char* end = nullptr;
      const unsigned long value = std::strtoul(seed_text, &end, 10);
      if (end != seed_text) {
         encoder_options.random_seed = static_cast<std::uint32_t>(value);
      }
   }
   // Test-only hook used by the guard probe in tools/compare_zip.py.
   if (const char* passes_text = std::getenv("KZIP_MAX_PASSES")) {
      char* end = nullptr;
      const unsigned long value = std::strtoul(passes_text, &end, 10);
      if (end != passes_text) {
         encoder_options.max_passes = static_cast<std::size_t>(value);
      }   }
   // Test-only hooks used by the cache-edge probe. They lower the strategy-1/2
   // run-cache cap and packed-arena base threshold in lockstep with a patched
   // target binary.
   auto parse_limit = [](const char* name, std::int32_t& target) {
      if (const char* text = std::getenv(name)) {
         char* end = nullptr;
         const long value = std::strtol(text, &end, 0);
         if (end != text) {
            target = static_cast<std::int32_t>(value);
         }
      }
   };
   parse_limit("KZIP_RUN_CAP", encoder_options.run_cap);
   parse_limit("KZIP_ARENA_LIMIT", encoder_options.arena_base_limit);
   return encoder_options;
}

std::string make_progress(std::size_t compressed, std::int64_t uncompressed,
                          ScratchModel& scratch) {
   int ratio = 0;
   if (uncompressed != 0) {
      ratio = 100 - static_cast<int>(
         (static_cast<std::int64_t>(compressed) * 100) / uncompressed);
   }
   char buffer[64];
   std::snprintf(buffer, sizeof buffer, "%9d/%9d bytes (%2d%c)",
                 static_cast<int>(compressed),
                 static_cast<int>(uncompressed), ratio, '%');
   // The target formats the same line into its scratch buffer before printing.
   scratch.format("%9d/%9d bytes (%2d%c)", static_cast<int>(compressed),
                  static_cast<int>(uncompressed), ratio, '%');
   return buffer;
}

// Reproduces the verbose "Block boundaries" banner printed after the splitter
// runs and before the first optimization pass.
void print_block_boundaries(const std::vector<std::size_t>& ends) {
   emitf("   \nBlock boundaries: ");
   if (ends.size() < 2) {
      emitf("none");
   } else {
      for (std::size_t i = 0; i + 1 < ends.size(); ++i) {
         if (i != 0) {
            emitf(",");
         }
         emitf("%x", static_cast<unsigned int>(ends[i]));
      }
   }
   emitf(" (%d block", static_cast<int>(ends.size()));
   if (ends.size() > 1) {
      emitf("s");
   }
   emitf(")\n\n");
   // The verbose banner always ends with an extra blank line.
   emitf("\n");
}

// Replays the console side of one file's optimization passes. Each pass prints
// a verbose "Fileleng" line or, on a converging (delta < 1) non-verbose pass, a
// progress line. A pass that chooses the stored fallback then erases the
// progress text and prints the stored line.
void replay_passes(const std::vector<kzip::EncoderPassLog>& passes,
                   std::int64_t uncompressed, bool verbose,
                   ScratchModel& scratch) {
   std::string progress;
   for (const kzip::EncoderPassLog& pass : passes) {
      if (verbose) {
         emitf("    Fileleng: %7d bytes", static_cast<int>(pass.size));
         if (pass.delta > 0 && !pass.saved_before) {
            const std::int64_t delta = pass.delta;
            const std::int64_t remaining =
               static_cast<std::int64_t>(pass.size) * 8 - delta;
            emitf(" -%7d.%03d -> %7d.%03d\n",
                  static_cast<int>(delta >> 3),
                  static_cast<int>((delta & 7) * 125),
                  static_cast<int>(remaining >> 3),
                  static_cast<int>((remaining & 7) * 125));
         }
      } else if (pass.delta < 1) {
         progress = make_progress(pass.size, uncompressed, scratch);
         emitf("%s", progress.c_str());
      }

      if (pass.stored) {
         // The target's stored-entry helper always formats this line into the
         // scratch buffer, even in verbose mode where it prints a different
         // line to the console.
         scratch.format("Stored :( %9d/%9d bytes ( 0%c)  ",
                        static_cast<int>(uncompressed),
                        static_cast<int>(uncompressed), '%');
         if (verbose) {
            emitf("\n    Stored :(%8d bytes (0%c) ",
                  static_cast<int>(uncompressed), '%');
         } else {
            // The target erases strlen(progress) + 10 characters: the progress
            // line plus the trailing "KSflating ".
            for (std::size_t i = 0; i < progress.size() + 10; ++i) {
               emitf("\b");
            }
            emitf("Stored :( %9d/%9d bytes ( 0%c)  ",
                  static_cast<int>(uncompressed),
                  static_cast<int>(uncompressed), '%');
         }
      }
   }
}

// Strategy 4 never runs the deflate pass loop, so the target's stored-entry
// helper sees an empty progress buffer and erases exactly ten characters.
void replay_stored_only(std::int64_t uncompressed, bool verbose,
                        ScratchModel& scratch) {
   scratch.format("Stored :( %9d/%9d bytes ( 0%c)  ",
                  static_cast<int>(uncompressed),
                  static_cast<int>(uncompressed), '%');
   if (verbose) {
      emitf("\n    Stored :(%8d bytes (0%c) ",
            static_cast<int>(uncompressed), '%');
   } else {
      for (int i = 0; i < 10; ++i) {
         emitf("\b");
      }
      emitf("Stored :( %9d/%9d bytes ( 0%c)  ",
            static_cast<int>(uncompressed), static_cast<int>(uncompressed),
            '%');
   }
}

} // namespace

static int run_cli(int argc, char** argv) {
   if (argc < 2) {
      print_usage();
      return 1;
   }

   Options options;
   int output_index = -1;
   for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      // The target treats any argument whose first byte is '-' as an option,
      // even a bare "-" (which matches no case and is silently ignored).
      if (!argument.empty() && argument[0] == '-') {
         if (argument.size() < 2) {
            continue;
         }
         const char option = fold_ascii(argument[1]);
         if (option == 'B') {
            parse_double(argument.c_str() + 2, options.threshold);
            options.block_count = 0;
         } else if (option == 'N') {
            if (argument.size() == 3 &&
                fold_ascii(argument[2]) == 'S') {
               options.no_stored_fallback = true;
            } else {
               long value = 0;
               parse_long(argument.c_str() + 2, value);
               // The target's option global is a 32-bit int, so the 64-bit
               // strtol-style result is truncated before the `< 1` test. That
               // makes e.g. -n2147483648 (0x80000000 -> INT_MIN) fall back to
               // automatic threshold splitting.
               const std::int32_t count =
                  std::bit_cast<std::int32_t>(
                     static_cast<std::uint32_t>(value));
               options.block_count =
                  count < 1 ? 0 : static_cast<std::size_t>(count);
            }
         } else if (option == 'Q') {
            options.quiet = true;
         } else if (option == 'R') {
            if (argument.size() >= 3 && fold_ascii(argument[2]) == 'N') {
               options.randomize_tables = true;
            } else {
               options.recurse = true;
            }
         } else if (option == 'S') {
            long value = 0;
            parse_long(argument.c_str() + 2, value);
            if (value < 1) {
               options.strategy = 0;
            } else if (value < 4) {
               options.strategy = static_cast<int>(value);
            } else {
               options.strategy = 4;
            }
         } else if (option == 'V') {
            options.verbose = true;
         } else if (option == 'Y') {
            if (argument.size() == 2) {
               options.assume_yes = true;
            }
         } else if (option == 'Z') {
            // The target matches exactly "-zl121" (case-insensitive z/l,
            // literal digits) and sets the Huffman empty-tree global.
            if (argument.size() == 6 && fold_ascii(argument[2]) == 'L' &&
                argument[3] == '1' && argument[4] == '2' &&
                argument[5] == '1') {
               options.lz121 = true;
            }
         }
      } else if (output_index < 0) {
         output_index = index;
      }
   }

   g_quiet = options.quiet;

   if (output_index < 0) {
      emitf("Need a .ZIP filename!\n");
      return 1;
   }

   // The target's two startup allocations happen after the output-name check
   // and before any discovery. Either failure is the same observable event.
   if (alloc_fail_requested("startup-records") ||
       alloc_fail_requested("startup-buffer")) {
      emitf("Error: malloc failed\n");
      return 1;
   }

   const std::string archive = output_archive_name(argv[output_index]);
   // The target's 64 KiB stream buffer is a process-wide static. Response-file
   // tokenization seeds it before any member is read, and its stale tail then
   // carries across members and chunks.
   std::array<std::uint8_t, kChunkSize> chunk_buffer{};
   SplitterPool splitter_pool;
   ScratchModel scratch;
   std::vector<FileRecord> records;
   try {
      for (int index = output_index + 1; index < argc; ++index) {
         const std::string argument = argv[index];
         if (!argument.empty() && argument[0] == '@') {
            for (const std::string& token :
                 read_response_file(argument.substr(1), chunk_buffer)) {
               discover(token, options.recurse, records);
            }
         } else {
            discover(argument, options.recurse, records);
         }
      }
      // With no filespec arguments the target always expands the default "*".
      if (output_index + 1 == argc) {
         discover("*", options.recurse, records);
      }
   } catch (const std::bad_alloc&) {
      // FUN_004054f0 and the startup allocations report this exact message.
      emitf("Error: malloc failed\n");
      return 1;
   }

   if (records.empty()) {
      emitf("No files found\n");
      return 1;
   }

   // Overwrite prompt: only when -y is absent, output is enabled, and the
   // archive can be opened for reading.
   if (!options.assume_yes && !options.quiet) {
      std::FILE* probe = std::fopen(archive.c_str(), "rb");
      if (probe != nullptr) {
         std::fclose(probe);
         emitf("%s already exists. Overwrite? (y/n)\n", archive.c_str());
         for (;;) {
            int answer = std::getchar();
            int folded = answer & 0xdf;
            if (answer == 0x1b || folded == 'N') {
               emitf("Operation cancelled.\n");
               return 1;
            }
            if (answer == 0 || answer == 0xe0) {
               answer = std::getchar();
               folded = answer & 0xdf;
            }
            if (folded == 'Y') {
               break;
            }
         }
      }
   }

   kzip::ZipWriter writer(archive);
   if (!writer.ok()) {
      emitf("Error: %s could not be opened\n", archive.c_str());
      return 1;
   }
   emitf("Creating ZIP: %s\n", archive.c_str());

   for (std::size_t index = 0; index < records.size(); ++index) {
      const FileRecord record = records[index];
      const std::string member = member_name(record, options.recurse);

      if (record.directory) {
         try {
            discover(member + "/*", options.recurse, records);
         } catch (const std::bad_alloc&) {
            emitf("Error: malloc failed\n");
            return 1;
         }
         continue;
      }

      const std::string source = join_path(record.prefix, record.name);
      emitf("  Adding: %-12s KSflating ", member.c_str());

      // The target mallocs the splitter buffer here when strategy is not 4 and
      // the -b threshold is positive. A failed malloc prints the short message
      // and leaves the already-opened output empty.
      if (options.strategy != 4 && !(options.threshold <= 0.0) &&
          alloc_fail_requested("file-buffer")) {
         emitf("malloc failed\n");
         return 1;
      }

      // The target writes the local header before it runs this file's passes,
      // so the header's placeholder bytes are the scratch state left by the
      // previous file.
      const std::array<std::uint8_t, 12> placeholder = scratch.placeholder();

      bool stored = false;
      std::vector<std::uint8_t> raw_input;
      std::vector<std::uint8_t> split_input;
      std::vector<std::uint8_t> encode_input;
      try {
         if (!read_file(source, record.size, chunk_buffer, raw_input,
                        encode_input)) {
            emitf("Error: %s not found\n", source.c_str());
            return 1;
         }

         // The splitter path mallocs the (sign-extended) 32-bit size. A
         // negative size therefore becomes a huge size_t and fails. With
         // -b0 (threshold <= 0) the target skips that allocation entirely.
         if (record.size < 0 && options.strategy != 4 &&
             !(options.threshold <= 0.0)) {
            emitf("malloc failed\n");
            return 1;
         }

         const bool uses_splitter =
            options.strategy != 4 && record.size > 0 &&
            !(options.threshold <= 0.0);
         if (uses_splitter) {
            // Reuse a freed splitter allocation of the same size, preserving
            // its uninitialized tail, then overlay the bytes actually read.
            split_input = splitter_pool.acquire(
               static_cast<std::size_t>(record.size));
            if (!raw_input.empty()) {
               std::memcpy(split_input.data(), raw_input.data(),
                           raw_input.size());
            }
         }

         std::uint16_t dos_time = 0;
         std::uint16_t dos_date = 0;
         kzip::dos_timestamp(record.mtime, dos_time, dos_date);
         const std::uint32_t declared_size =
            static_cast<std::uint32_t>(record.size);

         if (options.strategy == 4) {
            stored = true;
            replay_stored_only(record.size, options.verbose, scratch);
            writer.add_stored_declared(member, encode_input, declared_size,
                                       dos_time, dos_date, placeholder);
         } else {
            std::vector<std::size_t> ends;
            if (uses_splitter) {
               ends = kzip::choose_block_ends(
                  split_input,
                  kzip::SplitOptions{options.threshold, options.block_count});
               // The target frees the splitter buffer once block ends are
               // chosen, before the encoder re-reads the file.
               splitter_pool.release(std::move(split_input));
            } else {
               // -b0 (and the negative-size streaming case) skips the
               // splitter entirely and emits one block covering the input.
               ends.push_back(encode_input.size());
            }
            if (options.verbose) {
               print_block_boundaries(ends);
            }
            const kzip::EncoderResult result = kzip::encode_strategy0(
               encode_input, ends, make_encoder_options(options));
            replay_passes(result.pass_log, record.size, options.verbose,
                          scratch);
            stored = result.stored;
            if (stored) {
               writer.add_stored_declared(member, encode_input, declared_size,
                                          dos_time, dos_date, placeholder);
            } else {
               writer.add_deflated_declared(member, encode_input,
                                            declared_size, result.deflate,
                                            dos_time, dos_date, placeholder);
            }
         }
      } catch (const std::bad_alloc&) {
         // The per-file input allocation failure uses the short message.
         emitf("malloc failed\n");
         return 1;
      }
      // The target's per-file finish path closes the source and prints one
      // newline, ending the "Adding"/progress/stored line.
      emitf("\n");
   }

   writer.finish();

   // The total denominator sums the discovery-time 32-bit sizes of every
   // record, directories included, with wrapping int addition.
   std::int32_t denominator = 0;
   for (const FileRecord& record : records) {
      denominator = std::bit_cast<std::int32_t>(
         static_cast<std::uint32_t>(denominator) +
         static_cast<std::uint32_t>(record.size));
   }
   const std::uint32_t total = writer.size();
   int ratio = 0;
   if (denominator != 0) {
      // Matches `100 - (int)((long)(uVar26 * 100) / (long)iVar37)`, where
      // uVar26 is the sign-extended 32-bit output size viewed as unsigned.
      const std::uint64_t total_unsigned = static_cast<std::uint64_t>(
         static_cast<std::int64_t>(
            std::bit_cast<std::int32_t>(total)));
      const std::int64_t numerator =
         std::bit_cast<std::int64_t>(total_unsigned * 100u);
      ratio = 100 - static_cast<int>(
                       numerator / static_cast<std::int64_t>(denominator));
   }
   emitf("      Total (including headers): %9d/%9d bytes (%2d%c)",
         static_cast<int>(total), static_cast<int>(denominator), ratio, '%');
   if (options.verbose) {
      emitf("\nTook %lld tics\n", 0LL);
   }
   emitf("\n");
   return 0;
}

int main(int argc, char** argv) {
   try {
      return run_cli(argc, argv);
   } catch (const std::bad_alloc&) {
      // The target reports allocation failure and exits instead of aborting.
      emitf("Error: malloc failed\n");
      return 1;
   }
}
