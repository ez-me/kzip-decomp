#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace kzip {

// Little-endian ZIP container writer matching the target's local-header,
// central-directory, and EOCD layout.
//
// Local headers are emitted with version-needed 20 and no extra field. The
// CRC-32, compressed size, and uncompressed size fields are patched after the
// member data is written, exactly like the target's patch-back at
// local_header_offset + 14.
//
// Output goes through the target's 1 MiB ring buffer: every byte is appended
// at a wrapping 32-bit logical position, and flushes seek to the flushed
// position and write whole ring windows. The target's flush compares the two
// 32-bit positions as signed ints and drops the window when the write position
// has passed 2 GiB. On close it sign-extends the wrapped 32-bit logical size
// and calls ftruncate, so a length at or above 2 GiB makes ftruncate fail and
// leaves the ring writes' file length untouched. Reproducing these details is
// required for archives at and above 2 GiB.
class ZipWriter {
 public:
   explicit ZipWriter(const std::string& path);
   ZipWriter(const ZipWriter&) = delete;
   ZipWriter& operator=(const ZipWriter&) = delete;
   ~ZipWriter();

   [[nodiscard]] bool ok() const noexcept;

   // Total logical bytes written so far (the archive length after `finish()`).
   [[nodiscard]] std::uint32_t size() const noexcept;

   // Add a member whose payload is the raw input. The target uses method 0 for
   // strategy 4 and for the size-based stored fallback.
   bool add_stored(const std::string& name,
                   std::span<const std::uint8_t> input,
                   std::uint16_t dos_time,
                   std::uint16_t dos_date);

   // Variant used when the target's 32-bit discovery-time size differs from the
   // number of payload bytes actually produced (for example a >= 2 GiB file
   // whose st_size truncates to a negative int, or a file that shrank between
   // discovery and read). `uncompressed_size` is written verbatim into the
   // local and central headers; the CRC still covers `input`.
   bool add_stored_declared(const std::string& name,
                            std::span<const std::uint8_t> input,
                            std::uint32_t uncompressed_size,
                            std::uint16_t dos_time,
                            std::uint16_t dos_date,
                            std::array<std::uint8_t, 12> placeholder = {});

   // Add a member whose payload is a raw DEFLATE stream. `input` supplies the
   // uncompressed bytes for the CRC-32 and size fields.
   bool add_deflated(const std::string& name,
                     std::span<const std::uint8_t> input,
                     std::span<const std::uint8_t> payload,
                     std::uint16_t dos_time,
                     std::uint16_t dos_date);

   bool add_deflated_declared(const std::string& name,
                              std::span<const std::uint8_t> input,
                              std::uint32_t uncompressed_size,
                              std::span<const std::uint8_t> payload,
                              std::uint16_t dos_time,
                              std::uint16_t dos_date,
                              std::array<std::uint8_t, 12> placeholder = {});

   // Emit the central directory and EOCD, then flush, truncate to the wrapped
   // logical size, and close the output.
   bool finish();

 private:
   struct Entry {
      std::string name;
      std::uint32_t crc{};
      std::uint32_t compressed_size{};
      std::uint32_t uncompressed_size{};
      std::uint16_t dos_time{};
      std::uint16_t dos_date{};
      std::uint16_t method{};
      std::uint8_t attributes{0x20};
      std::uint32_t local_offset{};
   };

   // The target's output layer (0x402f90, 0x403060, 0x403090).
   class Output {
    public:
      explicit Output(const std::string& path);
      Output(const Output&) = delete;
      Output& operator=(const Output&) = delete;
      ~Output();

      [[nodiscard]] bool ok() const noexcept { return fd_ >= 0; }
      [[nodiscard]] std::uint32_t position() const noexcept { return pos_; }

      void append(const std::uint8_t* bytes, std::size_t count);
      // The target sets both the write and flush positions when it seeks.
      void seek(std::uint32_t position) noexcept {
         pos_ = position;
         flushed_ = position;
      }
      void flush();
      void finish();

    private:
      static constexpr std::uint32_t kRingSize = 0x100000;
      static constexpr std::uint32_t kRingMask = kRingSize - 1;

      int fd_ = -1;
      std::vector<std::uint8_t> ring_ =
         std::vector<std::uint8_t>(kRingSize, 0);
      std::uint32_t pos_ = 0;
      std::uint32_t flushed_ = 0;
   };

   bool write_bytes(std::span<const std::uint8_t> bytes);
   bool write_u16(std::uint16_t value);
   bool write_u32(std::uint32_t value);
   bool patch_header(std::uint32_t local_offset, std::uint32_t crc,
                     std::uint32_t compressed_size,
                     std::uint32_t uncompressed_size);
   bool write_local_header(const Entry& entry,
                           const std::array<std::uint8_t, 12>& placeholder);
   bool write_central_header(const Entry& entry);

   Output output_;
   std::vector<Entry> entries_;
};

// Standard reflected CRC-32 used by the target (polynomial 0xEDB88320,
// initial value 0xFFFFFFFF, complemented result).
std::uint32_t crc32(std::span<const std::uint8_t> bytes);

// Pack a `time_t` into the target's DOS time and date fields using local time
// and truncating seconds to an even value.
void dos_timestamp(std::int64_t unix_time, std::uint16_t& dos_time,
                   std::uint16_t& dos_date);

} // namespace kzip
