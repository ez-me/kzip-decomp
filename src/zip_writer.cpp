#include "zip_writer.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>

namespace kzip {
namespace {

constexpr std::uint32_t kLocalSignature = 0x04034b50u;
constexpr std::uint32_t kCentralSignature = 0x02014b50u;
constexpr std::uint32_t kEndSignature = 0x06054b50u;
constexpr std::uint16_t kVersion = 20;

const std::array<std::uint32_t, 256>& crc_table() {
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
   return table;
}

} // namespace

std::uint32_t crc32(std::span<const std::uint8_t> bytes) {
   const auto& table = crc_table();
   std::uint32_t crc = 0xffffffffu;
   for (std::uint8_t byte : bytes) {
      crc = (crc >> 8) ^ table[(crc ^ byte) & 0xffu];
   }
   return crc ^ 0xffffffffu;
}

void dos_timestamp(std::int64_t unix_time, std::uint16_t& dos_time,
                   std::uint16_t& dos_date) {
   const std::time_t seconds = static_cast<std::time_t>(unix_time);
   std::tm local{};
#if defined(_WIN32)
   localtime_s(&local, &seconds);
#else
   localtime_r(&seconds, &local);
#endif
   const int year = local.tm_year - 80;
   const int month = local.tm_mon + 1;
   const int day = local.tm_mday;
   dos_time = static_cast<std::uint16_t>(
      (local.tm_hour << 11) | (local.tm_min << 5) | (local.tm_sec >> 1));
   dos_date = static_cast<std::uint16_t>(
      ((year & 0x7f) << 9) | ((month & 0x0f) << 5) | (day & 0x1f));
}

// ---------------------------------------------------------------------------
// Target output layer.
//
// 0x403090 appends one byte at the wrapping 32-bit position and flushes once a
// full 1 MiB window is buffered. 0x402f90 flushes by seeking to the flushed
// position and writing one or two ring windows; the seek offset is the
// sign-extended 32-bit flushed position, and the initial `(int)pos <=
// (int)flushed` test is a signed comparison, so once the logical position
// passes 2 GiB the target silently drops windows. 0x403060 finishes with
// ftruncate(fd, (unsigned 32-bit logical size)) and close.
// ---------------------------------------------------------------------------

ZipWriter::Output::Output(const std::string& path) {
   fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
}

ZipWriter::Output::~Output() {
   if (fd_ >= 0) {
      ::close(fd_);
   }
}

void ZipWriter::Output::flush() {
   if (fd_ < 0) {
      return;
   }
   const std::uint32_t write_position = pos_;
   const std::uint32_t flush_position = flushed_;
   if (static_cast<std::int32_t>(write_position) <=
       static_cast<std::int32_t>(flush_position)) {
      flushed_ = write_position;
      return;
   }
   // The target ignores the lseek result. When the flushed position is
   // negative as a 32-bit int, this seek fails and the following writes append
   // at the current file offset instead.
   ::lseek(fd_, static_cast<off_t>(static_cast<std::int32_t>(flush_position)),
           SEEK_SET);
   if ((((write_position - 1u) ^ flush_position) & 0xfff00000u) == 0) {
      const auto count = static_cast<std::int32_t>(write_position -
                                                   flush_position);
      ::write(fd_, ring_.data() + (flush_position & kRingMask),
              static_cast<std::size_t>(count));
      flushed_ = write_position;
      return;
   }
   const auto first = static_cast<std::int32_t>(
      kRingSize - (flush_position & kRingMask));
   ::write(fd_, ring_.data() + (flush_position & kRingMask),
           static_cast<std::size_t>(first));
   ::write(fd_, ring_.data(),
           static_cast<std::size_t>(write_position & kRingMask));
   flushed_ = write_position;
}

void ZipWriter::Output::append(const std::uint8_t* bytes,
                               std::size_t count) {
   while (count != 0) {
      const std::uint32_t buffered = pos_ - flushed_;
      const std::uint32_t room = kRingSize - buffered;
      const std::size_t take = std::min<std::size_t>(room, count);
      const std::size_t index = pos_ & kRingMask;
      const std::size_t first = std::min<std::size_t>(take, kRingSize - index);
      std::copy_n(bytes, first, ring_.data() + index);
      if (take > first) {
         std::copy_n(bytes + first, take - first, ring_.data());
      }
      pos_ += static_cast<std::uint32_t>(take);
      bytes += take;
      count -= take;
      if (pos_ - flushed_ >= kRingSize) {
         flush();
      }
   }
}

void ZipWriter::Output::finish() {
   if (fd_ < 0) {
      return;
   }
   flush();
   // The target sign-extends the 32-bit logical size before ftruncate. A size
   // whose high bit is set becomes negative, ftruncate fails with EINVAL, and
   // the file keeps whatever length the ring writes produced.
   ::ftruncate(fd_, static_cast<off_t>(static_cast<std::int32_t>(pos_)));
   ::close(fd_);
   fd_ = -1;
}

// ---------------------------------------------------------------------------

ZipWriter::ZipWriter(const std::string& path) : output_(path) {}

ZipWriter::~ZipWriter() = default;

bool ZipWriter::ok() const noexcept { return output_.ok(); }

std::uint32_t ZipWriter::size() const noexcept { return output_.position(); }

bool ZipWriter::write_bytes(std::span<const std::uint8_t> bytes) {
   if (!output_.ok()) {
      return false;
   }
   // The target discards every write result, so output I/O failures do not
   // change the exit status or the reported sizes.
   output_.append(bytes.data(), bytes.size());
   return true;
}

bool ZipWriter::write_u16(std::uint16_t value) {
   const std::array<std::uint8_t, 2> bytes{
      static_cast<std::uint8_t>(value & 0xffu),
      static_cast<std::uint8_t>((value >> 8) & 0xffu)};
   return write_bytes(bytes);
}

bool ZipWriter::write_u32(std::uint32_t value) {
   const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value & 0xffu),
      static_cast<std::uint8_t>((value >> 8) & 0xffu),
      static_cast<std::uint8_t>((value >> 16) & 0xffu),
      static_cast<std::uint8_t>((value >> 24) & 0xffu)};
   return write_bytes(bytes);
}

bool ZipWriter::patch_header(std::uint32_t local_offset, std::uint32_t crc,
                             std::uint32_t compressed_size,
                             std::uint32_t uncompressed_size) {
   if (!output_.ok()) {
      return false;
   }
   // 0x405980/0x403dc0 save the logical end, flush, move both positions to
   // local_offset + 14, append the 12-byte patch, flush it, then restore the
   // end position. The seek and flush here are the target's, including the
   // signed drop behaviour at and above 2 GiB.
   const std::uint32_t saved = output_.position();
   output_.flush();
   output_.seek(local_offset + 14u);
   const std::array<std::uint8_t, 12> patch{
      static_cast<std::uint8_t>(crc & 0xffu),
      static_cast<std::uint8_t>((crc >> 8) & 0xffu),
      static_cast<std::uint8_t>((crc >> 16) & 0xffu),
      static_cast<std::uint8_t>((crc >> 24) & 0xffu),
      static_cast<std::uint8_t>(compressed_size & 0xffu),
      static_cast<std::uint8_t>((compressed_size >> 8) & 0xffu),
      static_cast<std::uint8_t>((compressed_size >> 16) & 0xffu),
      static_cast<std::uint8_t>((compressed_size >> 24) & 0xffu),
      static_cast<std::uint8_t>(uncompressed_size & 0xffu),
      static_cast<std::uint8_t>((uncompressed_size >> 8) & 0xffu),
      static_cast<std::uint8_t>((uncompressed_size >> 16) & 0xffu),
      static_cast<std::uint8_t>((uncompressed_size >> 24) & 0xffu)};
   output_.append(patch.data(), patch.size());
   output_.flush();
   output_.seek(saved);
   return true;
}

bool ZipWriter::write_local_header(
   const Entry& entry, const std::array<std::uint8_t, 12>& placeholder) {
   const std::uint16_t flags = entry.method == 8 ? 0x0002u : 0u;
   const std::uint16_t name_length =
      static_cast<std::uint16_t>(entry.name.size());
   // The target emits the local header before the payload and before it knows
   // the final CRC and sizes. Those 12 bytes come from its scratch buffer and
   // are patched afterwards; when the patch is dropped above 2 GiB the
   // placeholder is what remains in the file.
   if (!write_u32(kLocalSignature) || !write_u16(kVersion) ||
       !write_u16(flags) || !write_u16(entry.method) ||
       !write_u16(entry.dos_time) || !write_u16(entry.dos_date) ||
       !write_bytes(placeholder) || !write_u16(name_length) ||
       !write_u16(0)) {
      return false;
   }
   return write_bytes(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(entry.name.data()),
      entry.name.size()));
}

bool ZipWriter::write_central_header(const Entry& entry) {
   const std::uint16_t flags = entry.method == 8 ? 0x0002u : 0u;
   const std::uint16_t name_length =
      static_cast<std::uint16_t>(entry.name.size());
   if (!write_u32(kCentralSignature) || !write_u16(kVersion) ||
       !write_u16(kVersion) || !write_u16(flags) || !write_u16(entry.method) ||
       !write_u16(entry.dos_time) || !write_u16(entry.dos_date) ||
       !write_u32(entry.crc) || !write_u32(entry.compressed_size) ||
       !write_u32(entry.uncompressed_size) || !write_u16(name_length) ||
       !write_u16(0) || !write_u16(0) || !write_u16(0) || !write_u16(0) ||
       !write_u32(entry.attributes) || !write_u32(entry.local_offset)) {
      return false;
   }
   return write_bytes(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(entry.name.data()),
      entry.name.size()));
}

bool ZipWriter::add_stored(const std::string& name,
                           std::span<const std::uint8_t> input,
                           std::uint16_t dos_time, std::uint16_t dos_date) {
   return add_stored_declared(name, input,
                              static_cast<std::uint32_t>(input.size()),
                              dos_time, dos_date);
}

bool ZipWriter::add_stored_declared(const std::string& name,
                                    std::span<const std::uint8_t> input,
                                    std::uint32_t uncompressed_size,
                                    std::uint16_t dos_time,
                                    std::uint16_t dos_date,
                                    std::array<std::uint8_t, 12> placeholder) {
   Entry entry;
   entry.name = name;
   entry.crc = crc32(input);
   entry.compressed_size = static_cast<std::uint32_t>(input.size());
   entry.uncompressed_size = uncompressed_size;
   entry.dos_time = dos_time;
   entry.dos_date = dos_date;
   entry.method = 0;
   entry.attributes = 0x20;
   entry.local_offset = output_.position();

   if (!write_local_header(entry, placeholder)) {
      return false;
   }
   if (!write_bytes(input)) {
      return false;
   }
   entries_.push_back(entry);
   // The target's stored-entry writer patches the local header after the
   // payload even though the values are already known.
   return patch_header(entry.local_offset, entry.crc, entry.compressed_size,
                       entry.uncompressed_size);
}

bool ZipWriter::add_deflated(const std::string& name,
                             std::span<const std::uint8_t> input,
                             std::span<const std::uint8_t> payload,
                             std::uint16_t dos_time,
                             std::uint16_t dos_date) {
   return add_deflated_declared(name, input,
                                static_cast<std::uint32_t>(input.size()),
                                payload, dos_time, dos_date);
}

bool ZipWriter::add_deflated_declared(const std::string& name,
                                      std::span<const std::uint8_t> input,
                                      std::uint32_t uncompressed_size,
                                      std::span<const std::uint8_t> payload,
                                      std::uint16_t dos_time,
                                      std::uint16_t dos_date,
                                      std::array<std::uint8_t, 12>
                                         placeholder) {
   Entry entry;
   entry.name = name;
   entry.crc = crc32(input);
   entry.compressed_size = static_cast<std::uint32_t>(payload.size());
   entry.uncompressed_size = uncompressed_size;
   entry.dos_time = dos_time;
   entry.dos_date = dos_date;
   entry.method = 8;
   entry.attributes = 0x20;
   entry.local_offset = output_.position();

   if (!write_local_header(entry, placeholder)) {
      return false;
   }
   if (!write_bytes(payload)) {
      return false;
   }
   entries_.push_back(entry);
   return patch_header(entry.local_offset, entry.crc, entry.compressed_size,
                       entry.uncompressed_size);
}

bool ZipWriter::finish() {
   if (!output_.ok()) {
      return false;
   }
   const std::uint32_t central_offset = output_.position();
   for (const Entry& entry : entries_) {
      if (!write_central_header(entry)) {
         return false;
      }
   }
   const std::uint32_t central_size = output_.position() - central_offset;
   const std::uint16_t count = static_cast<std::uint16_t>(entries_.size());

   if (!write_u32(kEndSignature) || !write_u16(0) || !write_u16(0) ||
       !write_u16(count) || !write_u16(count) || !write_u32(central_size) ||
       !write_u32(central_offset) || !write_u16(0)) {
      return false;
   }
   // The target ignores flush, truncate, and close failures as well.
   output_.finish();
   return true;
}

} // namespace kzip
