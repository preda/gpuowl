// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "common.h"
#include "File.h"
#include "timeutil.h"

#include <cstdarg>
#include <cstdio>
#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <filesystem>

string hex(u64 x) {
  ostringstream out{};
  out << setbase(16) << setfill('0') << setw(16) << x;
  return out.str();
}

std::string rstripNewline(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
  return s;
}

u32 crc32(const void *data, size_t size) {
  u32 tab[16] = {
                 0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
                 0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
                 0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
                 0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C,
  };
  u32 crc = ~0;
  for (auto *p = (const unsigned char *) data, *end = p + size; p < end; ++p) {
    crc = tab[(crc ^  *p      ) & 0xf] ^ (crc >> 4);
    crc = tab[(crc ^ (*p >> 4)) & 0xf] ^ (crc >> 4);
  }
  return ~crc;
}

string formatBound(u32 b) {
  if (b >= 1'000'000 && b % 1'000'000 == 0) {
    return to_string(b / 1'000'000) + 'M';
  } else if (b >= 500'000 && b % 100'000 == 0) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1fM", float(b) / 1'000'000);
    return buf;
  } else {
    return to_string(b);
  }
}
