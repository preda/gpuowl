// Copyright Mihai Preda.

#pragma once

#include "log.h"

#include <cstdint>
#include <string>
#include <vector>

using u8  = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

static_assert(sizeof(u8)  == 1, "size u8");
static_assert(sizeof(u32) == 4, "size u32");
static_assert(sizeof(u64) == 8, "size u64");


using namespace std;
namespace std::filesystem{};
namespace fs = std::filesystem;

string hex(u64 x);

string rstripNewline(string s);

using Words = vector<u32>;

inline u64 res64(const Words& words) { return (u64(words[1]) << 32) | words[0]; }

inline Words makeWords(u32 E, u32 value) {
  Words ret((E-1)/32 +1);
  ret[0] = value;
  return ret;
}

inline u32 roundUp(u32 x, u32 multiple) { return ((x - 1) / multiple + 1) * multiple; }

u32 crc32(const void* data, size_t size);

inline u32 crc32(const std::vector<u32>& words) { return crc32(words.data(), sizeof(words[0]) * words.size()); }
