// Copyright Mihai Preda.

#pragma once

#include <cstdint>
#include <string>

using u8  = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

static_assert(sizeof(u8)  == 1, "size u8");
static_assert(sizeof(u32) == 4, "size u32");
static_assert(sizeof(u64) == 8, "size u64");

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

void initLog();
void initLog(const char *);

using namespace std;
namespace std::filesystem{};
namespace fs = std::filesystem;

string hex(u64 x);

string rstripNewline(string s);
