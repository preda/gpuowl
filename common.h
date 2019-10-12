// Copyright Mihai Preda.

#pragma once

#include <string>

using byte = unsigned char;
using u8 = byte;
using i64 = long long;
using u64 = unsigned long long;
using i32 = int;
using u32 = unsigned;

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
