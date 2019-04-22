// Copyright Mihai Preda.

#pragma once

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;

static_assert(sizeof(u32) == 4,   "size u32");
static_assert(sizeof(u64) == 8,   "size u64");

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

void initLog();
void initLog(const char *);

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;
