// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "common.h"
#include "File.h"
#include "timeutil.h"

#include <cstdarg>
#include <cstdio>
#include <vector>
#include <memory>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <filesystem>

vector<File> logFiles;
string globalCpuName;

void initLog() { logFiles.emplace_back(stdout, "stdout"); }

void initLog(const char *logName) {
  if (auto fo = File::openAppend(logName)) {
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
    setlinebuf(fo.get());
#endif
    logFiles.push_back(std::move(fo));
  }
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y-%m-%d %H:%M:%S"); }

void log(const char *fmt, ...) {
  static std::mutex logMutex;
  
  char buf[2 * 1024];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  
  string prefix = shortTimeStr() + (globalCpuName.empty() ? "" : " ") + globalCpuName;

  std::unique_lock lock(logMutex);
  for (auto &f : logFiles) {
    fprintf(f.get(), f.get() == stdout ? "\r%s %s" : "%s %s", prefix.c_str(), buf);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

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
