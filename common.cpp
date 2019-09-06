// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "common.h"
#include "file.h"
#include "timeutil.h"

#include <cstdarg>
#include <cstdio>
#include <vector>
#include <memory>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <filesystem>

vector<unique_ptr<FILE>> logFiles;
string globalCpuName;

static unique_ptr<FILE> open(const string &name, const char *mode, bool doLog) {
  std::unique_ptr<FILE> f{fopen(name.c_str(), mode)};
  if (!f && doLog) {
    log("Can't open '%s' (mode '%s')\n", name.c_str(), mode);
    throw(fs::filesystem_error("can't open file"s, fs::path(name), {}));
  }
  return f;
}

unique_ptr<FILE> openRead(const string &name, bool doLog) { return open(name, "rb", doLog); }
unique_ptr<FILE> openWrite(const string &name) { return open(name, "wb", true); }
unique_ptr<FILE> openAppend(const string &name) { return open(name, "ab", true); }

void initLog() { logFiles.push_back(unique_ptr<FILE>(stdout)); }

void initLog(const char *logName) {
  if (auto fo = openAppend(logName)) {
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
    fprintf(f.get(), "%s %s", prefix.c_str(), buf);
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
