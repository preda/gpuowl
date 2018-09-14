// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "common.h"
#include "file.h"
#include "timeutil.h"

#include <cstdarg>
#include <cstdio>
#include <vector>
#include <memory>

vector<unique_ptr<FILE>> logFiles;
string globalCpuName;

void initLog(const char *logName) {
  logFiles.push_back(std::unique_ptr<FILE>(stdout));
  if (auto fo = open(logName, "a")) {
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
    setlinebuf(fo.get());
#endif
    logFiles.push_back(std::move(fo));
  }
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y-%m-%d %H:%M:%S"); }

void log(const char *fmt, ...) {
  char buf[2 * 1024];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  
  string prefix = shortTimeStr() + (globalCpuName.empty() ? "" : " ") + globalCpuName;

  for (auto &f : logFiles) {
    fprintf(f.get(), "%s %s", prefix.c_str(), buf);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

unique_ptr<FILE> open(const string &name, const char *mode, bool doLog) {
  std::unique_ptr<FILE> f{fopen(name.c_str(), mode)};
  if (!f && doLog) { log("Can't open '%s' (mode '%s')\n", name.c_str(), mode); }
  return f;
}
