// Copyright (C) Mihai Preda

#include "log.h"
#include "File.h"
#include "timeutil.h"

#include <cstdio>
#include <mutex>

vector<File> logFiles;
string globalCpuName;

thread_local string context;

void initLog() { logFiles.emplace_back(stdout, "stdout"); }

void initLog(const char *logName) {
  auto fo = File::openAppend(logName);
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
    setlinebuf(fo.get());
#endif
  logFiles.push_back(std::move(fo));
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y%m%d %H:%M:%S"); }

void log(const char *fmt, ...) {
  static std::mutex logMutex;
  
  char buf[2 * 1024];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  
  string prefix = shortTimeStr() + " "s + (globalCpuName.empty() ? "" : globalCpuName + " "s) + context;

  std::unique_lock lock(logMutex);
  for (auto &f : logFiles) {
    fprintf(f.get(), f.get() == stdout ? "\r%s %s" : "%s %s", prefix.c_str(), buf);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

LogContext::LogContext(const string& s) : part{s} {
  assert(!s.empty()); // && (s.find(' ') == string::npos));
  context = context + s;
}

LogContext::~LogContext() {
  auto p = context.rfind(part);
  assert(p != string::npos);
  context = context.substr(0, p);
}
