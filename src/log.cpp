// Copyright (C) Mihai Preda

#include "log.h"
#include "File.h"
#include "timeutil.h"

#include <cstdio>
#include <mutex>

vector<File> logFiles;

thread_local string context;
thread_local vector<string> contextParts;

string logContext() { return context; }

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
  
  char buf[4 * 1024];

  va_list va;
  va_start(va, fmt);
  vsnprintf(buf, sizeof(buf), fmt, va);
  va_end(va);
  
  string prefix = shortTimeStr() + ' ' + context;

  std::unique_lock lock(logMutex);
  for (auto &f : logFiles) {
    fprintf(f.get(), "%s %s", prefix.c_str(), buf);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

LogContext::LogContext(const string& s) : part{s} {
  contextParts.push_back(s);
  context += s;
}

LogContext::~LogContext() {
  assert(!contextParts.empty());
  assert(context.size() >= contextParts.back().size());
  context = context.substr(0, context.size() - contextParts.back().size());
  contextParts.pop_back();
}
