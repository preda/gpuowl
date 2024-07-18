// Copyright (C) Mihai Preda

#include "log.h"
#include "File.h"
#include "timeutil.h"

// #include <cstdio>
#include <mutex>

// vector<File> logFiles;

thread_local string context;
thread_local vector<string> contextParts;

thread_local File logFile;

File stdoutFile{stdout, "stdout"};

string logContext() { return context; }

void initLog(const char *logName) {
  assert(!logFile);
  logFile = File::openAppend(logName);
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y%m%d %H:%M:%S"); }

static char logBuf[32 * 1024];

void log(const char *fmt, ...) {
  static std::mutex logMutex;

  string prefix = shortTimeStr() + ' ' + context;

  std::unique_lock lock(logMutex);
  int pos = 0;
  snprintf(logBuf, sizeof(logBuf), "%s %n", prefix.c_str(), &pos);

  va_list va;
  va_start(va, fmt);
  vsnprintf(logBuf + pos, sizeof(logBuf) - pos, fmt, va);
  va_end(va);
  string_view s{logBuf};

  if (logFile) { logFile.write(s); }
  stdoutFile.write(s);
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
