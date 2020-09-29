#include "log.h"
#include "File.h"

#include <cstdio>
#include <mutex>

vector<File> logFiles;
string globalCpuName;
string logContext;

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
