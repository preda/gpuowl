#include "common.h"
#include <ctime>
#include <cstdarg>

vector<unique_ptr<FILE>> logFiles;

void initLog(const char *logName) {
  logFiles.push_back(std::unique_ptr<FILE>(stdout));
  if (auto fo = open(logName, "a")) {
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
    setlinebuf(fo.get());
#endif
    logFiles.push_back(std::move(fo));
  }
}

void log(const char *fmt, ...) {
  va_list va;
  for (auto &f : logFiles) {
    va_start(va, fmt);
    vfprintf(f.get(), fmt, va);
    va_end(va);
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

string timeStr(const char *format) {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), format, localtime(&t));
  return buf;
}

string timeStr() {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", gmtime(&t));   // equivalent to: "%F %T"
  return buf;
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y-%m-%d %H:%M:%S"); }
