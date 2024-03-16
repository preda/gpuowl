// Copyright (C) Mihai Preda

#pragma once

#include <string>

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

void initLog();
void initLog(const char *);

struct LogContext {
  explicit LogContext(const std::string& s);
  ~LogContext();

private:
  std::string part;
};
