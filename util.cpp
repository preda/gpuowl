#include "util.h"

#include <cstdio>

string formatBound(u32 b) {
  if (b >= 1'000'000 && b % 1'000'000 == 0) {
    return to_string(b / 1'000'000) + 'M';
  } else if (b >= 500'000 && b % 100'000 == 0) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1fM", float(b) / 1'000'000);
    return buf;
  } else {
    return to_string(b);
  }
}
