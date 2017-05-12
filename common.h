#pragma once

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif
