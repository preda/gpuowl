#include "clpp.h"
#include <limits>

atomic<size_t> AllocTrac::totalAlloc = 0;
size_t AllocTrac::maxAlloc = std::numeric_limits<size_t>::max();
