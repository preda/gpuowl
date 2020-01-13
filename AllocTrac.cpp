// Copyright (C) Mihai Preda.

#include "AllocTrac.h"
#include <limits>

std::atomic<size_t> AllocTrac::totalAlloc = 0;
size_t AllocTrac::maxAlloc = size_t(31) * 512 * 1024 * 1024;
// std::numeric_limits<size_t>::max();
