// Copyright (C) Mihai Preda.

#include "AllocTrac.h"
#include <limits>

std::atomic<size_t> AllocTrac::totalAlloc = 0;
size_t AllocTrac::maxAlloc = size_t(15) * 1024 * 1024 * 1024; // 15 GB
