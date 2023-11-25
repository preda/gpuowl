// Copyright (C) Mihai Preda

#include "File.h"
#include <filesystem>
#include <system_error>

using namespace std;

i64 File::size(const fs::path &name) {
  error_code dummy;
  return filesystem::file_size(name, dummy);
}
