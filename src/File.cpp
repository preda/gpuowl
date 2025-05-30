// Copyright (C) Mihai Preda

#include "File.h"
#include <filesystem>
#include <system_error>

using namespace std;

File::File(const std::filesystem::path& path, const string& mode, bool throwOnError)
  : readOnly{mode == "rb"}, name{path.string()} {
  assert(readOnly || throwOnError);

  f = fopen(name.c_str(), mode.c_str());
  if (!f && throwOnError) {
    log("Can't open '%s' (mode '%s')\n", name.c_str(), mode.c_str());
    throw(fs::filesystem_error("can't open file"s, path, {}));
  }

  if (mode == "ab") {
    assert(f);
#if HAS_SETLINEBUF
    setlinebuf(f);
#endif
  }
}

File::~File() {
  if (!f) { return; }

  if (!readOnly) { datasync(); }

  fclose(f);
  f = nullptr;
}

i64 File::size(const fs::path &name) {
  error_code dummy;
  return filesystem::file_size(name, dummy);
}

File& File::operator=(File&& other) {
  assert(this != &other);
  this->~File();
  new (this) File(std::move(other));
  return *this;
}
