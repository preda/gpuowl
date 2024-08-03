// Copyright (C) Mihai Preda

#include "fs.h"
#include "File.h"

#include <thread>

namespace {

std::string toString(const std::thread::id& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

bool sizeMatches(const fs::path& path, u64 initialSize) {
  error_code dummy;
  return !initialSize || (initialSize == fs::file_size(path, dummy));
}

} // namespace

void fancyRename(const fs::path& src, const fs::path& dst, bool keepOld) {
  // The normal behavior of rename() is to unlink a pre-existing destination name and to rename successfully
  // See https://en.cppreference.com/w/cpp/filesystem/rename
  // But this behavior is not obeyed by some Windows implementations (mingw/msys?)
  // In that case, we attempt to explicitly remove the destination beforehand

  try {
    if (keepOld) { fs::rename(dst, dst + ".bak"); }

    fs::rename(src, dst);

    return;
  } catch (const fs::filesystem_error& e) {
  }

  std::error_code dummy;

  if (keepOld) {
    fs::remove(dst + ".bak", dummy);
    fs::rename(dst, dst + ".bak", dummy);
  } else {
    fs::remove(dst, dummy);
  }

  // Retry the rename following destination removal above.
  // If this rename does not succeed, let it out-throw.
  fs::rename(src, dst);
}

/*
  try {
    // The normal behavior of rename() is to unlink a pre-existing destination name and to rename successfully
    // See https://en.cppreference.com/w/cpp/filesystem/rename
    // But this behavior is not obeyed by some Windows implementations (mingw/msys?)
    if (keepOld) { fs::rename(dst, dst + ".bak"); }
    fs::rename(src, dst);
  } catch (const fs::filesystem_error& e) {
    std::error_code dummy;
    if (keepOld) {
      fs::remove(dst + ".bak", dummy);
      fs::rename(dst, dst + ".bak", dummy);
    } else {
      // So if rename() threw, we attempt to remove the destination explicitly
      fs::remove(dst, dummy);
    }
    fs::rename(src, dst); // If this rename does not succeed, let it throw
  }
*/

u64 fileSize(const fs::path& path) {
  error_code dummy;
  auto size = fs::file_size(path, dummy);
  if (size == decltype(size)(-1)) { size = 0; }
  return size;
}

bool deleteLine(const fs::path& path, const string& targetLine, u64 initialSize) {
  if (!initialSize) { initialSize = fileSize(path); }

  fs::path tmp = path + ("-"s + toString(this_thread::get_id()));
  File fi = File::openRead(path);
  File fo = File::openWrite(tmp);

  bool found = false;
  for (const string& line : fi) {
    if (!found && line == targetLine) {
      found = true;
    } else {
      fo.write(line);
    }
  }

  if (!found || !sizeMatches(path, initialSize)) { return false; }

  fancyRename(tmp, path);
  return true;
}
