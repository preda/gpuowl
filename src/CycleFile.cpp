// Copyright (C) Mihai Preda

#include "CycleFile.h"
#include <filesystem>

CycleFile::CycleFile(const fs::path& name, bool keepOld) :
  name{name},
  f{File::openWrite(name + ".new")},
  keepOld{keepOld}
{}

CycleFile::~CycleFile() {
  if (!f) { return; }

  string old = name + ".bak";
  string tmp = f->name;

  f.reset();

  std::error_code dummy;
  try {
    if (keepOld) { fs::rename(name, old, dummy); }

    // The normal behavior of rename() is to unlink a pre-existing destination name and to rename successfully
    // See https://en.cppreference.com/w/cpp/filesystem/rename
    // But this behavior is not obeyed by some Windows implementations (mingw/msys?)
    fs::rename(tmp, name);
  } catch (const fs::filesystem_error& e) {
    // So if rename() throws, we attempt to remove the destination explicitly
    fs::remove(old, dummy);
    fs::rename(name, old, dummy);
    fs::rename(tmp, name, dummy);
    if (!keepOld) { fs::remove(old, dummy); }
  }
}

File* CycleFile::operator->() { return f.operator->(); }
File& CycleFile::operator*() { return f.operator*(); }

void CycleFile::reset() { f.reset(); }
