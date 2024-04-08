// Copyright (C) Mihai Preda

#include "CycleFile.h"
#include <filesystem>

CycleFile::CycleFile(const std::filesystem::__cxx11::path& name) :
  name{name},
  f{File::openWrite(name + ".new")}
{}

CycleFile::~CycleFile() {
  if (f) {
    f.reset();
    std::error_code dummy;
    fs::remove(name + ".bak");
    fs::rename(name, name + ".bak", dummy);
    fs::rename(name + ".new", name, dummy);
  }
}

File* CycleFile::operator->() { return f.operator->(); }
File& CycleFile::operator*() { return f.operator*(); }

void CycleFile::reset() { f.reset(); }
