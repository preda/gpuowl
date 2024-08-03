// Copyright (C) Mihai Preda

#include "CycleFile.h"
#include "fs.h"

CycleFile::CycleFile(const fs::path& name) :
  name{name},
  f{File::openWrite(name + ".new")}
{}


CycleFile::~CycleFile() {
  if (!f) { return; }
  f.reset();
  fancyRename(name + ".new", name);
}

File* CycleFile::operator->() { return f.operator->(); }
File& CycleFile::operator*() { return f.operator*(); }

void CycleFile::reset() { f.reset(); }
