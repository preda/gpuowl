// Copyright (C) Mihai Preda

#include "CycleFile.h"
#include "fs.h"

CycleFile::CycleFile(const fs::path& name, bool keepOld) :
  name{name},
  f{File::openWrite(name + ".new")},
  keepOld{keepOld}
{}


CycleFile::~CycleFile() {
  if (!f) { return; }
  f.reset();
  fancyRename(name + ".new", name, keepOld);
}

File* CycleFile::operator->() { return f.operator->(); }
File& CycleFile::operator*() { return f.operator*(); }

void CycleFile::reset() { f.reset(); }
