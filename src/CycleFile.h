// Copyright (C) Mihai Preda

#pragma once

#include "File.h"
#include <filesystem>

/* CycleFile writes the new file to "name.new".
   When done writing, it renames "name.new" to "name".
*/
class CycleFile {
  const fs::path name;
  optional<File> f;

public:
  explicit CycleFile(const fs::path& name);
  ~CycleFile();

  File* operator->();
  File& operator*();

  // Cancel the rename
  void reset();
};
