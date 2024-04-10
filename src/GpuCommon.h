// Copyright (C) Mihai Preda

#pragma once

class Args;
class TrigBufCache;
class Background;

// Data that's normally shared between Gpu instances
class GpuCommon {
public:
  Args* args;
  TrigBufCache* bufCache;
  Background* background;
};
