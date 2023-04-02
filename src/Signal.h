// Copyright 2018 Mihai Preda

#pragma once

class Signal {
  bool isOwner;
  
public:
  Signal();
  ~Signal();
  
  unsigned stopRequested();
  void release();
};
