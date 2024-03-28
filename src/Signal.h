// Copyright 2018 Mihai Preda

#pragma once

class Signal {
  bool isOwner;
  
public:
  Signal();
  ~Signal();
  
  static unsigned stopRequested();
  void release();
};
