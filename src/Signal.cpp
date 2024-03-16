// Copyright (C) Mihai Preda.

#include "Signal.h"

#include <csignal>

using namespace std;

static volatile sig_atomic_t signalled = 0;

static void (*oldHandler)(int) = 0;

static void signalHandler(int signal) { signalled = signal; }

Signal::Signal() {
  if (!oldHandler) {
    oldHandler = signal(SIGINT, signalHandler);
    isOwner = true;
  }
}

Signal::~Signal() { release(); }

unsigned Signal::stopRequested() { return signalled; }

void Signal::release() {
  if (isOwner) {
    isOwner = false;
    signal(SIGINT, oldHandler);
    oldHandler = 0;
  }
}
