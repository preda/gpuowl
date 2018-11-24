#include "Signal.h"

#include <signal.h>
// #include <cassert>

static volatile int stop = 0;
static void (*oldHandler)(int) = 0;
static void myHandler(int dummy) { stop = 1; }

Signal::Signal() {
  if (!oldHandler) {
    oldHandler = signal(SIGINT, myHandler);
    isOwner = true;
    // assert(oldHandler);
  }
}

Signal::~Signal() { release(); }

bool Signal::stopRequested() { return stop; }

void Signal::release() {
  if (isOwner) {
    isOwner = false;
    // assert(oldHandler);
    signal(SIGINT, oldHandler);
    oldHandler = 0;
  }
}
