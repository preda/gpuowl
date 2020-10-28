// Copyright Mihai Preda.
#include "common.h"
#include "log.h"
#include "Gpu.h"

int main(int argc, char **argv) {
  initLog();
  Gpu gpu{};
  log("Bye\n");
}
