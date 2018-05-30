// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"
#include "common.h"

#include <string>
#include <cstring>

struct Args {
  enum {CARRY_SHORT = 0, CARRY_LONG = 1, TAIL_FUSED = 0, TAIL_SPLIT = 1};
  
  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  int device;
  bool timeKernels;
  int carry;
  int tail;
  
  Args() :
    device(-1),
    timeKernels(false),
    carry(CARRY_LONG),
    tail(TAIL_FUSED)
  { }
  
  // return false to stop.
  bool parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log(R"""(
Command line options:

-user <name>      : specify the user name.
-cpu  <name>      : specify the hardware name.
-time             : display kernel profiling information.
-tail fused|split : selects tail kernels variant (default 'fused').

-carry long|short : selects carry propagation (default 'long').
    Long carry is safe but may be slower. 'short' may be used only with bits-per-word >= 15.

-device <N>   : select specific device among:
)""");
      
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        std::string info = getLongInfo(devices[i]);
        log("    %d : %s\n", i, info.c_str());
      }      
      return false;
    } else if (!strcmp(arg, "-dump")) {
      if (i < argc - 1 && argv[i + 1][0] != '-') {
        dump = argv[++i];
      } else {
        log("-dump expects name");
        return false;
      }
    } else if (!strcmp(arg, "-user")) {
      if (i < argc - 1) {
        user = argv[++i];
      } else {
        log("-user expects name\n");
        return false;
      }
    } else if (!strcmp(arg, "-cpu")) {
      if (i < argc - 1) {
        cpu = argv[++i];
      } else {
        log("-cpu expects name\n");
        return false;
      }
    } else if (!strcmp(arg, "-cl")) {
      if (i < argc - 1) {
        clArgs = argv[++i];
      } else {
        log("-cl expects options string to pass to CL compiler\n");
        return false;
      }
    } else if(!strcmp(arg, "-time")) {
      timeKernels = true;
    } else if (!strcmp(arg, "-carry")) {
      if (i < argc - 1) {
        std::string s = argv[++i];
        if (s == "short" || s == "long") {
          carry = s == "short" ? CARRY_SHORT : CARRY_LONG;
          continue;
        }
      }
      log("-carry expects short|long\n");
      return false;
    } else if (!strcmp(arg, "-tail")) {
      if (i < argc - 1) {
        std::string s = argv[++i];
        if (s == "fused" || s == "split") {
          tail = s == "fused" ? TAIL_FUSED : TAIL_SPLIT;
          continue;
        }
      }
      log("-tail expects fused|split\n");
      return false;      
    } else if (!strcmp(arg, "-device") || !strcmp(arg, "-d")) {
      if (i < argc - 1) {
        device = atoi(argv[++i]);
        int nDevices = getNumberOfDevices();
        if (device < 0 || device >= nDevices) {
          log("invalid -device %d (must be between [0, %d]\n", device, nDevices - 1);
          return false;
        }        
      } else {
        log("-device expects <N> argument\n");
        return false;
      }
    } else {
      log("Argument '%s' not understood\n", arg);
      return false;
    }
  }

  return true;
}

};
