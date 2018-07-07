// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

// #include "clwrap.h"
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
  int blockSize;
  int fftSize;
  
  Args() :
    device(-1),
    timeKernels(false),
    carry(CARRY_SHORT),
    tail(TAIL_FUSED),
    blockSize(200),
    fftSize(0)
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
-fft <size>       : specify FFT size, such as: 5000K, 4M, +2, -1.
-device <N>       : select specific device.
)""");
      /*
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        std::string info = getLongInfo(devices[i]);
        log("    %d : %s\n", i, info.c_str());
      }
      */
      return false;
    } else if (!strcmp(arg, "-fft")) {
      if (i < argc - 1) {
        string s = argv[++i];
        fftSize = atoi(s.c_str()) * ((s.back() == 'K') ? 1024 : ((s.back() == 'M') ? 1024 * 1024 : 1));
      } else {
        log("-fft expects <size>\n");
        return false;
      }
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
    } else if (!strcmp(arg, "-block")) {
      if (i < argc - 1) {
        std::string s = argv[++i];
        if (s == "10" || s == "20" || s == "200") {
          blockSize = atoi(s.c_str());
          continue;
        }
      }
      log("-block expects 10 | 20 | 200\n");
      return false;      
    } 

    else if (!strcmp(arg, "-device") || !strcmp(arg, "-d")) {
      if (i < argc - 1) {
        device = atoi(argv[++i]);
        /*
        int nDevices = getNumberOfDevices();
        if (device < 0 || device >= nDevices) {
          log("invalid -device %d (must be between [0, %d]\n", device, nDevices - 1);
          return false;
        }
        */
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
