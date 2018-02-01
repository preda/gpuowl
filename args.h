// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"
#include "common.h"

#include <string>
#include <cstring>

struct Args {
  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  int step;
  int fftSize;
  int device;
  bool timeKernels, debug;
  bool useLongCarry, useLongTail;
  int verbosity;
  
Args() : step(0), fftSize(0),
    device(-1), timeKernels(false), debug(false),
    useLongCarry(false), useLongTail(false),
    verbosity(0) { }
  
  // return false to stop.
  bool parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log("Command line options:\n\n"
          "-size 2M|4M|8M : override FFT size.\n"
          "-user <name>  : specify the user name.\n"
          "-cpu  <name>  : specify the hardware name.\n"
          "-longCarry    : use not-fused carry kernels (may be slower).\n"
          "-longTail     : use not-fused tail kernels  (may be slower).\n"
          "-dump <path>  : dump compiled ISA to the folder <path> that must exist.\n"
          "-time kernels : display kernel profiling information (slower).\n"
          "-verbosity <level> : change amount of information logged. [0-2, default 0].\n"
          "-device <N>   : select specific device among:\n");
      
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        std::string info = getLongInfo(devices[i]);
        log("    %d : %s\n", i, info.c_str());
      }      
      return false;
    } else if (!strcmp(arg, "-verbosity")) {
      if (i < argc - 1) {
        verbosity = atoi(argv[++i]);        
      } else {
        log("-verbosity expects <level>\n");
        return false;
      }
    } else if (!strcmp(arg, "-dump")) {
      if (i < argc - 1 && argv[i + 1][0] != '-') {
        dump = argv[++i];
      } else {
        log("-dump expects name");
        return false;
      }
    } else if (!strcmp(arg, "-debug")) {
      debug = true;
    } else if (!strcmp(arg, "-step")) {
      if (i < argc - 1) {
        step = atoi(argv[++i]);
        if (step <= 0 || step % 1000) {
          log("invalid -step '%s', must be positive and multiple of 1000.\n", argv[i]);
          return false;
        }
      } else {
        log("-step expects <N> argument\n");
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
      if (i < argc - 1 && !strcmp(argv[++i], "kernels")) {
        timeKernels = true;
      } else {
        log("-time expects 'kernels'\n");
        return false;
      }
    } else if (!strcmp(arg, "-longCarry")) {
      useLongCarry = true;
    } else if (!strcmp(arg, "-longTail")) {
      useLongTail = true;
    } else if (!strcmp(arg, "-size")) {
      if (i < argc - 1) {
        const char *value = argv[++i];
        if (int len = strlen(value)) {
          char c = value[len - 1];
          fftSize = atoi(value) * ((c == 'M' || c == 'm') ? 1024 * 1024 : (c == 'K' || c == 'k') ? 1024 : 1);
        }
      } else {
        log("-size expects size 2M | 4M | 8M\n");
        return false;
      }
    } else if (!strcmp(arg, "-device")) {
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

  assert(step % 1000 == 0);
  return true;
}

};
