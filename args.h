// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"
#include "common.h"

#include <string>
#include <cstring>

struct Args {
  std::string clArgs, uid;
  int step, saveStep;
  int device;
  bool timeKernels, useLegacy;
  
  Args() : step(100000), saveStep(10000000), device(-1), timeKernels(false), useLegacy(false) { }

  void logConfig() {
    std::string uidStr    = (uid.empty()    ? "" : " -uid "    + uid);
    std::string clStr     = (clArgs.empty() ? "" : " -cl \""   + clArgs + "\"");
    
    std::string tailStr =
      uidStr
      + clStr
      + (timeKernels ? " -time kernels" : "")
      + (useLegacy   ? " -legacy"       : "")
      + (device >= 0 ? " -device " + device : "")
      ;
      
    log("Config: -step %d -savestep %d %s\n", step, saveStep, tailStr.c_str());
  }

  // return false to stop.
  bool parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log("Command line options:\n\n"
          "-step     <N>     : to log, validate and save every <N> [default 100K] iterations.\n"
          "-savestep <N>     : to persist checkpoint every <N> [default 10M] iterations.\n"
          "-uid user/machine : set UID: string to be prepended to the result line\n"
          "-cl \"<OpenCL compiler options>\", e.g. -cl \"-save-temps=tmp/ -O2\"\n"
          "-legacy           : use legacy kernels\n"
          "-device <N>       : select specific device among:\n");
      
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        char info[256];
        getDeviceInfo(devices[i], sizeof(info), info);
        log("    %d : %s\n", i, info);
      }      
      return false;
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
    } else if (!strcmp(arg, "-savestep")) {
      if (i < argc - 1) {
        saveStep = atoi(argv[++i]);
        if (saveStep <= 0) {
          log("invalid -savestep '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-savestep expects <N> argument\n");
        return false;
      }
    } else if (!strcmp(arg, "-uid")) {
      if (i < argc - 1) {
        uid = argv[++i];
      } else {
        log("-uid expects userName/computerName\n");
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
    } else if (!strcmp(arg, "-legacy")) {
      useLegacy = true;
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

  assert(step > 0 && !(step % 1000));
  
  if (saveStep < step)  { saveStep = step; }

  // make them multiple of logStep
  saveStep  -= saveStep % step;

  logConfig();
  return true;
}

};
