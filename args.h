// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"
#include "common.h"

#include <string>
#include <cstring>

struct Args {
  enum {DP, SP, M61, M31};
  
  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  int step, saveStep;
  int fftSize;
  int fftKind;
  string fftKindStr;
  int device;
  bool timeKernels, useLegacy, debug;
  
Args() : step(0), saveStep(10000000), fftSize(0), fftKind(DP), fftKindStr("DP"),
    device(-1), timeKernels(false), useLegacy(false), debug(false) { }
  
  // return false to stop.
  bool parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log("Command line options:\n\n"
          "-step     <N> : validate and checkpoint every <N> iterations.\n"
          "-savestep <N> : to persist checkpoint every <N> [default 10M] iterations.\n"
          "-size 2M|4M|8M : override FFT size.\n"
          "-fft DP|SP|M61|M31  : choose FFT variant [default DP]:\n"
          "                DP  : double precision floating point.\n"
          "                SP  : single precision floating point.\n"
          "                M61 : Fast Galois Transform (FGT) modulo M(61).\n"
          "                M31 : FGT modulo M(31).\n"
          "-user <name>  : specify the user name.\n"
          "-cpu  <name>  : specify the hardware name.\n"  
          "-legacy       : use legacy kernels\n"
          "-dump <path>  : dump compiled ISA to the folder <path> that must exist.\n"
          "-cl \"extra OpenCL compiler options\"\n"
          "-device <N>   : select specific device among:\n");
      
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        std::string info = getDeviceInfo(devices[i]);
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
    } else if (!strcmp(arg, "-legacy")) {
      useLegacy = true;
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
    } else if (!strcmp(arg, "-fft")) {
      if (i < argc - 1) {
        std::string s = argv[++i];
        fftKindStr = s;
        if (s == "DP") {
          fftKind = DP;
        } else if (s == "SP") {
          fftKind = SP;
        } else if (s == "M61") {
          fftKind = M61;
        } else if (s == "M31") {
          fftKind = M31;
        } else {
          log("-fft expects DP | SP | M61 | M31\n");
          return false;
        }
      } else {
        log("-kind expects DP | SP | NTT\n");
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
  
  if (saveStep < 500000)  { saveStep = 500000; }
  saveStep  -= saveStep % 500000;

  // logConfig();
  return true;
}

};
