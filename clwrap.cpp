// Copyright (C) 2017-2018 Mihai Preda.

#include "clpp.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <string>
#include <new>
#include <memory>

using namespace std;

// starting at 0 to -70
array<string, 71> ERR_MES = {
"SUCCESS", "DEVICE_NOT_FOUND", "DEVICE_NOT_AVAILABLE", "COMPILER_NOT_AVAILABLE",
"MEM_OBJECT_ALLOCATION_FAILURE", "OUT_OF_RESOURCES", "OUT_OF_HOST_MEMORY", "PROFILING_INFO_NOT_AVAILABLE",
"MEM_COPY_OVERLAP", "IMAGE_FORMAT_MISMATCH", "IMAGE_FORMAT_NOT_SUPPORTED", "BUILD_PROGRAM_FAILURE",
"MAP_FAILURE", "MISALIGNED_SUB_BUFFER_OFFSET", "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", "COMPILE_PROGRAM_FAILURE",
"LINKER_NOT_AVAILABLE", "LINK_PROGRAM_FAILURE", "DEVICE_PARTITION_FAILED", "KERNEL_ARG_INFO_NOT_AVAILABLE",
"", "", "", "", "", "", "", "", "", "",
"INVALID_VALUE", "INVALID_DEVICE_TYPE", "INVALID_PLATFORM", "INVALID_DEVICE",
"INVALID_CONTEXT", "INVALID_QUEUE_PROPERTIES", "INVALID_COMMAND_QUEUE", "INVALID_HOST_PTR",
"INVALID_MEM_OBJECT", "INVALID_IMAGE_FORMAT_DESCRIPTOR", "INVALID_IMAGE_SIZE", "INVALID_SAMPLER",
"INVALID_BINARY", "INVALID_BUILD_OPTIONS", "INVALID_PROGRAM", "INVALID_PROGRAM_EXECUTABLE",
"INVALID_KERNEL_NAME", "INVALID_KERNEL_DEFINITION", "INVALID_KERNEL", "INVALID_ARG_INDEX",
"INVALID_ARG_VALUE", "INVALID_ARG_SIZE", "INVALID_KERNEL_ARGS", "INVALID_WORK_DIMENSION",
"INVALID_WORK_GROUP_SIZE", "INVALID_WORK_ITEM_SIZE", "INVALID_GLOBAL_OFFSET", "INVALID_EVENT_WAIT_LIST",
"INVALID_EVENT", "INVALID_OPERATION", "INVALID_GL_OBJECT", "INVALID_BUFFER_SIZE",
"INVALID_MIP_LEVEL", "INVALID_GLOBAL_WORK_SIZE", "INVALID_PROPERTY", "INVALID_IMAGE_DESCRIPTOR",
"INVALID_COMPILER_OPTIONS", "INVALID_LINKER_OPTIONS", "INVALID_DEVICE_PARTITION_COUNT", "INVALID_PIPE_SIZE",
"INVALID_DEVICE_QUEUE"
};

static string errMes(int err) {
  return (err <= 0 && err >= -70) ? ERR_MES[-err] : ""s;
}

class gpu_error : public std::runtime_error {
public:
  const int err;
  
  gpu_error(int err, const string& mes) : runtime_error(errMes(err) + " " + mes), err(err) {}

  gpu_error(int err, const char *file, int line, const char *func, const string& mes)
    : gpu_error(err, mes + " at " + file + ":" + to_string(line) + " " + func) {
  }
};

class gpu_bad_alloc : public std::bad_alloc {
  string w;
  
public:
  gpu_bad_alloc(const string& w) : w(w) {}
  gpu_bad_alloc(size_t size) : gpu_bad_alloc("GPU size "s + to_string(size)) {}

  const char *what() const noexcept override { return w.c_str(); }
};

void check(int err, const char *file, int line, const char *func, const string& mes) {  
  if (err != CL_SUCCESS) {
    // log("CL error %s (%d) %s\n", errMes(err).c_str(), err, mes.c_str());
    throw gpu_error(err, file, line, func, mes);
  }
}

vector<cl_device_id> getDeviceIDs(bool onlyGPU) {
  cl_platform_id platforms[16];
  int nPlatforms = 0;
  CHECK1(clGetPlatformIDs(16, platforms, (unsigned *) &nPlatforms));
  vector<cl_device_id> ret;
  cl_device_id devices[64];
  for (int i = 0; i < nPlatforms; ++i) {
    unsigned n = 0;
    auto kind = onlyGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
    CHECK1(clGetDeviceIDs(platforms[i], kind, 64, devices, &n));
    for (unsigned k = 0; k < n; ++k) { ret.push_back(devices[k]); }
  }
  return ret;
}

int getNumberOfDevices() {
  cl_platform_id platforms[8];
  unsigned nPlatforms;
  CHECK1(clGetPlatformIDs(8, platforms, &nPlatforms));
  
  unsigned n = 0;
  for (int i = 0; i < (int) nPlatforms; ++i) {
    unsigned delta = 0;
    CHECK1(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &delta));
    n += delta;
  }
  return n;
}

void getInfo(cl_device_id id, int what, size_t bufSize, void *buf) { CHECK1(clGetDeviceInfo(id, what, bufSize, buf, NULL)); }

bool getInfoMaybe(cl_device_id id, int what, size_t bufSize, void *buf) {
  return clGetDeviceInfo(id, what, bufSize, buf, NULL) == CL_SUCCESS;
}

u64 getFreeMem(cl_device_id id) {
  u64 memSize = 0;
  getInfo(id, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, sizeof(memSize), &memSize);
  return memSize * 1024; // KB to Bytes.
}

static string getTopology(cl_device_id id) {
  char topology[64];
  cl_device_topology_amd top;
  if (!getInfoMaybe(id, CL_DEVICE_TOPOLOGY_AMD, sizeof(top), &top)) { return ""; }
  snprintf(topology, sizeof(topology), "@%x:%u.%u",
           (unsigned) (unsigned char) top.pcie.bus, (unsigned) top.pcie.device, (unsigned) top.pcie.function);
  return topology;
}

static string getBoardName(cl_device_id id) {
  char boardName[64];
  return getInfoMaybe(id, CL_DEVICE_BOARD_NAME_AMD, sizeof(boardName), boardName) ? boardName : "";
}

string getHwName(cl_device_id id) {
  char name[64];
  getInfo(id, CL_DEVICE_NAME, sizeof(name), name);
  return name;
}

static string getFreq(cl_device_id device) {
  unsigned computeUnits, frequency;
  getInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits);
  getInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(frequency), &frequency);

  char info[64];
  snprintf(info, sizeof(info), "%ux%4u", computeUnits, frequency);
  return info;
}

string getShortInfo(cl_device_id device) { return getHwName(device) + "-" + getFreq(device) + "-" + getTopology(device); }
string getLongInfo(cl_device_id device) { return getShortInfo(device) + " " + getBoardName(device); }

cl_device_id getDevice(int argsDevId) {
  cl_device_id device = nullptr;
  if (argsDevId >= 0) {
    auto devices = getDeviceIDs(false);    
    assert(int(devices.size()) > argsDevId);
    device = devices[argsDevId];
  } else {
    auto devices = getDeviceIDs(true);
    if (devices.empty()) {
      log("No GPU device found. See -h for how to select a specific device.\n");
      throw("No device specified, and no GPU device found");
    }
    device = devices[0];
  }
  return device;
}

Context createContext(cl_device_id id) {  
  int err;
  Context context(clCreateContext(NULL, 1, &id, NULL, NULL, &err));
  CHECK2(err, "clCreateContext");
  return context;
}


void release(cl_context context) { CHECK1(clReleaseContext(context)); }
void release(cl_program program) { CHECK1(clReleaseProgram(program)); }
void release(cl_mem buf)         { CHECK1(clReleaseMemObject(buf)); }
void release(cl_queue queue)     { CHECK1(clReleaseCommandQueue(queue)); }
void release(cl_kernel k)        { CHECK1(clReleaseKernel(k)); }

void dumpBinary(cl_program program, const string &fileName) {
  if (auto fo = openWrite(fileName)) {
    size_t size;
    CHECK1(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, NULL));
    char *buf = new char[size + 1];
    CHECK1(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(&buf), &buf, NULL));
    auto nWrote = fwrite(buf, size, 1, fo.get());
    assert(nWrote == 1);
    delete[] buf;
  } else {
    throw "dump "s + fileName;
  }
}

static cl_program loadSource(cl_context context, const string &source) {
  const char *ptr = source.c_str();
  size_t size = source.size();
  int err;
  cl_program program = clCreateProgramWithSource(context, 1, &ptr, &size, &err);
  CHECK2(err, "clCreateProgramWithSource");
  return program;
}

static void build(cl_program program, cl_device_id device, const string &args) {
  Timer timer;
  int err = clBuildProgram(program, 0, NULL, args.c_str(), NULL, NULL);
  bool ok = (err == CL_SUCCESS);
  if (!ok) { log("OpenCL compilation error %d (args %s)\n", err, args.c_str()); }
  
  size_t logSize;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
  if (logSize > 1) {
    std::unique_ptr<char> buf(new char[logSize + 1]);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buf.get(), &logSize);
    buf.get()[logSize] = 0;
    log("%s\n", buf.get());
  }
  if (ok) {
    log("OpenCL compilation in %d ms, with \"%s\"\n", timer.deltaMillis(), args.c_str());
  } else {
    release(program);
    CHECK2(err, "clBuildProgram");
  }
}

cl_program compile(cl_device_id device, cl_context context, const string &source, const string &extraArgs,
                   const vector<pair<string, unsigned>> &defines) {
  string strDefines;
  string config;
  for (auto d : defines) {
    strDefines = strDefines + "-D" + d.first + "=" + to_string(d.second) + "u ";
    config = config + "_" + to_string(d.second);
  }
  string args = strDefines + extraArgs + " " + "-I. -cl-fast-relaxed-math -cl-std=CL2.0";

  cl_program program = 0;

  if ((program = loadSource(context, source))) {
    build(program, device, args);
    return program;
  }
  
  return 0;
}
  // Other options:
  // * -cl-uniform-work-group-size
  // * -fno-bin-llvmir
  // * various: -fno-bin-source -fno-bin-amdil

cl_kernel makeKernel(cl_program program, const char *name) {
  int err;
  cl_kernel k = clCreateKernel(program, name, &err);
  CHECK2(err, name);
  return k;
}

cl_mem _makeBuf(cl_context context, unsigned kind, size_t size, const void *ptr) {
  // if (getFreeMem(
  
  int err;
  cl_mem buf = clCreateBuffer(context, kind, size, (void *) ptr, &err);
  if (err == CL_OUT_OF_RESOURCES || err == CL_MEM_OBJECT_ALLOCATION_FAILURE) { throw gpu_bad_alloc(size); }
  
  CHECK2(err, "clCreateBuffer");
  return buf;
}

cl_queue makeQueue(cl_device_id d, cl_context c) {
  int err;
  cl_queue q = clCreateCommandQueue(c, d, 0, &err);
  CHECK2(err, "clCreateCommandQueue");
  return q;
}

void flush( cl_queue q) { CHECK1(clFlush(q)); }
void finish(cl_queue q) { CHECK1(clFinish(q)); }

void run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
  CHECK2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, &groupSize, 0, NULL, NULL), name.c_str());
}

void read(cl_queue queue, bool blocking, cl_mem buf, size_t size, void *data, size_t start) {
  CHECK1(clEnqueueReadBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, cl_mem buf, size_t size, const void *data, size_t start) {
  CHECK1(clEnqueueWriteBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

/*
void read(cl_queue queue, bool blocking, Buffer &buf, size_t size, void *data, size_t start) {
  CHECK1(clEnqueueReadBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, Buffer &buf, size_t size, const void *data, size_t start) {
  CHECK1(clEnqueueWriteBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}
*/

void copyBuf(cl_queue queue, const cl_mem src, cl_mem dst, size_t size) {
  CHECK1(clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, NULL));
}

int getKernelNumArgs(cl_kernel k) {
  int nArgs = 0;
  CHECK1(clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(nArgs), &nArgs, NULL));
  return nArgs;
}

int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name) {
  size_t size[3];
  CHECK2(clGetKernelWorkGroupInfo(k, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size), &size, NULL), name);
  return size[0];
}

std::string getKernelArgName(cl_kernel k, int pos) {
  char buf[128];
  size_t size = 0;
  CHECK1(clGetKernelArgInfo(k, pos, CL_KERNEL_ARG_NAME, sizeof(buf), buf, &size));
  assert(size >= 0 && size < sizeof(buf));
  buf[size] = 0;
  return buf;
}

/*
void Queue::zero(Buffer &buf, size_t size) {
  assert(size % sizeof(int) == 0);
  int zero = 0;
  fillBuf(queue.get(), buf, &zero, sizeof(zero), size);
  // CHECK(clEnqueueFillBuffer(queue.get(), buf.get(), &zero, sizeof(zero), 0, size, 0, 0, 0));
  // finish();
}
*/

void fillBuf(cl_queue q, cl_mem buf, void *pat, size_t patSize, size_t size, size_t start) {
  CHECK1(clEnqueueFillBuffer(q, buf, pat, patSize, start, size ? size : patSize, 0, 0, 0));
}

u32 getAllocableBlocks(cl_device_id device, u32 blockSize, u32 minFree) {
  vector<Buffer<std::byte>> buffers;

  Context context = createContext(device);
  
  while (getFreeMem(device) >= minFree) {
    try {
      buffers.emplace_back(context, BUF_RW, blockSize);
    } catch (const bad_alloc&) {
      break;
    }
  }
  return buffers.empty() ? 0 : (buffers.size() - 1);
}
