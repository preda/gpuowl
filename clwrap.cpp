// Copyright (C) 2017-2018 Mihai Preda.

#include "clwrap.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <string>
#include <new>
#include <memory>

using namespace std;

bool check(int err, const char *mes) {
  bool ok = (err == CL_SUCCESS);
  if (!ok) {
    if (mes) {
      log("error %d (%s)\n", err, mes);
    } else {
      log("error %d\n", err);
    }
  }
  return ok;
}

#define CHECK(what) assert(check(what));
#define CHECK2(what, mes) assert(check(what, mes));

vector<cl_device_id> getDeviceIDs(bool onlyGPU) {
  cl_platform_id platforms[16];
  int nPlatforms = 0;
  CHECK(clGetPlatformIDs(16, platforms, (unsigned *) &nPlatforms));
  vector<cl_device_id> ret;
  cl_device_id devices[64];
  for (int i = 0; i < nPlatforms; ++i) {
    unsigned n = 0;
    CHECK(clGetDeviceIDs(platforms[i], onlyGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 64, devices, &n));
    for (unsigned k = 0; k < n; ++k) { ret.push_back(devices[k]); }
  }
  return ret;
}

int getNumberOfDevices() {
  cl_platform_id platforms[8];
  unsigned nPlatforms;
  CHECK(clGetPlatformIDs(8, platforms, &nPlatforms));
  
  unsigned n = 0;
  for (int i = 0; i < (int) nPlatforms; ++i) {
    unsigned delta = 0;
    CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &delta));
    n += delta;
  }
  return n;
}

void getInfo(cl_device_id id, int what, size_t bufSize, void *buf) { CHECK(clGetDeviceInfo(id, what, bufSize, buf, NULL)); }

bool getInfoMaybe(cl_device_id id, int what, size_t bufSize, void *buf) {
  return clGetDeviceInfo(id, what, bufSize, buf, NULL) == CL_SUCCESS;
}

u32 getFreeMemory(cl_device_id id) {
  u64 memSize = 0;
  getInfo(id, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, sizeof(memSize), &memSize);
  return memSize;
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
      return 0;
    }
    device = devices[0];
  }
  return device;
}

vector<cl_device_id> toDeviceIds(const vector<u32> &devices) {
  vector<cl_device_id> ids;
  for (u32 d : devices) { ids.push_back(getDevice(d)); }
  return ids;
}

Context createContext(const vector<u32> &devices) {  
  assert(devices.size() > 0);
  auto ids = toDeviceIds(devices);
  int err;
  Context context(clCreateContext(NULL, ids.size(), ids.data(), NULL, NULL, &err));
  CHECK2(err, "clCreateContext");
  return move(context);
}

Context createContext(cl_device_id id) {  
  int err;
  Context context(clCreateContext(NULL, 1, &id, NULL, NULL, &err));
  CHECK2(err, "clCreateContext");
  return move(context);
}


void release(cl_context context) { CHECK(clReleaseContext(context)); }
void release(cl_program program) { CHECK(clReleaseProgram(program)); }
void release(cl_mem buf)         { CHECK(clReleaseMemObject(buf)); }
void release(cl_queue queue)     { CHECK(clReleaseCommandQueue(queue)); }
void release(cl_kernel k)        { CHECK(clReleaseKernel(k)); }

bool dumpBinary(cl_program program, const string &fileName) {
  if (auto fo = openWrite(fileName)) {
    size_t size;
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, NULL));
    char *buf = new char[size + 1];
    CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(&buf), &buf, NULL));
    fwrite(buf, 1, size, fo.get());
    delete[] buf;
    return true; 
  }
  return false;
}

static string readFile(const string &name) {
  string ret;
  if (auto fi = openRead(name)) {
    char buf[1024];
    while (true) {
      size_t n = fread(buf, 1, sizeof(buf), fi.get());
      ret.append(buf, n);
      if (n < sizeof(buf)) { break; }
    }
  }
  return ret;
}

static cl_program loadBinary(cl_device_id device, cl_context context, const string &binFile) {
  string binary = readFile(binFile);
  cl_program program = 0;
  if (!binary.empty()) {  
    cl_device_id devices[] = {device};
    size_t sizes[] = {binary.size()};
    const unsigned char *binaries[] = {(const unsigned char *) binary.c_str()};
    int binStatus[] = {0};
    int err = 0;
    program = clCreateProgramWithBinary(context, 1, devices, sizes, binaries, binStatus, &err);
    if (err != CL_SUCCESS) {
      log("Error loading pre-compiled kernel from '%s' (error %d, %d)\n", binFile.c_str(), err, binStatus[0]);
    } else {
      log("Loaded pre-compiled kernel from '%s'\n", binFile.c_str());
    }
  }
  return program;
}

static cl_program loadSource(cl_context context, const string &name) {
  string stub = string("#include \"") + name + ".cl\"\n";
  
  const char *ptr = stub.c_str();
  size_t size = stub.size();
  int err;
  cl_program program = clCreateProgramWithSource(context, 1, &ptr, &size, &err);
  CHECK2(err, "clCreateProgramWithSource");
  return program;  
}

static bool build(cl_program program, const vector<cl_device_id> &devices, const string &args) {
  Timer timer;
  int err = clBuildProgram(program, 0, NULL, args.c_str(), NULL, NULL);
  bool ok = (err == CL_SUCCESS);
  if (!ok) { log("OpenCL compilation error %d (args %s)\n", err, args.c_str()); }
  
  size_t logSize;
  for (cl_device_id device : devices) {
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
  if (logSize > 1) {
    std::unique_ptr<char> buf(new char[logSize + 1]);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buf.get(), &logSize);
    buf.get()[logSize] = 0;
    log("%s\n", buf.get());
  }
  }
  if (ok) { log("OpenCL compilation in %d ms, with \"%s\"\n", timer.deltaMillis(), args.c_str()); }
  return ok;
}

cl_program compile(const vector<cl_device_id> &devices, cl_context context, const string &name, const string &extraArgs,
                   const vector<pair<string, unsigned>> &defines) {
  string strDefines;
  string config;
  for (auto d : defines) {
    strDefines = strDefines + "-D" + d.first + "=" + to_string(d.second) + "u ";
    config = config + "_" + to_string(d.second);
  }
  string args = strDefines + extraArgs + " " + "-I. -cl-fast-relaxed-math -cl-std=CL2.0";

  cl_program program = 0;

  string binFile = string("precompiled/") + getHwName(devices.front()) + "_" + name + config + ".so";
  const bool usePrecompiled = false;
  if (usePrecompiled && (program = loadBinary(devices.front(), context, binFile))) {
    if (build(program, devices, args)) {
      return program;
    } else {
      release(program);
    }
  }

  if ((program = loadSource(context, name))) {
    if (build(program, devices, args)) {
      if (usePrecompiled) { dumpBinary(program, binFile); }      
      return program;
    } else {
      release(program);
    }
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

void setArg(cl_kernel k, int pos, void* const& svm) { CHECK(clSetKernelArgSVMPointer(k, pos, svm)); }

void setArg(cl_kernel k, int pos, const Buffer &buf) { setArg(k, pos, buf.get()); }

cl_mem makeBuf(cl_context context, unsigned kind, size_t size, const void *ptr) {
  int err;
  cl_mem buf = clCreateBuffer(context, kind, size, (void *) ptr, &err);
  if (err == CL_OUT_OF_RESOURCES || err == CL_MEM_OBJECT_ALLOCATION_FAILURE) { throw bad_alloc{}; }  
  CHECK2(err, "clCreateBuffer");
  return buf;
}

cl_mem makeBuf(Context &context, unsigned kind, size_t size, const void *ptr) { return makeBuf(context.get(), kind, size, ptr); }

cl_queue makeQueue(cl_device_id d, cl_context c) {
  int err;
  cl_queue q = clCreateCommandQueue(c, d, 0, &err);
  CHECK2(err, "clCreateCommandQueue");
  return q;
}

void flush( cl_queue q) { CHECK(clFlush(q)); }
void finish(cl_queue q) { CHECK(clFinish(q)); }

void run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
  CHECK2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, &groupSize, 0, NULL, NULL), name.c_str());
}

void read(cl_queue queue, bool blocking, cl_mem buf, size_t size, void *data, size_t start) {
  CHECK(clEnqueueReadBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

void read(cl_queue queue, bool blocking, Buffer &buf, size_t size, void *data, size_t start) {
  CHECK(clEnqueueReadBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, cl_mem buf, size_t size, const void *data, size_t start) {
  CHECK(clEnqueueWriteBuffer(queue, buf, blocking, start, size, data, 0, NULL, NULL));
}

void write(cl_queue queue, bool blocking, Buffer &buf, size_t size, const void *data, size_t start) {
  CHECK(clEnqueueWriteBuffer(queue, buf.get(), blocking, start, size, data, 0, NULL, NULL));
}

void copyBuf(cl_queue queue, Buffer &src, Buffer &dst, size_t size) {
  CHECK(clEnqueueCopyBuffer(queue, src.get(), dst.get(), 0, 0, size, 0, NULL, NULL));
}

int getKernelNumArgs(cl_kernel k) {
  int nArgs = 0;
  CHECK(clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(nArgs), &nArgs, NULL));
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
  CHECK(clGetKernelArgInfo(k, pos, CL_KERNEL_ARG_NAME, sizeof(buf), buf, &size));
  assert(size >= 0 && size < sizeof(buf));
  buf[size] = 0;
  return buf;
}

void Queue::zero(Buffer &buf, size_t size) {
  assert(size % sizeof(int) == 0);
  int zero = 0;
  CHECK(clEnqueueFillBuffer(queue.get(), buf.get(), &zero, sizeof(zero), 0, size, 0, 0, 0));
  // finish();
}

u32 getAllocableBlocks(cl_device_id device, u32 blockSizeBytes) {
  assert(blockSizeBytes % 1024 == 0);
  vector<Buffer> buffers;

  auto hostBuf = make_unique<u32[]>(blockSizeBytes);

  Context context = createContext(device);
  
  u32 freeKB = getFreeMemory(device);

  while (true) {
    try {
      buffers.emplace_back(makeBuf(context, BUF_CONST, blockSizeBytes, hostBuf.get()));
      u32 newFreeKB = getFreeMemory(device);
      if (newFreeKB == freeKB) { break; }
      freeKB = newFreeKB;
    } catch (const bad_alloc&) {
      break;
    }
  }
  return buffers.size();
}
