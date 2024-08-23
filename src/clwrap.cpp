// Copyright (C) 2017-2024 Mihai Preda.

#include "timeutil.h"
#include "File.h"
#include "clwrap.h"

#include <cmath>
#include <cstdio>
#include <cassert>
#include <string>
#include <new>
#include <memory>
#include <vector>
#include <array>

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

string errMes(int err) {
  string nb = " ("s + to_string(err) + ")";
  string mes = (err <= 0 && err >= -70) ? ERR_MES[-err] : 
    (err == -1001) ? "ICD_NOT_FOUND" : ""s;
  return mes + nb;
}

class gpu_error : public std::runtime_error {
public:
  const int err;
  
  gpu_error(int err, string_view mes) : runtime_error(errMes(err) + " " + string(mes)), err(err) {}

  gpu_error(int err, const char *file, int line, const char *func, string_view mes)
    : gpu_error(err, string(mes) + " at " + file + ":" + to_string(line) + " " + func) {
  }
};

void check(int err, const char *file, int line, const char *func, string_view mes) {  
  if (err != CL_SUCCESS) {
    // log("CL error %s (%d) %s\n", errMes(err).c_str(), err, mes.c_str());
    throw gpu_error(err, file, line, func, mes);
  }
}

static void getInfo_(cl_device_id id, int what, size_t bufSize, void *buf, string_view whatStr) {
  CHECK2(clGetDeviceInfo(id, what, bufSize, buf, NULL), whatStr);
}

#define GET_INFO(id, what, where) getInfo_(id, what, sizeof(where), &where, #what)


string getBdfFromDevice(cl_device_id id) {
  char topology[64] = {0};
  cl_device_topology_amd top;
  try {
    GET_INFO(id, CL_DEVICE_TOPOLOGY_AMD, top);
    snprintf(topology, sizeof(topology), "%02x:%02x.%x",
             (unsigned) (unsigned char) top.pcie.bus, (unsigned) top.pcie.device, (unsigned) top.pcie.function);
  } catch (const gpu_error& err) {
  }
  return topology;
}

vector<cl_device_id> getAllDeviceIDs() {
  cl_platform_id platforms[16];
  int nPlatforms = 0;
  int err = clGetPlatformIDs(16, platforms, (unsigned *) &nPlatforms);
  if (err == -1001) {
    log("No OpenCL platforms found (ICD_NOT_FOUND)\n");
    assert(nPlatforms == 0);
  } else {
    CHECK2(err, "clGetPlatformIDs");
  }

  vector<cl_device_id> ret;
  cl_device_id devices[64];
  for (int i = 0; i < nPlatforms; ++i) {
    // log("platform %d\n", i);
    unsigned n = 0;
    auto kind = false ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
    err = clGetDeviceIDs(platforms[i], kind, 64, devices, &n);
    if (err != CL_SUCCESS) {
      continue;
    }
    for (unsigned k = 0; k < n; ++k) { ret.push_back(devices[k]); }
  }
  return ret;
}

string getDriverVersion(cl_device_id id) {
  try {
    char buf[256];
    GET_INFO(id, CL_DRIVER_VERSION, buf);
    return buf;
  } catch (const gpu_error& e) {
    return "";
  }
}

string getDriverVersionByPos(int pos) {
  assert(pos >= 0);
  return getDriverVersion(getAllDeviceIDs().at(pos));
}

bool hasFreeMemInfo(cl_device_id id) {
  try {
    u64 dummy = 0;
    GET_INFO(id, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, dummy);
    return true;
  } catch (const gpu_error& err) {
    return false;
  }
}

u64 getFreeMem(cl_device_id id) {
  try {
    u64 memSize = 0; 
    GET_INFO(id, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, memSize);
    return memSize * 1024; // KB to Bytes.    
  } catch (const gpu_error& err) {
    return u64(64) * 1024 * 1024 * 1024; // return huge size (64G) when free-info not available
  }
}

float getGpuRamGB(cl_device_id id) {
  try {
    u64 totSize = 0; 
    GET_INFO(id, CL_DEVICE_GLOBAL_MEM_SIZE, totSize);
    return ldexp(totSize, -30); // to GB
  } catch (const gpu_error& err) {
  }
  return 0;
}

string getBoardName(cl_device_id id) {
  char boardName[128] = {0};
  try {
    GET_INFO(id, CL_DEVICE_BOARD_NAME_AMD, boardName);
  } catch (const gpu_error& err) {
  }
  return boardName;
}

string getDeviceName(cl_device_id id) {
  char name[128] = {0};
  GET_INFO(id, CL_DEVICE_NAME, name);
  return name;
}

bool isAmdGpu(cl_device_id id) {
  u32 pcieId = 0;
  GET_INFO(id, CL_DEVICE_VENDOR_ID, pcieId);
  return pcieId == 0x1002;
}

/*
static string getFreq(cl_device_id device) {
  unsigned computeUnits, frequency;
  GET_INFO(device, CL_DEVICE_MAX_COMPUTE_UNITS, computeUnits);
  GET_INFO(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, frequency);

  char info[64];
  snprintf(info, sizeof(info), "%u@%4u", computeUnits, frequency);
  return info;
}
*/

string getShortInfo(cl_device_id device) { return getDeviceName(device); }

cl_device_id getDevice(u32 argsDeviceId) {
  auto devices = getAllDeviceIDs();
  if (devices.empty()) {
    log("No OpenCL device found. Check clinfo\n");
    throw("No OpenCL device found. Check clinfo");
  }
  if (argsDeviceId >= devices.size()) {
    log("Requested device #%u, but only %u devices found\n", argsDeviceId, unsigned(devices.size()));
    throw("Invalid -d device");
  }
  return devices[argsDeviceId];
}

cl_context createContext(cl_device_id id) {  
  int err;
  cl_context context = clCreateContext(NULL, 1, &id, NULL, NULL, &err);
  CHECK2(err, "clCreateContext");
  return context;
}


void release(cl_context context) { CHECK1(clReleaseContext(context)); }
void release(cl_program program) { CHECK1(clReleaseProgram(program)); }
void release(cl_mem buf)         { CHECK1(clReleaseMemObject(buf)); }
void release(cl_queue queue)     { CHECK1(clReleaseCommandQueue(queue)); }
void release(cl_kernel k)        { CHECK1(clReleaseKernel(k)); }
void release(cl_event event)     { CHECK1(clReleaseEvent(event)); }

Program loadSource(cl_context context, const string &source) {
  const char *ptr = source.c_str();
  size_t size = source.size();
  int err = 0;
  cl_program program = clCreateProgramWithSource(context, 1, &ptr, &size, &err);
  CHECK2(err, "clCreateProgramWithSource");
  return Program{program};
}

string getBuildLog(cl_program program, cl_device_id deviceId) {
  size_t logSize;
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
  if (logSize > 1) {
    logSize = std::min(logSize, size_t(5000));
    std::unique_ptr<char[]> buf(new char[logSize + 1]);
    clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, buf.get(), &logSize);
    buf.get()[logSize] = 0;
    return buf.get();
  }
  return {};
}

Program loadBinary(cl_context context, cl_device_id id, string_view fileName) {
  File f = File::openRead(fileName);
  if (!f) { return {}; }
  string bytes = f.readAll();
  size_t size = bytes.size();
  const unsigned char *ptr = reinterpret_cast<const unsigned char *>(bytes.c_str());
  int err = 0;
  cl_program program = clCreateProgramWithBinary(context, 1, &id, &size, &ptr, NULL, &err);
  if (err) {
    log("Load binary %s : %s\n", string(fileName).c_str(), errMes(err).c_str());
    return {};
  }
  if ((err = clBuildProgram(program, 1, &id, NULL, NULL, NULL))) {
    log("Build binary %s : %s\n", string(fileName).c_str(), errMes(err).c_str());
    return {};
  }
  return Program{program};
}

string getBinary(cl_program program) {
  size_t size;
  CHECK1(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size, NULL));
  auto buf = make_unique<char[]>(size + 1);
  char *ptr = buf.get();
  CHECK1(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(&buf), &ptr, NULL));
  return {buf.get(), size};
}

void saveBinary(cl_program program, string_view fileName) {
  File::openWrite(fileName).write(getBinary(program));
}

cl_kernel loadKernel(cl_program program, const char *name) {
  int err;
  cl_kernel k = clCreateKernel(program, name, &err);
  if (err == CL_INVALID_KERNEL_NAME) { return nullptr; }
  CHECK2(err, name);
  return k;
}

cl_mem makeBuf_(cl_context context, unsigned kind, size_t size, const void *ptr) {
  int err;
  cl_mem buf = clCreateBuffer(context, kind, size, (void *) ptr, &err);
  if (err == CL_OUT_OF_RESOURCES || err == CL_MEM_OBJECT_ALLOCATION_FAILURE) { throw bad_alloc{}; }
  
  CHECK2(err, "clCreateBuffer");
  return buf;
}

cl_queue makeQueue(cl_device_id d, cl_context c, bool isProfile) {
  int err;
  cl_queue_properties props[4] = {0};
  props[0] = CL_QUEUE_PROPERTIES;
  props[1] = isProfile ? CL_QUEUE_PROFILING_ENABLE : 0;
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE not supported on ROCm 6.1 or earlier

  cl_queue q = clCreateCommandQueueWithProperties(c, d, props, &err);
  CHECK2(err, "clCreateCommandQueue");
  return q;
}

void flush( cl_queue q) { CHECK1(clFlush(q)); }
void finish(cl_queue q) { CHECK1(clFinish(q)); }

EventHolder run(cl_queue queue, cl_kernel kernel,
                size_t groupSize, size_t workSize,
                vector<cl_event>&& waits,
                const string &name, bool genEvent) {
  cl_event event{};
  CHECK2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, &groupSize,
                                waits.size(), waits.empty() ? 0 : waits.data(), genEvent ? &event : nullptr),
         name.c_str());
  return genEvent ? EventHolder{event} : EventHolder{};
}

EventHolder read(cl_queue queue, vector<cl_event>&& waits,
                 bool blocking, cl_mem buf, size_t size, void *data, bool genEvent) {
  size_t start = 0;
  cl_event event{};
  CHECK1(clEnqueueReadBuffer(queue, buf, blocking, start, size, data,
                             waits.size(), waits.empty() ? 0 : waits.data(), genEvent ? &event : nullptr));
  return genEvent ? EventHolder{event} : EventHolder{};
}

EventHolder write(cl_queue queue, vector<cl_event>&& waits,
                  bool blocking, cl_mem buf, size_t size, const void *data, bool genEvent) {
  size_t start = 0;
  cl_event event{};
  CHECK1(clEnqueueWriteBuffer(queue, buf, blocking, start, size, data,
                              waits.size(), waits.empty() ? 0 : waits.data(), genEvent ? &event : nullptr));
  return genEvent ? EventHolder{event} : EventHolder{};
}

EventHolder copyBuf(cl_queue queue, vector<cl_event>&& waits,
                    const cl_mem src, cl_mem dst, size_t size, bool genEvent) {
  cl_event event{};
  CHECK1(clEnqueueCopyBuffer(queue, src, dst, 0, 0, size,
                             waits.size(), waits.empty() ? 0 : waits.data(), genEvent ? &event : nullptr));
  return genEvent ? EventHolder{event} : EventHolder{};
}

EventHolder fillBuf(cl_queue q, vector<cl_event>&& waits,
                    cl_mem buf, const void *pat, size_t patSize, size_t size, bool genEvent) {
  assert(size);
  cl_event event{};
  CHECK1(clEnqueueFillBuffer(q, buf, pat, patSize, 0 /*start*/, size,
                             waits.size(), waits.empty() ? 0 : waits.data(), genEvent ? &event : nullptr));
  return genEvent ? EventHolder{event} : EventHolder{};
}

void waitForEvents(vector<cl_event>&& waits) {
  if (!waits.empty()) {
    CHECK1(clWaitForEvents(waits.size(), waits.data()));
  }
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

u32 getEventInfo(cl_event event) {
  u32 status = -1;
  CHECK1(clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, 0));
  return status;
}

static i64 delta(u64 a, u64 b) { return b - a; }
  // return b >= a ? i64(b - a) : -i64(a - b); }

array<i64, 3> getEventNanos(cl_event event) {
  u64 prev{};
  array<i64, 3> ret{};

  // return {0,0,0};
  constexpr const u32 what[] = {
    CL_PROFILING_COMMAND_QUEUED,
    CL_PROFILING_COMMAND_SUBMIT,
    CL_PROFILING_COMMAND_START,
    CL_PROFILING_COMMAND_END
  };

  for (int i = 0; i < 4; ++i) {
    u64 t{};
    CHECK1(clGetEventProfilingInfo(event, what[i], sizeof(t), &t, 0));
    if (i) { ret[i - 1] = delta(prev, t); }
    prev = t;
  }
  return ret;
}

cl_context getQueueContext(cl_command_queue q) {
  cl_context ret;
  CHECK1(clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ret, 0));
  return ret;
}

cl_device_id getQueueDevice(cl_command_queue q) {
  cl_device_id id;
  CHECK1(clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(id), &id, 0));
  return id;
}
