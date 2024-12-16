// Copyright Mihai Preda.

#pragma once

#include "tinycl.h"

#include <string>
#include <string_view>
#include <vector>
#include <memory>

using cl_queue = cl_command_queue;


void release(cl_context context);
void release(cl_kernel k);
void release(cl_mem buf);
void release(cl_program program);
void release(cl_queue queue);
void release(cl_event event);

template<typename T>
struct Deleter {
  using pointer = T;
  void operator()(T t) const { release(t); }
};

namespace std {
template<> struct default_delete<cl_context> : public Deleter<cl_context> {};
template<> struct default_delete<cl_kernel> : public Deleter<cl_kernel> {};
template<> struct default_delete<cl_mem> : public Deleter<cl_mem> {};
template<> struct default_delete<cl_program> : public Deleter<cl_program> {};
template<> struct default_delete<cl_queue> : public Deleter<cl_queue> {};
template<> struct default_delete<cl_event> : public Deleter<cl_event> {};
}

template<typename T> using Holder = std::unique_ptr<T, Deleter<T> >;

using QueueHolder = std::unique_ptr<cl_queue>;
using KernelHolder = std::unique_ptr<cl_kernel>;
using EventHolder = std::unique_ptr<cl_event>;
using Program = std::unique_ptr<cl_program>;

class Context;

std::string getUUID(int seqId);

std::string errMes(int err);
void check(int err, const char *file, int line, const char *func, string_view mes);

#define CHECK1(err) check(err, __FILE__, __LINE__, __func__, #err)
#define CHECK2(err, mes) check(err, __FILE__, __LINE__, __func__, mes)

vector<cl_device_id> getAllDeviceIDs();
string getShortInfo(cl_device_id device);

string getDeviceName(cl_device_id id);
string getBoardName(cl_device_id id);
float getGpuRamGB(cl_device_id id);

// Get GPU free memory in bytes.
u64 getFreeMem(cl_device_id id);
bool hasFreeMemInfo(cl_device_id id);
bool isAmdGpu(cl_device_id id);
string getDriverVersion(cl_device_id id);
string getDriverVersionByPos(int pos);

string getBdfFromDevice(cl_device_id id);

cl_context createContext(cl_device_id id);

string getBuildLog(cl_program program, cl_device_id deviceId);

Program loadBinary(cl_context context, cl_device_id deviceId, string_view fileName);
Program loadSource(cl_context context, const string& source);
cl_kernel loadKernel(cl_program program, const char *name);
void saveBinary(cl_program program, string_view fileName);

template<typename T>
void setArg(cl_kernel k, int pos, const T &value, const string& name) {
  CHECK2(clSetKernelArg(k, pos, sizeof(value), &value), (name + '[' + to_string(pos) + "] size " + to_string(sizeof(value))).c_str());
}

/*
template<>
void setArg<int>(cl_kernel k, int pos, const int &value, const string& name);
*/

cl_mem makeBuf_(cl_context context, unsigned kind, size_t size, const void *ptr = 0);
cl_queue makeQueue(cl_device_id d, cl_context c, bool enableProfile);

void flush( cl_queue q);
void finish(cl_queue q);

EventHolder run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize,
                vector<cl_event>&& waits, const string &name, bool genEvent);

EventHolder read(cl_queue queue, vector<cl_event>&& waits,
                 bool blocking, cl_mem buf, size_t size, void *data, bool genEvent);

EventHolder write(cl_queue queue, vector<cl_event>&& waits,
                  bool blocking, cl_mem buf, size_t size, const void *data, bool genEvent);

EventHolder copyBuf(cl_queue queue, vector<cl_event>&& waits, const cl_mem src, cl_mem dst, size_t size, bool genEvent);

EventHolder fillBuf(cl_queue q, vector<cl_event>&& waits, cl_mem buf, const void *pat, size_t patSize, size_t size, bool genEvent);

void waitForEvents(vector<cl_event>&& waits);


int getKernelNumArgs(cl_kernel k);
int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name);
std::string getKernelArgName(cl_kernel k, int pos);

cl_device_id getDevice(u32 argsDevId);

// Returns the 3 intervals: queued, submit, run
std::array<i64, 3> getEventNanos(cl_event event);

u32 getEventInfo(cl_event event);

cl_context getQueueContext(cl_command_queue q);
