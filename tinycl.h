// Copyright (C) 2017 Mihai Preda.

#include <cstdint>
#include <cstddef>

typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

typedef int cl_int;
typedef unsigned cl_uint;
typedef unsigned uint;
typedef cl_uint cl_bool;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_program_info;
typedef unsigned long cl_ulong;
typedef cl_ulong cl_mem_flags;
typedef cl_ulong cl_device_type;
typedef cl_ulong cl_queue_properties;

extern "C" {
cl_uint clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint,  cl_device_id *, cl_uint *);
cl_context clCreateContext(const intptr_t *, cl_uint, const cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
cl_int clReleaseContext(cl_context);
cl_int clReleaseProgram(cl_program);
cl_int clReleaseCommandQueue(cl_command_queue);
cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
cl_int clGetProgramInfo(cl_program, cl_program_info, size_t, void *, size_t *);
cl_kernel clCreateKernel(cl_program, const char *, cl_int *);
cl_int clReleaseKernel(cl_kernel);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
cl_int clReleaseMemObject(cl_mem);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
cl_int clFlush(cl_command_queue);
cl_int clFinish(cl_command_queue);
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
}

#define CL_SUCCESS                                  0
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_PROGRAM_BINARY_SIZES                     0x1165
#define CL_PROGRAM_BINARIES                         0x1166
#define CL_PROGRAM_BUILD_LOG                        0x1183
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)
#define CL_MEM_HOST_NO_ACCESS                       (1 << 9)
