// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

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

typedef unsigned cl_bool;
typedef unsigned cl_program_build_info;
typedef unsigned cl_program_info;
typedef unsigned cl_device_info;
typedef unsigned cl_kernel_info;
typedef unsigned cl_kernel_arg_info;
typedef unsigned cl_kernel_work_group_info;
typedef unsigned cl_profiling_info;
typedef unsigned cl_event_info;
typedef unsigned cl_command_queue_info;

typedef u64 cl_mem_flags;
typedef u64 cl_svm_mem_flags;
typedef u64 cl_device_type;
typedef u64 cl_queue_properties;

extern "C" {

unsigned clGetPlatformIDs(unsigned, cl_platform_id *, unsigned *);
int clGetDeviceIDs(cl_platform_id, cl_device_type, unsigned,  cl_device_id *, unsigned *);
cl_context clCreateContext(const intptr_t *, unsigned, const cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, int *);
int clReleaseContext(cl_context);
int clReleaseProgram(cl_program);
int clReleaseCommandQueue(cl_command_queue);
int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, unsigned, const size_t *, const size_t *, const size_t *, unsigned, const cl_event *, cl_event *);

cl_program clCreateProgramWithSource(cl_context, unsigned, const char **, const size_t *, int *);
cl_program clCreateProgramWithBinary(cl_context, unsigned, const cl_device_id *, const size_t *, const unsigned char **, int *, int *);  

int clBuildProgram(cl_program, unsigned, const cl_device_id *, const char *,
                   void (*)(cl_program, void *), void *);
int clCompileProgram(cl_program, unsigned, const cl_device_id *, const char *,
                     unsigned numHeaders, const cl_program* headers, const char* const* headerNames,
                     void (*)(cl_program, void *), void *);
cl_program clLinkProgram(cl_context context, unsigned nDevice, const cl_device_id* devices, const char* options,
                         unsigned nProgs, const cl_program* progs,
                         void (*)(cl_program, void *), void *, int* err);

int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
int clGetProgramInfo(cl_program, cl_program_info, size_t, void *, size_t *);
int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *, size_t *);
int clGetPlatformInfo(cl_platform_id, cl_device_info, size_t, void *, size_t *);
int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*);

  
cl_kernel clCreateKernel(cl_program, const char *, int *);
int clReleaseKernel(cl_kernel);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, int *);
int clReleaseMemObject(cl_mem);
cl_command_queue clCreateCommandQueueWithProperties(cl_context, cl_device_id, const cl_queue_properties *, int *);
  
int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
                        unsigned numEvents, const cl_event *waitEvents, cl_event *outEvent);
int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
                         unsigned numEvent, const cl_event *waitEvents, cl_event *outEvent);
int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                        unsigned numEvent, const cl_event *waitEvents, cl_event *outEvent);
int clEnqueueFillBuffer(cl_command_queue, cl_mem, const void *, size_t patternSize, size_t offset, size_t size,
                        unsigned numEvent, const cl_event *waitEvents, cl_event *outEvent);
  
int clFlush(cl_command_queue);
int clFinish(cl_command_queue);
int clSetKernelArg(cl_kernel, unsigned, size_t, const void *);

int clReleaseEvent(cl_event);
int clWaitForEvents(unsigned numEvents, const cl_event *);

int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
int clGetKernelArgInfo(cl_kernel, unsigned, cl_kernel_arg_info, size_t, void *, size_t *);
int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);

int clGetEventInfo(cl_event, cl_event_info, size_t, void*, size_t*);
int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void*, size_t*);
  
void* clSVMAlloc(cl_context, cl_svm_mem_flags, size_t, unsigned alignment);
void clSVMFree(cl_context, void*);

int clSetKernelArgSVMPointer(cl_kernel, unsigned, const void *);
  
}

#define CL_SUCCESS              0
#define CL_DEVICE_TYPE_GPU      (1 << 2)
#define CL_DEVICE_TYPE_ALL      0xFFFFFFFF
#define CL_PLATFORM_VERSION     0x0901
#define CL_DEVICE_VENDOR_ID     0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE        0x101F
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT 0x1024
#define CL_DEVICE_NAME          0x102B
#define CL_DRIVER_VERSION       0x102D
#define CL_DEVICE_VERSION       0x102F
#define CL_DEVICE_BUILT_IN_KERNELS 0x103F

#define CL_PROGRAM_BINARY_SIZES 0x1165
#define CL_PROGRAM_BINARIES     0x1166
#define CL_PROGRAM_BUILD_LOG    0x1183

#define CL_MEM_READ_WRITE       (1 << 0)
#define CL_MEM_WRITE_ONLY       (1 << 1)
#define CL_MEM_READ_ONLY        (1 << 2)

#define CL_MEM_USE_HOST_PTR     (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR   (1 << 4)
#define CL_MEM_COPY_HOST_PTR    (1 << 5)

#define CL_MEM_HOST_WRITE_ONLY  (1 << 7)
#define CL_MEM_HOST_READ_ONLY   (1 << 8)
#define CL_MEM_HOST_NO_ACCESS   (1 << 9)

#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1 << 10)
#define CL_MEM_SVM_ATOMICS           (1 << 11)

#define CL_QUEUE_PROFILING_ENABLE    (1 << 1)
#define CL_QUEUE_PROPERTIES       0x1093

/* cl_command_queue_info */
#define CL_QUEUE_CONTEXT                            0x1090
#define CL_QUEUE_DEVICE                             0x1091
#define CL_QUEUE_REFERENCE_COUNT                    0x1092
#define CL_QUEUE_PROPERTIES                         0x1093
#define CL_QUEUE_SIZE                               0x1094

#define CL_PROFILING_COMMAND_QUEUED                 0x1280
#define CL_PROFILING_COMMAND_SUBMIT                 0x1281
#define CL_PROFILING_COMMAND_START                  0x1282
#define CL_PROFILING_COMMAND_END                    0x1283
#define CL_PROFILING_COMMAND_COMPLETE               0x1284

#define CL_EVENT_COMMAND_QUEUE                      0x11D0
#define CL_EVENT_COMMAND_TYPE                       0x11D1
#define CL_EVENT_REFERENCE_COUNT                    0x11D2
#define CL_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define CL_EVENT_CONTEXT                            0x11D4

/* cl_command_type */
#define CL_COMMAND_NDRANGE_KERNEL                   0x11F0
#define CL_COMMAND_TASK                             0x11F1
#define CL_COMMAND_NATIVE_KERNEL                    0x11F2
#define CL_COMMAND_READ_BUFFER                      0x11F3
#define CL_COMMAND_WRITE_BUFFER                     0x11F4
#define CL_COMMAND_COPY_BUFFER                      0x11F5
#define CL_COMMAND_READ_IMAGE                       0x11F6
#define CL_COMMAND_WRITE_IMAGE                      0x11F7
#define CL_COMMAND_COPY_IMAGE                       0x11F8
#define CL_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define CL_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define CL_COMMAND_MAP_BUFFER                       0x11FB
#define CL_COMMAND_MAP_IMAGE                        0x11FC
#define CL_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define CL_COMMAND_MARKER                           0x11FE
#define CL_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define CL_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define CL_COMMAND_READ_BUFFER_RECT                 0x1201
#define CL_COMMAND_WRITE_BUFFER_RECT                0x1202
#define CL_COMMAND_COPY_BUFFER_RECT                 0x1203
#define CL_COMMAND_USER                             0x1204
#define CL_COMMAND_BARRIER                          0x1205
#define CL_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define CL_COMMAND_FILL_BUFFER                      0x1207
#define CL_COMMAND_FILL_IMAGE                       0x1208
#define CL_COMMAND_SVM_FREE                         0x1209
#define CL_COMMAND_SVM_MEMCPY                       0x120A
#define CL_COMMAND_SVM_MEMFILL                      0x120B
#define CL_COMMAND_SVM_MAP                          0x120C
#define CL_COMMAND_SVM_UNMAP                        0x120D

/* command execution status */
#define CL_COMPLETE                                 0x0
#define CL_RUNNING                                  0x1
#define CL_SUBMITTED                                0x2
#define CL_QUEUED                                   0x3

#define CL_KERNEL_NUM_ARGS        0x1191
#define CL_KERNEL_ARG_NAME        0x119A
#define CL_KERNEL_ATTRIBUTES      0x1195

#define CL_KERNEL_COMPILE_WORK_GROUP_SIZE 0x11B1

// AMD
#define CL_DEVICE_PCIE_ID_AMD     0x4034
#define CL_DEVICE_TOPOLOGY_AMD    0x4037
#define CL_DEVICE_BOARD_NAME_AMD  0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD 0x4039

// Error codes
#define CL_MEM_OBJECT_ALLOCATION_FAILURE -4
#define CL_OUT_OF_RESOURCES -5

typedef union
{
    struct { u32 type; u32 data[5]; } raw;
    struct { u32 type; char unused[17]; char bus; char device; char function; } pcie;
} cl_device_topology_amd;

#define CL_COMMAND_NDRANGE_KERNEL                   0x11F0
#define CL_COMMAND_TASK                             0x11F1
#define CL_COMMAND_NATIVE_KERNEL                    0x11F2
#define CL_COMMAND_READ_BUFFER                      0x11F3
#define CL_COMMAND_WRITE_BUFFER                     0x11F4
#define CL_COMMAND_COPY_BUFFER                      0x11F5
#define CL_COMMAND_READ_IMAGE                       0x11F6
#define CL_COMMAND_WRITE_IMAGE                      0x11F7
#define CL_COMMAND_COPY_IMAGE                       0x11F8
#define CL_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define CL_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define CL_COMMAND_MAP_BUFFER                       0x11FB
#define CL_COMMAND_MAP_IMAGE                        0x11FC
#define CL_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define CL_COMMAND_MARKER                           0x11FE
#define CL_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define CL_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define CL_COMMAND_READ_BUFFER_RECT                 0x1201
#define CL_COMMAND_WRITE_BUFFER_RECT                0x1202
#define CL_COMMAND_COPY_BUFFER_RECT                 0x1203
#define CL_COMMAND_USER                             0x1204
#define CL_COMMAND_BARRIER                          0x1205
#define CL_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define CL_COMMAND_FILL_BUFFER                      0x1207
#define CL_COMMAND_FILL_IMAGE                       0x1208
#define CL_COMMAND_SVM_FREE                         0x1209
#define CL_COMMAND_SVM_MEMCPY                       0x120A
#define CL_COMMAND_SVM_MEMFILL                      0x120B
#define CL_COMMAND_SVM_MAP                          0x120C
#define CL_COMMAND_SVM_UNMAP                        0x120D

// ---- error codes

#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define CL_COMPILE_PROGRAM_FAILURE                  -15
#define CL_LINKER_NOT_AVAILABLE                     -16
#define CL_LINK_PROGRAM_FAILURE                     -17
#define CL_DEVICE_PARTITION_FAILED                  -18
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19
#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64
#define CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define CL_INVALID_COMPILER_OPTIONS                 -66
#define CL_INVALID_LINKER_OPTIONS                   -67
#define CL_INVALID_DEVICE_PARTITION_COUNT           -68
#define CL_INVALID_PIPE_SIZE                        -69
#define CL_INVALID_DEVICE_QUEUE                     -70
