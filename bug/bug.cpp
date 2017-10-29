#include "clwrap.h"

void log(const char *fmt, ...) {
  va_list va;
  va_start(va, fmt);
  vprintf(fmt, va);
  va_end(va);
}


int main() {
  cl_device_id device = 0;
  getDeviceIDs(true, 1, &device);
  
  cl_context context = createContext(device);
  cl_queue queue = makeQueue(device, context);
  cl_program program = compile(device, context, "bug.cl", "-save-temps=tmp");
  cl_kernel kernel = makeKernel(program, "bug");
  int size = sizeof(int) * 2 * 256 * 8;
  cl_mem buf1 = makeBuf(context, CL_MEM_READ_WRITE, size);
  cl_mem buf2 = makeBuf(context, CL_MEM_READ_WRITE, size);
  cl_mem trig = makeBuf(context, CL_MEM_READ_WRITE, size);
  
  setArg(kernel, 0, buf1);
  setArg(kernel, 1, buf2);
  setArg(kernel, 2, trig);
  
  int *data = new int[256 * 2 * 8]();
  data[0] = 1;
  write(queue, true, buf1, size, data);

  for (int i = 0; i < 256 * 8; ++i) {
    data[2*i]   = 0x49fb5248;
    data[2*i+1] = 0x46515668;
  }
  write(queue, true, trig, size, data);

  run(queue, kernel, 64, "bug", 64);
  read(queue, true, buf2, size, data);
  for (int thread = 0; thread < 33; ++thread) {
    for (int i = 0; i < 8; ++i) {
      printf("%d %d: %8x %8x\n", thread, i, data[(thread + i * 64) * 2], data[(thread + i * 64) * 2 + 1]);
    }
  }  
}
