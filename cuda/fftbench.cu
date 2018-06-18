#include <cufft.h>

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <initializer_list>
#include <algorithm>
#include <vector>

#include "timeutil.h"

#define CHECK(what) assert(what == cudaSuccess)

bool isGood(int n) {
  int a = 0, b = 0, c = 0, d = 0;
  while (n % 2 == 0) { n /= 2; ++a; }
  while (n % 3 == 0) { n /= 3; ++b; }
  while (n % 5 == 0) { n /= 5; ++c; }
  while (n % 7 == 0) { n /= 7; ++d; }
  bool good = n == 1;
  if (good) { printf("[%2d %2d %2d %2d] ", a, b, c, d); }
  return good;
}

int main(int argc, char **argv) {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  
  Timer timer;
  int N = 20 * 1024 * 1024;
  double *data = new double[N]();  
  double *buf1;
  CHECK(cudaMalloc((void **)&buf1, (N + 1) * sizeof(double)));
  CHECK(cudaMemcpy(buf1, data, N * sizeof(double), cudaMemcpyHostToDevice));

  double *buf2;
  CHECK(cudaMalloc((void **)&buf2, (N + 1) * sizeof(double)));
  CHECK(cudaMemcpy(buf2, data, N * sizeof(double), cudaMemcpyHostToDevice));

  std::vector<std::pair<float, int>> v;
  for (int k = 1024 * 1024; k <= N; k += 1024) {
    if (!isGood(k)) { continue; }
    int size = k;
    cufftHandle plan1, plan2;
    CHECK(cufftPlan1d(&plan1, size, CUFFT_D2Z, 1));
    CHECK(cufftPlan1d(&plan2, size, CUFFT_Z2D, 1));
    size_t planSize1, planSize2;
    CHECK(cufftGetSize(plan1, &planSize1));
    CHECK(cufftGetSize(plan2, &planSize2));
    
    timer.deltaMillis();
    int reps = 400;
    
    for (int i = 0; i < reps / 2; ++i) {
      CHECK(cufftExecD2Z(plan1, buf1, (double2 *) buf2));
      CHECK(cufftExecZ2D(plan2, (double2 *) buf2, buf1));
    }
    CHECK(cudaDeviceSynchronize());
    float tt = timer.deltaMillis() / float(reps);
    printf("%5dK %2.2f\n", size / 1024, tt);
    
    while (!v.empty() && v.back().first > tt) { v.pop_back(); }
    v.push_back(std::make_pair(tt, k));
    
    CHECK(cufftDestroy(plan1));
    CHECK(cufftDestroy(plan2));
  }

  printf("\n----\n");
  for (auto x : v) {
    isGood(x.second);
    printf("%.1f %.2f\n", x.second / float(1024), x.first);
  }
  printf("\n----\n");
  for (auto x : v) {
    int size = x.second;
    if (size % (1024 * 1024) == 0) {
      printf("%dM\n", size/(1024 * 1024));
    } else if (size % 1024 == 0) {
      printf("%dK\n", size / 1024);
    } else {
      printf("%d\n", size);
    }
  }
  CHECK(cudaFree(buf1));
  CHECK(cudaFree(buf2));
}
