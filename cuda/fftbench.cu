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

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  
  Timer timer;
  int N = 16 * 1024 * 1024;
  double2 *data = new double2[N]();
  /*
  for (int i = 0; i < N; ++i) {
    data[i].x = i;
    data[i].y = i / 2;
  }
  printf("init data %d\n", timer.deltaMillis());
  */
  
  double2 *buf1;
  // double2 *buf2;
  CHECK(cudaMalloc((void **)&buf1, N * sizeof(double2)));
  CHECK(cudaMemcpy(buf1, data, N * sizeof(double2), cudaMemcpyHostToDevice));
  
  // CHECK(cudaMalloc((void **)&buf2, N * sizeof(double2)));
  // CHECK(cudaMemcpy(buf2, data, N * sizeof(double2), cudaMemcpyHostToDevice));
  // printf("copy %d\n", timer.deltaMillis());

  std::vector<std::pair<float, int>> v;
  for (int k = 2048*1024; k <= 11 * 1024*1024; k += 1024) {
    if (!isGood(k)) { continue; }
    int size = k;
    cufftHandle plan;
    CHECK(cufftPlan1d(&plan, size, CUFFT_Z2Z, 1));
    size_t planSize;
    CHECK(cufftGetSize(plan, &planSize));
    timer.deltaMillis();
    int reps = 400;
    for (int i = 0; i < reps; ++i) { CHECK(cufftExecZ2Z(plan, buf1, buf1, CUFFT_FORWARD)); }
    CHECK(cudaDeviceSynchronize());
    int t = timer.deltaMillis();
    float tt = t / float(reps);
    printf("%5dK %2.2fms %.2f (%d MB)\n", size / 1024, tt, t * 1024 / float(k), int(planSize / (1024 * 1024)));
    while (!v.empty() && v.back().first > tt) { v.pop_back(); }
    v.push_back(std::make_pair(tt, k));
    CHECK(cufftDestroy(plan));
  }

  printf("\n----\n");
  for (auto x : v) {
    isGood(x.second);
    printf("%.1f %.2f\n", x.second / float(1024), x.first);
  }
  CHECK(cudaFree(buf1));
  // CHECK(cudaFree(buf2));
}
