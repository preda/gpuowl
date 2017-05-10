#include <cstdio>

const int N = 4 * 1024 * 1024;

int *read(const char *name) {
  int *data1 = new int[N];
  FILE *fi = fopen(name, "rb");
  fread(data1, sizeof(int) * N, 1, fi);
  fclose(fi);
  int *data2 = new int[N];
  for (int line = 0; line < 2048; ++line) {
    for (int col = 0; col < 1024; ++col) {
      data2[(col * 2048 + line) * 2] = data1[(line * 1024 + col) * 2];
      data2[(col * 2048 + line) * 2 + 1] = data1[(line * 1024 + col) * 2 + 1];
    }
  }
  delete[] data1;
  return data2;
}

void cmp(int *a, int *b) {
  for (int i = 0; i < N; ++i) {
    if (a[i] != b[i]) {
      int col  = i / 2 / 2048;
      int line = i / 2 % 2048;
      printf("%d (x %d y %d) %d %d\n", i, col, line, a[i], b[i]);
    }
  }
}

int main(int argc, char **argv) {
  int *a = read(argv[1]);
  int *b = read(argv[2]);
  for (int i = 0; i < 10; ++i) { printf("%6d ", a[i]); }
  printf("\n");
  for (int i = 0; i < 10; ++i) { printf("%6d ", b[i]); }
  printf("\n");
  cmp(a, b);
}
