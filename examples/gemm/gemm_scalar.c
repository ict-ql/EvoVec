#include <stdint.h>

#define GEMM_M 16
#define GEMM_N 16
#define GEMM_K 64

#if defined(__clang__) || defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE void gemm_i8_i32_16x16x64(
    int32_t c[GEMM_M][GEMM_N],
    const int8_t a[GEMM_M][GEMM_K],
    const int8_t b[GEMM_K][GEMM_N]) {
  for (int row = 0; row < GEMM_M; ++row) {
    for (int col = 0; col < GEMM_N; ++col) {
      int32_t acc = c[row][col];
      for (int k = 0; k < GEMM_K; ++k) {
        acc += (int32_t)a[row][k] * (int32_t)b[k][col];
      }
      c[row][col] = acc;
    }
  }
}

NOINLINE void gemm_u8s8_i32_16x16x64(
    int32_t c[GEMM_M][GEMM_N],
    const uint8_t a[GEMM_M][GEMM_K],
    const int8_t b[GEMM_K][GEMM_N]) {
  for (int row = 0; row < GEMM_M; ++row) {
    for (int col = 0; col < GEMM_N; ++col) {
      int32_t acc = c[row][col];
      for (int k = 0; k < GEMM_K; ++k) {
        acc += (int32_t)a[row][k] * (int32_t)b[k][col];
      }
      c[row][col] = acc;
    }
  }
}
