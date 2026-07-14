#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(__linux__) && defined(__x86_64__)
#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

#define GEMM_M 160
#define GEMM_N 160
#define GEMM_K 640
#define DEFAULT_ITERS 100000

#if defined(__linux__) && defined(__x86_64__)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

#if defined(__clang__) || defined(__GNUC__)
#define ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64
#endif

void gemm_i8_i32_16x16x64(
    int32_t c[GEMM_M][GEMM_N],
    const int8_t a[GEMM_M][GEMM_K],
    const int8_t b[GEMM_K][GEMM_N]);

static double now_seconds(void) {
#if defined(_WIN32)
  LARGE_INTEGER freq;
  LARGE_INTEGER counter;
  if (!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&counter)) {
    fprintf(stderr, "QueryPerformanceCounter failed\n");
    exit(2);
  }
  return (double)counter.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime");
    exit(2);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static long parse_iters(int argc, char **argv) {
  if (argc < 2) {
    return DEFAULT_ITERS;
  }
  char *end = NULL;
  long value = strtol(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || value <= 0) {
    fprintf(stderr, "usage: %s [positive_iterations]\n", argv[0]);
    exit(2);
  }
  return value;
}

static void request_amx_permission(void) {
#if defined(__linux__) && defined(__x86_64__)
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc != 0) {
    fprintf(stderr,
            "AMX XTILEDATA permission failed: errno=%d (%s)\n",
            errno,
            strerror(errno));
    exit(2);
  }
  fprintf(stderr, "AMX XTILEDATA permission OK\n");
#endif
}

static void init_inputs(
    int32_t c[GEMM_M][GEMM_N],
    int8_t a[GEMM_M][GEMM_K],
    int8_t b[GEMM_K][GEMM_N]) {
  for (int row = 0; row < GEMM_M; ++row) {
    for (int col = 0; col < GEMM_N; ++col) {
      c[row][col] = (row == col) ? 1 : 0;
    }
  }
  for (int row = 0; row < GEMM_M; ++row) {
    for (int k = 0; k < GEMM_K; ++k) {
      a[row][k] = (int8_t)(((row * 17 + k * 3) % 3) - 1);
    }
  }
  for (int k = 0; k < GEMM_K; ++k) {
    for (int col = 0; col < GEMM_N; ++col) {
      b[k][col] = (int8_t)(((k * 5 + col * 11) % 3) - 1);
    }
  }
}

static int64_t checksum_matrix(const int32_t c[GEMM_M][GEMM_N]) {
  int64_t checksum = 0;
  for (int row = 0; row < GEMM_M; ++row) {
    for (int col = 0; col < GEMM_N; ++col) {
      checksum += (int64_t)c[row][col] * (int64_t)(row * GEMM_N + col + 1);
    }
  }
  return checksum;
}

int main(int argc, char **argv) {
  static ALIGN64 int32_t c[GEMM_M][GEMM_N];
  static ALIGN64 int8_t a[GEMM_M][GEMM_K];
  static ALIGN64 int8_t b[GEMM_K][GEMM_N];

  const long iters = parse_iters(argc, argv);
  request_amx_permission();
  init_inputs(c, a, b);

  const double start = now_seconds();
  for (long iter = 0; iter < iters; ++iter) {
    gemm_i8_i32_16x16x64(c, a, b);
  }
  const double end = now_seconds();

  const double elapsed = end - start;
  const int64_t checksum = checksum_matrix(c);
  printf("iters=%ld elapsed_sec=%.9f ns_per_call=%.3f checksum=%" PRId64 "\n",
         iters,
         elapsed,
         elapsed * 1e9 / (double)iters,
         checksum);
  return 0;
}
