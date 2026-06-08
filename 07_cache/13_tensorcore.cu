#include <chrono>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <typeinfo>

using namespace std;
using namespace nvcuda;

#define BM 128
#define BN 256
#define BK 64
#define APAD 8
#define BPAD 8
#define GROUP 8
#define WARPS_M 2
#define WARPS_N 4
#define NWARPS (WARPS_M * WARPS_N)
#define NTHREADS (NWARPS * 32)
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define MMI (WM / 16)
#define NNJ (WN / 8)
#define LDAh (BM + APAD)
#define LDBn (BK + BPAD)
#define ASTAGE (BK * LDAh)
#define BSTAGE (BN * LDBn)
#define SMEM_HALFS (2 * (ASTAGE + BSTAGE))
#define AVEC (BK * BM / 8 / NTHREADS)
#define BVEC (BK * BN / 8 / NTHREADS)

__global__ void convert_kernel(const float *src, half *dst, size_t n) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    float4 s = *reinterpret_cast<const float4 *>(&src[i]);
    *reinterpret_cast<half2 *>(&dst[i]) = __floats2half2_rn(s.x, s.y);
    *reinterpret_cast<half2 *>(&dst[i + 2]) = __floats2half2_rn(s.z, s.w);
  }
}

__device__ __forceinline__ void ldm_trans_x4(uint32_t *r, const half *p) {
  unsigned ad = (unsigned)__cvta_generic_to_shared(p);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(ad));
}

__device__ __forceinline__ void ldm_x2(uint32_t *r, const half *p) {
  unsigned ad = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(ad));
}

__device__ __forceinline__ void mma_m16n8k16(float *c, const uint32_t *a,
                                             const uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void
load_global(float4 *ra, float4 *rb, const half *d_a, const half *d_b, int M,
            int K, int offm, int offn, int k0, int tid) {
#pragma unroll
  for (int v = 0; v < AVEC; v++) {
    int lin = (tid + v * NTHREADS) * 8;
    int kk = lin / BM, mm = lin % BM;
    ra[v] = *reinterpret_cast<const float4 *>(&d_a[(k0 + kk) * M + offm + mm]);
  }
#pragma unroll
  for (int v = 0; v < BVEC; v++) {
    int lin = (tid + v * NTHREADS) * 8;
    int nn = lin / BK, kk = lin % BK;
    rb[v] = *reinterpret_cast<const float4 *>(&d_b[(offn + nn) * K + k0 + kk]);
  }
}

__device__ __forceinline__ void store_shared(half *Asb, half *Bsb, float4 *ra,
                                             float4 *rb, int tid) {
#pragma unroll
  for (int v = 0; v < AVEC; v++) {
    int lin = (tid + v * NTHREADS) * 8;
    int kk = lin / BM, mm = lin % BM;
    *reinterpret_cast<float4 *>(&Asb[kk * LDAh + mm]) = ra[v];
  }
#pragma unroll
  for (int v = 0; v < BVEC; v++) {
    int lin = (tid + v * NTHREADS) * 8;
    int nn = lin / BK, kk = lin % BK;
    *reinterpret_cast<float4 *>(&Bsb[nn * LDBn + kk]) = rb[v];
  }
}

__global__ void __launch_bounds__(NTHREADS)
    kernel(int dim_m, int dim_n, int dim_k, half *d_a, half *d_b, float *d_c) {
  const int M = dim_m, N = dim_n, K = dim_k;
  const int GMn = M / BM, GNn = N / BN;
  int pid = blockIdx.x;
  int npg = GROUP * GNn;
  int g = pid / npg;
  int first_m = g * GROUP;
  int gsize = GMn - first_m;
  if (gsize > GROUP)
    gsize = GROUP;
  int bx = first_m + (pid % gsize);
  int by = (pid % npg) / gsize;

  const int offm = BM * bx, offn = BN * by;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
  const int wm = warp % WARPS_M, wn = warp / WARPS_M;
  const int cgid = lane >> 2, tig = lane & 3;

  extern __shared__ half smem[];
  half *As = smem;
  half *Bs = smem + 2 * ASTAGE;

  float acc[MMI][NNJ][4];
#pragma unroll
  for (int i = 0; i < MMI; i++)
#pragma unroll
    for (int j = 0; j < NNJ; j++)
#pragma unroll
      for (int t = 0; t < 4; t++)
        acc[i][j][t] = 0.0f;

  float4 ra[AVEC], rb[BVEC];
  const int NK = K / BK;

  load_global(ra, rb, d_a, d_b, M, K, offm, offn, 0, tid);
  store_shared(As, Bs, ra, rb, tid);
  __syncthreads();

  const int a_k = (lane % 8) + 8 * (lane / 16);
  const int a_m = 8 * ((lane / 8) & 1);
  const int b_n = lane % 8;
  const int b_k = 8 * (lane / 8);

  int buf = 0;
  for (int kt = 0; kt < NK; kt++) {
    half *cAs = As + buf * ASTAGE;
    half *cBs = Bs + buf * BSTAGE;
    if (kt + 1 < NK)
      load_global(ra, rb, d_a, d_b, M, K, offm, offn, (kt + 1) * BK, tid);
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks++) {
      int ak = ks * 16;
      uint32_t Areg[MMI][4];
#pragma unroll
      for (int mi = 0; mi < MMI; mi++) {
        int am = wm * WM + mi * 16;
        ldm_trans_x4(Areg[mi], cAs + (ak + a_k) * LDAh + (am + a_m));
      }
      uint32_t Breg[NNJ][2];
#pragma unroll
      for (int nj = 0; nj < NNJ; nj++) {
        int bn = wn * WN + nj * 8;
        ldm_x2(Breg[nj], cBs + (bn + b_n) * LDBn + (ak + b_k));
      }
#pragma unroll
      for (int mi = 0; mi < MMI; mi++)
#pragma unroll
        for (int nj = 0; nj < NNJ; nj++)
          mma_m16n8k16(acc[mi][nj], Areg[mi], Breg[nj]);
    }
    if (kt + 1 < NK) {
      store_shared(As + (buf ^ 1) * ASTAGE, Bs + (buf ^ 1) * BSTAGE, ra, rb,
                   tid);
      __syncthreads();
      buf ^= 1;
    }
  }

#pragma unroll
  for (int mi = 0; mi < MMI; mi++)
#pragma unroll
    for (int nj = 0; nj < NNJ; nj++) {
      int base_m = offm + wm * WM + mi * 16 + cgid;
      int base_n = offn + wn * WN + nj * 8 + tig * 2;
      d_c[(base_n + 0) * M + base_m] = acc[mi][nj][0];
      d_c[(base_n + 1) * M + base_m] = acc[mi][nj][1];
      d_c[(base_n + 0) * M + base_m + 8] = acc[mi][nj][2];
      d_c[(base_n + 1) * M + base_m + 8] = acc[mi][nj][3];
    }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C_ref, *C_cand;
  cudaMallocManaged(&A, (size_t)m * k * sizeof(float));
  cudaMallocManaged(&B, (size_t)k * n * sizeof(float));
  cudaMallocManaged(&C_ref, (size_t)m * n * sizeof(float));
  cudaMallocManaged(&C_cand, (size_t)m * n * sizeof(float));
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[(size_t)k * i + j] = drand48();
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[(size_t)n * i + j] = drand48();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      C_ref[(size_t)m * i + j] = C_cand[(size_t)m * i + j] = 0;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  // cuBLAS reference: 2 warmup iters, then ``Nt`` timed iters averaged
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2)
      tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A,
                 CUDA_R_32F, m, B, CUDA_R_32F, k, &beta, C_ref, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops =
      (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_gflops = double(num_flops) / tcublas / 1.0e9;
  // Candidate kernel: 2 warmup iters, then ``Nt`` timed iters averaged
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2)
      tic = chrono::steady_clock::now();
    static half *Ah = nullptr, *Bh = nullptr;
    static bool _inited = false;
    if (!_inited) {
      cudaMalloc(&Ah, (size_t)m * k * sizeof(half));
      cudaMalloc(&Bh, (size_t)k * n * sizeof(half));
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)(SMEM_HALFS * sizeof(half)));
      _inited = true;
    }
    size_t nA = (size_t)m * k, nB = (size_t)k * n;
    convert_kernel<<<(unsigned)((nA / 4 + 255) / 256), 256>>>(A, Ah, nA);
    convert_kernel<<<(unsigned)((nB / 4 + 255) / 256), 256>>>(B, Bh, nB);
    dim3 block = dim3(NTHREADS);
    dim3 grid = dim3((m / BM) * (n / BN));
    kernel<<<grid, block, (int)(SMEM_HALFS * sizeof(half))>>>(m, n, k, Ah, Bh,
                                                              C_cand);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcandidate = chrono::duration<double>(toc - tic).count() / Nt;
  double candidate_gflops = double(num_flops) / tcandidate / 1.0e9;
  double err = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      err += fabs(C_ref[(size_t)m * i + j] - C_cand[(size_t)m * i + j]);
    }
  }
  double mean_err = err / double((size_t)n * m);
  printf("CUBLAS: %.2f Gflops, CANDIDATE: %.2f Gflops\n", cublas_gflops,
         candidate_gflops);
  printf("error: %lf\n", mean_err);
  printf("RESULT cublas_gflops=%.6f candidate_gflops=%.6f"
         " cublas_ms=%.6f candidate_ms=%.6f mean_abs_error=%.9f"
         " num_flops=%lld m=%d n=%d k=%d\n",
         cublas_gflops, candidate_gflops, tcublas * 1000.0, tcandidate * 1000.0,
         mean_err, (long long)num_flops, m, n, k);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_ref);
  cudaFree(C_cand);
  cublasDestroy(cublas_handle);
  return 0;
}
