#include <cstdio>
#include <cstdlib>

__global__ void clear_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range)
    bucket[i] = 0;
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bucket[key[i]], 1);
}

__global__ void write_sorted_key(int *key, int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int j = 0;
    for (int k = 0; k < i; k++)
      j += bucket[k];
    for (int k = 0; k < bucket[i]; k++)
      key[j + k] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  int threads = 256;
  clear_bucket<<<(range + threads - 1) / threads, threads>>>(bucket, range);
  count_bucket<<<(n + threads - 1) / threads, threads>>>(key, bucket, n);
  write_sorted_key<<<(range + threads - 1) / threads, threads>>>(key, bucket,
                                                                 range);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
