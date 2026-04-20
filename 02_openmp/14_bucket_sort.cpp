#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

// declare omp reduction on std::vector data;
// just adds two std::vectors together using std::plus;
// initializer builds a private std::vector variable per thread;
#pragma omp declare reduction(                                                 \
        vector_int_add : std::vector<int> : std::transform(                    \
                omp_out.begin(), omp_out.end(), omp_in.begin(),                \
                    omp_out.begin(), std::plus<int>()))                        \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

int main() {
  int n = 100000000;
  int range = 50000;
  std::vector<int> key(n);
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    // printf("%d ", key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);

#pragma omp parallel for reduction(vector_int_add : bucket)
  for (int i = 0; i < n; i++) {
    bucket[key[i]]++;
  }

  std::vector<int> offset(range, 0);

  // shift by one element to the right
#pragma omp parallel for
  for (int i = 1; i < range; i++)
    offset[i] = bucket[i - 1];
  std::vector<int> tmp(offset.size());

// prefix _exclusive_ scan sum over offset (= bucket)
#pragma omp parallel
  for (int j = 1; j < range; j <<= 1) {
#pragma omp for
    for (int i = 0; i < range; i++)
      tmp[i] = offset[i];
#pragma omp for
    for (int i = j; i < range; i++)
      offset[i] += tmp[i - j];
  }

#pragma omp parallel for
  for (int i = 0; i < range; i++) {
    int j = offset[i];
    for (; bucket[i] > 0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i = n - 20; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}
