#include <cstdio>
#include <cstdlib>
#include <vector>

int GRAIN_SIZE = 50;

void merge(std::vector<int> &vec, int begin, int mid, int end) {
  std::vector<int> tmp(end - begin + 1);
  int left = begin;
  int right = mid + 1;
  for (int i = 0; i < tmp.size(); i++) {
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++];
  }
  for (int i = 0; i < tmp.size(); i++)
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int> &vec, int begin, int end) {
  if (begin < end) {
    int mid = (begin + end) / 2;
#pragma omp taskgroup
    {
      if (mid - begin > GRAIN_SIZE) {
#pragma omp task shared(vec)
        {
          merge_sort(vec, begin, mid);
        }
      } else {
        merge_sort(vec, begin, mid);
      }
      if (end - mid + 1 > GRAIN_SIZE) {
#pragma omp task shared(vec)
        {
          merge_sort(vec, mid + 1, end);
        }
      } else {
        merge_sort(vec, mid + 1, end);
      }
    }
#pragma omp taskwait
    merge(vec, begin, mid, end);
  }
}

int main() {
  int n = 100000;
  std::vector<int> vec(n);
  for (int i = 0; i < n; i++) {
    vec[i] = rand() % (10 * n);
    // printf("%d ", vec[i]);
  }
  printf("\n");
#pragma omp parallel
  {
#pragma omp single
    merge_sort(vec, 0, n - 1);
  }
  for (int i = n - 20; i < n; i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}
