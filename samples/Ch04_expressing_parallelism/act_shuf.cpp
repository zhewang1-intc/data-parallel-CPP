#include <algorithm>
#include <chrono>
#include <cstring>
#include <math.h>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

inline float randn(float _off = -1.f) {
  return 2 * (rand() + 0.5f) / (RAND_MAX + 1.f) + _off;
}

template <typename T>
void shuf_ref(int m, int k, int *idx, std::vector<T> &act) {
  std::vector<T> act_bk = act;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      act[i * k + j] = act_bk[i * k + idx[j]];
}

template <typename T, int ACT_BK_SLM, int SHUF_SLM>
void shuf_tar(queue &q, T *act, int *idx, int m, int k) {
  assert(ACT_BK_SLM >= k);
  q.submit([&](handler &h) {
    // sycl::stream str(8192, 1024, h);
    sycl::local_accessor<T> act_bk_slm{ACT_BK_SLM, h};
    sycl::local_accessor<T> act_shuf_slm{SHUF_SLM, h};
    auto split_k_factor = (k + SHUF_SLM - 1) / SHUF_SLM;
    assert(k % split_k_factor == 0);
    h.parallel_for(
        nd_range(sycl::range<2>(m, k), sycl::range<2>(1, k / split_k_factor)),
        [=](nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
          int i = it.get_global_id(0);
          int j = it.get_global_id(1);
          auto split_idx = j % (k / split_k_factor);
          auto sg = it.get_sub_group();
          // cpy raw act to act_bk_slm & shuf_idx to shuf_idx_slm
          // for utilize simd lane.
          for (int ii = 0; ii < split_k_factor; ii++) {
            act_bk_slm[split_idx + ii * sg.get_local_range()[0]] =
                act[i * k + split_idx + ii * sg.get_local_range()[0]];
          }
          it.barrier();
          act[i * k + j] = act_bk_slm[j];
          auto shuf_idx = idx[j];
          // shuffle activation
          act_shuf_slm[split_idx] = act_bk_slm[shuf_idx];
          it.barrier();
          // write back
          act[i * k + j] = act_shuf_slm[split_idx];
        });
  });
}

template <typename T> void ut(int m, int k) {
  sycl::queue q;
  std::vector<T> act_ref(m * k);
  std::vector<T> act_tar(m * k, 1);
  std::transform(act_ref.begin(), act_ref.end(), act_ref.begin(),
                 [](float) { return static_cast<T>(randn()); });
  std::vector<int> shuf_idx(k);
  for (int i = 0; i < k; i++)
    shuf_idx[i] = i;
  T *act_device = aligned_alloc_device<T>(64, m * k, q);
  int *idx_device = aligned_alloc_device<int>(64, k, q);
  T *act_device_warm_up = aligned_alloc_device<T>(64, m * k, q);
  int *idx_device_warm_up = aligned_alloc_device<int>(64, k, q);
  std::random_shuffle(shuf_idx.begin(), shuf_idx.end());
  q.submit([&](handler &h) {
    h.memcpy(act_device, act_ref.data(), m * k * sizeof(T));
  });
  q.submit([&](handler &h) {
    h.memcpy(idx_device, shuf_idx.data(), k * sizeof(int));
  });
  q.wait();
  shuf_ref(m, k, shuf_idx.data(), act_ref);

  int warm_up = 100;
  for (int i = 0; i < warm_up; i++) {
    shuf_tar<T, 1024, 1024>(q, act_device_warm_up, idx_device_warm_up, m, k);
    q.wait();
  }

  using namespace std::chrono;

  auto m_start = high_resolution_clock::now();
  shuf_tar<T, 1024, 1024>(q, act_device, idx_device, m, k);
  auto m_end = high_resolution_clock::now();
  std::cout << "device cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  q.submit([&](handler &h) {
    h.memcpy(act_tar.data(), act_device, m * k * sizeof(T));
  });
  q.wait();

  // for (int i = 0; i < act_tar.size(); i++) {
  //   std::cout << act_tar[i] << "vs" << act_ref[i] << std::endl;
  // }

  // if (std::all_of(act_tar.begin(), act_tar.end(), [](T i) { return i == 0;
  // })) {
  if (std::equal(act_tar.begin(), act_tar.end(), act_ref.begin())) {
    printf("pass\n");
  } else {
    printf("fail\n");
  }
}

void dump_device_info() {
  auto d_selector = sycl::default_selector_v;
  sycl::queue q;
  std::cout << "name: " + q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "max XVE: " +
                   std::to_string(
                       q.get_device()
                           .get_info<sycl::info::device::max_compute_units>())
            << std::endl;
  std::cout
      << "thread_per_XVE: " +
             std::to_string(
                 q.get_device()
                     .get_info<
                         sycl::info::device::ext_intel_gpu_hw_threads_per_eu>())
      << std::endl;
  std::cout << "max freq: " +
                   std::to_string(
                       q.get_device()
                           .get_info<sycl::info::device::max_clock_frequency>())
            << std::endl;
  std::cout
      << "slm/L1-cache size: " +
             std::to_string(
                 q.get_device().get_info<sycl::info::device::local_mem_size>())
      << std::endl;
  std::cout
      << "global-mem size: " +
             std::to_string(
                 q.get_device().get_info<sycl::info::device::global_mem_size>())
      << std::endl;
  std::cout
      << "L2 cache size: " +
             std::to_string(
                 q.get_device()
                     .get_info<sycl::info::device::global_mem_cache_size>())
      << std::endl;
  // bank: 16
}

int main() {
  dump_device_info();
  ut<half>(1024, 1024);
  //   ut<__fp16>(1, 11008);
}