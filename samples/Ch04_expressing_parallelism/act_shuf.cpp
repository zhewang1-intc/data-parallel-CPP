#include <algorithm>
#include <chrono>
#include <cstring>
#include <math.h>
#include <sycl/ext/intel/esimd.hpp>
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

template <typename T> void shuf_tar(queue &q, T *act, int *idx, int m, int k) {
  q.submit([&](handler &h) {
    sycl::local_accessor<T> act_shuf_slm{256, h};
    unsigned int *sync_ptr = malloc_shared<unsigned int>(1, q);
    *sync_ptr = 0;
    int dim_2 = 1024;
    int dim_1 = (m * k + 1023) / 1024;
    dim_1 = dim_1 > 1024 ? 1024 : dim_1;
    int dim_0 = dim_1 == 1024 ? (m * k + 1024 * 1024 - 1) / (1024 * 1024) : 1;
    // std::cout << dim_0 << " " << dim_1 << " " << dim_2 << std::endl;
    // sycl::stream str(8192, 8192, h);
    h.parallel_for(nd_range(sycl::range<3>(dim_0, dim_1, dim_2),
                            sycl::range<3>(1, 1, 256)),
                   [=](nd_item<3> it) [[intel::reqd_sub_group_size(32)]] {
                     auto linear_idx = it.get_global_linear_id();
                     if (linear_idx < m * k) {
                       auto k_offset = linear_idx % k;
                       auto m_offset = linear_idx / k;
                       auto shuf_idx = idx[k_offset];
                       auto local_idx = it.get_local_linear_id();
                       act_shuf_slm[local_idx] = act[m_offset * k + shuf_idx];
                       if (local_idx == 0) {
                         atomic_ref<unsigned int, memory_order::acq_rel,
                                    memory_scope::device,
                                    access::address_space::global_space>
                             sync_atomic(*sync_ptr);
                         sync_atomic++;
                         while (sync_atomic < m * k / 256)
                           ;
                       }
                       it.barrier(access::fence_space::global_and_local);
                       act[linear_idx] = act_shuf_slm[local_idx];
                     }
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

  int warm_up = 1000;
  for (int i = 0; i < warm_up; i++) {
    shuf_tar<T>(q, act_device_warm_up, idx_device_warm_up, m, k);
    q.wait();
  }

  using namespace std::chrono;

  auto m_start = high_resolution_clock::now();
  shuf_ref(m, k, shuf_idx.data(), act_ref);
  auto m_end = high_resolution_clock::now();
  std::cout << "CPU cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  m_start = high_resolution_clock::now();
  shuf_tar<T>(q, act_device, idx_device, m, k);
  q.wait();
  m_end = high_resolution_clock::now();
  std::cout << "GPU cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  q.submit([&](handler &h) {
    h.memcpy(act_tar.data(), act_device, m * k * sizeof(T));
  });
  q.wait();

  // for (int i = 0; i < act_tar.size(); i++) {
  //   // std::cout << act_tar[i] << "vs" << act_ref[i] << std::endl;
  //   std::cout << act_tar[i] - act_ref[i] << std::endl;
  // }

  // if (std::all_of(act_tar.begin(), act_tar.end(), [](T i) { return i == 0;
  // })) {
  if (std::equal(act_tar.begin(), act_tar.end(), act_ref.begin())) {
    printf("pass\n");
  } else {
    printf("fail\n");
  }
}

template <typename T>
void gather_ut(int input_elt_num = 32, int gather_elt_num = 8) {
  sycl::queue q;
  T *input_vec = malloc_shared<T>(input_elt_num, q);
  uint32_t *gather_offset = malloc_shared<uint32_t>(gather_elt_num, q);
  assert(input_elt_num % gather_elt_num == 0);
  auto gather_stride = input_elt_num / gather_elt_num;
  for (int i = 0; i < input_elt_num; i++)
    input_vec[i] = i;
  for (int i = 0; i < gather_elt_num; i++)
    gather_offset[i] = i * gather_stride * sizeof(T);

  q.submit([&](handler &h) {
    h.parallel_for(
        nd_range(sycl::range<1>(input_elt_num / 32),
                 sycl::range<1>(input_elt_num / 32)),
        [=](nd_item<1> it) [[intel::sycl_explicit_simd]] {
          auto esimd_gather =
              sycl::ext::intel::esimd::block_load<uint32_t, 8>(gather_offset);
          auto esimd_vec =
              sycl::ext::intel::esimd::gather<T, 8>(input_vec, esimd_gather);
          sycl::ext::intel::esimd::block_store(input_vec, esimd_vec);
        });
  });
  q.wait();
  for (int i = 0; i < input_elt_num; i++)
    std::cout << input_vec[i] << std::endl;
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
  std::cout
      << "max work-group size: "
      << std::to_string(
             q.get_device().get_info<sycl::info::device::max_work_group_size>())
      << std::endl;
  std::cout
      << "max sub-group size: "
      << std::to_string(
             q.get_device().get_info<sycl::info::device::max_num_sub_groups>())
      << std::endl;
  // bank: 16
}

int main() {
  // dump_device_info();
  // ut<half>(32, 4096);
  // ut<half>(1, 11008);
  gather_ut<half>();
  // ut<half>(6, 11008);
}