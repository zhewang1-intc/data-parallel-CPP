#include <sycl/sycl.hpp>
using namespace sycl;

#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>
#include <math.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline float randn(float _off = -1.f)
{
    return 2 * (rand() + 0.5f) / (RAND_MAX + 1.f) + _off;
}

struct bit4x2
{
    int8_t x : 4;
    int8_t y : 4;
    bit4x2(int8_t v) : x(v), y(v) {}
    bit4x2() : x(0), y(0) {}
};

struct int4x2 : bit4x2
{
    int4x2(int8_t v) : bit4x2(v) {}
    int4x2() : bit4x2() {}
    static int8_t convert(int8_t src)
    {
        int32_t dst = src;
        dst = dst >= 0 ? dst + 8 : dst - 8;
        dst = dst / 16;
        dst = dst > 7 ? 7 : dst;
        dst = dst < -8 ? -8 : dst;
        return static_cast<int8_t>(dst);
    }
};

class CompressWei4Bit
{
public:
    CompressWei4Bit(int K, int N, int blksize, bool sym = false) : _K(K), _N(N), _blksize(blksize), _sym(sym)
    {
        assert(sym == false);
        assert((_K * _N) % 2 == 0); // no consider padding now.
        assert(_K % blksize == 0);
        _write_buf = (char *)malloc(get_buf_size());
    }
    virtual ~CompressWei4Bit()
    {
        if (_write_buf != nullptr)
            free(_write_buf);
    }

    void serialize(void *buf)
    {
        size_t offset = 0;
        memcpy((char *)buf + offset, &_N, sizeof(_N));
        offset += sizeof(_N);
        memcpy((char *)buf + offset, &_K, sizeof(_K));
        offset += sizeof(_K);
        memcpy((char *)buf + offset, &_blksize, sizeof(_blksize));
        offset += sizeof(_blksize);
        memcpy((char *)buf + offset, &_sym, sizeof(_sym));
        offset += sizeof(_sym);
        memcpy((char *)buf + offset, _write_buf, get_buf_size());
    }

    void deserialize(void *buf)
    {
        size_t offset = 0;
        memcpy(&_N, (char *)buf + offset, sizeof(_N));
        offset += sizeof(_N);
        memcpy(&_K, (char *)buf + offset, sizeof(_K));
        offset += sizeof(_K);
        memcpy(&_blksize, (char *)buf + offset, sizeof(_blksize));
        offset += sizeof(_blksize);
        memcpy(&_sym, (char *)buf + offset, sizeof(_sym));
        offset += sizeof(_sym);
        memcpy(_write_buf, (char *)buf + offset, get_buf_size());
    }

    size_t get_serialize_size()
    {
        return get_meta_data_size() + get_buf_size();
    }

    size_t get_meta_data_size()
    {
        return sizeof(_N) + sizeof(_K) + sizeof(_blksize) + sizeof(_sym);
    }

    void *get_4bit_wei_ptr()
    {
        return _write_buf;
    }

    void *get_scale_ptr()
    {
        return _write_buf + get_4bit_wei_size();
    }

private:
    size_t get_4bit_wei_size()
    {
        return _N * _K / 2;
    }
    size_t get_scale_size()
    {
        return _K / _blksize * _N * sizeof(float);
    }
    size_t get_zp_size()
    {
        return 0;
    }
    size_t get_buf_size()
    {

        return get_4bit_wei_size() + get_scale_size() + get_zp_size();
    }

    int _N, _K, _blksize;
    bool _sym;
    char *_write_buf;
};

void s8_quant_row_blk(const float *srcptr, int8_t *dstptr, int row, int col, int ld_src,
                      int ld_dst, float *scales, int blocksize)
{
    int raw_blocksize = blocksize;
    for (int i = 0; i < col; i++)
    {
        int align_row_loop = row / blocksize * blocksize;
        int j = 0;

        auto s4_fullrange_calc_store_scale_and_quantv_sym = [&](int blocksize)
        {
            float amax = 0.f, max = 0.f;
            for (size_t ij = 0; ij < blocksize; ij++)
            {
                auto v = srcptr[(j + ij) * ld_src + i];
                if (amax < std::abs(v))
                {
                    amax = std::abs(v);
                    max = v;
                }
            }
            float scale = max / -8.f;
            float rscale = scale != 0.f ? 1.f / scale : 0.f;
            scales[j / raw_blocksize * ld_dst + i] = scale;
            for (size_t ij = 0; ij < blocksize; ij++)
            {
                auto quant_v = srcptr[(j + ij) * ld_src + i] * rscale;
                int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
                dstptr[(j + ij) * ld_dst + i] = x << 4;
            }
        };
        for (; j < align_row_loop; j += blocksize)
            s4_fullrange_calc_store_scale_and_quantv_sym(blocksize);
        if (j < row)
            s4_fullrange_calc_store_scale_and_quantv_sym(row - align_row_loop);
    }
}

void compress_s8_s4(const int8_t *srcptr, int4x2 *dstptr, int row, int col,
                    int ld_src, int ld_dst)
{
    for (int j = 0; j < row; j++)
    {
        for (int ii = 0; ii < col; ii += 2)
        {
            int4x2 tmp;
            tmp.x = int4x2::convert(srcptr[j * ld_src + ii + 0]);
            tmp.y = int4x2::convert(srcptr[j * ld_src + ii + 1]);
            dstptr[j * ld_dst / 2 + ii / 2] = tmp;
        }
    }
}

void *quantize(float *weight, int k, int n, int blksize, bool transpose, std::string weight_type, std::string cmpt_type)
{
    CompressWei4Bit compress_wei(k, n, blksize);
    void *ret = malloc(compress_wei.get_serialize_size());
    assert(!transpose);
    if (weight_type == "s4fullrange_scalef32")
    {
        std::vector<int8_t> s8quant_tmp(k * n);
        float *scale = reinterpret_cast<float *>(compress_wei.get_scale_ptr());
        s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, n, n, scale, blksize);
        int4x2 *wei = reinterpret_cast<int4x2 *>(compress_wei.get_4bit_wei_ptr());
        compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
        compress_wei.serialize(ret);
    }
    else
    {
        assert(0);
    }
    return ret;
}

enum SIGN_INT_TYPE
{
    S4_CLIP,
    S4_FULLRANGE
};

template <SIGN_INT_TYPE S4_T>
inline int8_t get_s8(int8_t v)
{
    switch (S4_T)
    {
    case S4_CLIP:
        return v << 4;
    case S4_FULLRANGE:
        v &= 0x0f;
        return v - 8;
    default:
        assert(false);
        break;
    }
    return int8_t(0);
}

template <SIGN_INT_TYPE S4_T>
void decompress_s4_s8(int4x2 *srcptr, int8_t *dstptr, int row, int col, int ld_src, int ld_dst)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j += 2)
        {
            auto tmp = srcptr[i * ld_src / 2 + j / 2];
            dstptr[i * ld_dst + j + 0] = get_s8<S4_T>(tmp.x);
            dstptr[i * ld_dst + j + 1] = get_s8<S4_T>(tmp.y);
        }
    }
}

template <typename _DST_T, typename _S_T>
void decompress_kblock_s8_f32(int8_t *srcptr, _DST_T *dstptr, int row, int col, int ld_src, int ld_dst,
                              _S_T *scales, int8_t *zero_points, int k_offset, int kblock, int NPad)
{
    for (int i = 0; i < row; i++)
    {
        int kpos = (k_offset + i) / kblock;
        auto sptr = scales + kpos * NPad;
        for (int j = 0; j < col; j += 1)
        {
            float tmp = (float)(srcptr[i * ld_src + j]);
            if (zero_points != nullptr)
                tmp -= (float)(zero_points[kpos * NPad + j]);
            dstptr[i * ld_dst + j] = static_cast<_DST_T>(tmp * sptr[j]);
        }
    }
}

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N>
void gpu_dequant_s4fullrange_f32_KxN(queue &q, buffer<int8_t, 2> &src, buffer<float, 2> &dst, buffer<float, 1> &scale, int k, int n, int k_pos, int n_pos)
{
    q.submit([&](handler &h)
             {
                 accessor s4_wei{src, h};
                 accessor fp32_wei{dst, h};
                 accessor s{scale, h};
                 range global{TILE_K, TILE_N};
                 range local{LOCAL_K, LOCAL_N};
                 h.parallel_for(nd_range{global,local},[=](nd_item<2> it){
int j=it.get_global_id(0);
int i=it.get_global_id(1);
fp32_wei[i][j]=i+j;
                 }); });
}

std::vector<float> dequantize(void *s4_wei, int k, int n, int blksize, bool transpose, std::string weight_type, std::string cmpt_type)
{
    assert(!transpose);
    CompressWei4Bit obj(k, n, blksize, false);
    obj.deserialize(s4_wei);
    std::vector<float> f32_wei(k * n);
    std::vector<int8_t> s8_wei(k * n);
    int4x2 *raw_wei = reinterpret_cast<int4x2 *>(obj.get_4bit_wei_ptr());
    float *scale = reinterpret_cast<float *>(obj.get_scale_ptr());
    if (weight_type == "s4fullrange_scalef32")
    {
        decompress_s4_s8<S4_FULLRANGE>(raw_wei, s8_wei.data(), k, n, n, n);
        decompress_kblock_s8_f32(s8_wei.data(), f32_wei.data(), k, n, n, n, reinterpret_cast<float *>(obj.get_scale_ptr()), nullptr, 0, blksize, n);
    }
    else
    {
        assert(0);
    }
    return f32_wei;
}

void dump_matrix(float *mat, int k, int n)
{
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    int K = 16;
    int N = 16;
    int blksize = 4;
    bool sym = false;
    std::vector<float> f32wei(K * N);
    for (int i = 0; i < K * N; i++)
        f32wei[i] = randn() * 10;
    dump_matrix(f32wei.data(), K, N);
    void *s4_wei = quantize(f32wei.data(), K, N, blksize, false, "s4fullrange_scalef32", "fp32");
    std::cout << "================" << std::endl;
    auto dq_wei = dequantize(s4_wei, K, N, blksize, false, "s4fullrange_scalef32", "fp32");
    dump_matrix(dq_wei.data(), K, N);

    buffer<float, 2> dst_buf(dq_wei.data(), range<2>(K, N));
    buffer<float, 1> scale_buf(reinterpret_cast<float *>(s4_wei), range<1>(K / blksize * N));
    buffer<int8_t, 2> src_buf(reinterpret_cast<int8_t *>(s4_wei), range<2>(K, N));
    queue q;

    gpu_dequant_s4fullrange_f32_KxN<16, 16, 4, 4>(q, src_buf, dst_buf, scale_buf, 16, 16, 0, 0);

    q.wait();

    host_accessor hs(dst_buf);
    dump_matrix(hs.get_pointer(), K, N);
    free(s4_wei);
    return 0;
}