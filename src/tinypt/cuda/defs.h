#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef __NVCC__
#include "tinypt/cpu/defs.h"
#include <vector>
#endif

namespace tinypt {
namespace cuda {

static inline bool is_enabled() {
#ifdef TINYPT_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

static inline void check_cuda_status(cudaError status, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s:%d: cuda error: %s\n", file, line, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError status = (call);                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            cudaDeviceReset();                                                                                         \
            TINYPT_THROW << "cuda error: " << cudaGetErrorString(status);                                              \
        }                                                                                                              \
    } while (false)

static constexpr float INF = 1e20;
static constexpr float EPS = 1e-3;

__device__ static inline bool is_close(float x, float y) { return std::abs(x - y) < EPS; }

__device__ static inline float square(float x) { return x * x; }

template <typename T>
__device__ static inline void dev_swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
struct Array {
    Array() : _data(nullptr), _size(0) {}
    Array(T *data, uint32_t size) : _data(data), _size(size) {}

#ifndef __NVCC__
    static Array create(size_t n) { return create(std::vector<T>(n)); }

    static Array create(const std::vector<T> &cpu_array) {
        T *data = nullptr;
        if (!cpu_array.empty()) {
            CHECK_CUDA(cudaMalloc(&data, sizeof(T) * cpu_array.size()));
            CHECK_CUDA(cudaMemcpy(data, cpu_array.data(), sizeof(T) * cpu_array.size(), cudaMemcpyHostToDevice));
        }
        return Array(data, cpu_array.size());
    }

    static void destroy(Array &array) {
        if (array._data) {
            CHECK_CUDA(cudaFree(array._data));
            array._data = nullptr;
            array._size = 0;
        }
    }

    std::vector<T> to_cpu() const {
        std::vector<T> cpu_array(_size);
        CHECK_CUDA(cudaMemcpy(cpu_array.data(), _data, sizeof(T) * _size, cudaMemcpyDeviceToHost));
        return cpu_array;
    }

    void from_cpu(const std::vector<T> &cpu_array) const {
        TINYPT_CHECK(_size == cpu_array.size());
        CHECK_CUDA(cudaMemcpy(_data, cpu_array.data(), sizeof(T) * _size, cudaMemcpyHostToDevice));
    }
#endif
    __device__ const T &operator[](int idx) const { return _data[idx]; }
    __device__ T &operator[](int idx) { return _data[idx]; }
    __device__ uint32_t size() const { return _size; }
    __device__ bool empty() const { return _size == 0; }
    __device__ const T *data() const { return _data; }

  private:
    T *_data;
    uint32_t _size;
};

} // namespace cuda
} // namespace tinypt