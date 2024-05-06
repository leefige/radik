#ifndef TEST_ERROR_CHECK_HPP_
#define TEST_ERROR_CHECK_HPP_

#include <exception>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#define RADIX_TOPK_ERROR_CHECK

inline void __cudaCheck(const cudaError_t &err, const char *file, const int line) {
#ifdef RADIX_TOPK_ERROR_CHECK
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "cuda call failed (" << err << ") at " << file << ":"
           << line << " : " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(ss.str());
    }
#endif
    return;
}

inline void __curandCheck(const curandStatus_t &err, const char *file, const int line) {
#ifdef RADIX_TOPK_ERROR_CHECK
    if (err != CURAND_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "curand call failed (" << err << ") at " << file << ":"
           << line << " : " << err << std::endl;
        throw std::runtime_error(ss.str());
    }
#endif
    return;
}

#define CHECK_CUDA(err) __cudaCheck((err), __FILE__, __LINE__)

#define CHECK_CURAND(err) __curandCheck((err), __FILE__, __LINE__)

#endif  // TEST_ERROR_CHECK_HPP_
