#ifndef TOPK_RADIXSELECT_ERROR_CHECK_HPP_
#define TOPK_RADIXSELECT_ERROR_CHECK_HPP_

#include <exception>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#define RADIX_TOPK_ERROR_CHECK

namespace radix_topk {

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

inline void __kernelCheck(const char *file, const int line) {
#ifdef RADIX_TOPK_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "kernel launch failed (" << err << ") at " << file << ":"
           << line << " : " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(ss.str());
    }
#endif
    return;
}

inline void __kernelCheckSync(const cudaStream_t &stream, const char *file, const int line) {
#ifdef RADIX_TOPK_ERROR_CHECK
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "kernel failed (" << err << ") at " << file << ":"
           << line << " : " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(ss.str());
    }
#endif
    return;
}

}   // namespace radix_topk

#define CUDA_CHECK(err) radix_topk::__cudaCheck((err), __FILE__, __LINE__)

#define KERNEL_CHECK() radix_topk::__kernelCheck(__FILE__, __LINE__)

#define KERNEL_CHECK_SYNC(stream) radix_topk::__kernelCheckSync((stream), __FILE__, __LINE__)

#endif  // TOPK_RADIXSELECT_ERROR_CHECK_HPP_
