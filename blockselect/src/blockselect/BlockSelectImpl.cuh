/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../BlockSelectKernel.cuh"
#include "../Limits.cuh"

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                      \
    do {                                    \
        auto err__ = (X);                   \
        FAISS_ASSERT_FMT(                   \
                err__ == cudaSuccess,       \
                "CUDA error %d %s",         \
                (int)err__,                 \
                cudaGetErrorString(err__)); \
    } while (0)

/// Wrapper to synchronously probe for CUDA errors
// #define FAISS_GPU_SYNC_ERROR 1

#ifdef FAISS_GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                     \
    do {                                      \
        CUDA_VERIFY(cudaDeviceSynchronize()); \
    } while (0)
#else
#define CUDA_TEST_ERROR()                \
    do {                                 \
        CUDA_VERIFY(cudaGetLastError()); \
    } while (0)
#endif

#define BLOCK_SELECT_DECL(TYPE, DIR, WARP_Q)                     \
    extern void runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(     \
            const TYPE *in,                                      \
            TYPE *outK,                                          \
            int *outV,                                           \
            bool dir,                                            \
            int k,                                               \
            int taskNum,                                         \
            int taskLen,                                         \
            cudaStream_t stream);


#define BLOCK_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                         \
    void runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(                          \
            const TYPE *in,                                                    \
            TYPE *outK,                                                        \
            int *outV,                                                         \
            bool dir,                                                          \
            int k,                                                             \
            int taskNum,                                                       \
            int taskLen,                                                       \
            cudaStream_t stream) {                                             \
                                                                               \
        auto grid = dim3(taskNum);                                             \
                                                                               \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;    \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        FAISS_ASSERT(k <= WARP_Q);                                             \
        FAISS_ASSERT(dir == DIR);                                              \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelect<TYPE, int, DIR, WARP_Q, THREAD_Q, kBlockSelectNumThreads>  \
                <<<grid, block, 0, stream>>>(                                  \
                    in, outK, outV, kInit, vInit, k, taskNum, taskLen);        \
        CUDA_TEST_ERROR();                                                     \
    }

#define BLOCK_SELECT_CALL(TYPE, DIR, WARP_Q) \
    runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(in, outK, outV, dir, k, taskNum, taskLen, stream)

