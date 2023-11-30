#ifndef TOPK_PQ_GRIDSELECT_CUH_
#define TOPK_PQ_GRIDSELECT_CUH_

#include "topK_PQ_config.h"
#include "utils.cuh"


template <typename Config>
struct GridSelectKernel {

static_assert(Config::kBlockSize == 32, "invalid block size for grid select");

typedef typename Config::IdxType IdxType;
typedef typename Config::ValType ValType;


static __device__ __forceinline__ void
KernelFn(const ValType *valIn,
         const IdxType *idxIn,
         ValType *valOut,
         IdxType *idxOut,
         const int K,
         const int n) {
    Pair<IdxType, ValType> threadQ[Config::kThreadQLength];
    Pair<IdxType, ValType> warpQ[Config::kWarpQCapacity];

    const int warpQLength = (K + 31) / 32;
    // 1. each block(warp) deal with a part of task
    #pragma unroll
    for (int i = 0; i < Config::kThreadQLength; ++i) {
        threadQ[i].val = Config::kAscend ? getMax<ValType>() : getMin<ValType>();
    }
    #pragma unroll
    for (int i = 0; i < warpQLength; ++i) {
        warpQ[i].val = Config::kAscend ? getMax<ValType>() : getMin<ValType>();
    }

    const int lane = threadIdx.x;
    const int stepSize = 32 * gridDim.x;
    int start = blockIdx.y * n;
    int idx = 32 * blockIdx.x + threadIdx.x;
    for (; idx - lane < n - stepSize; idx += stepSize) {
        Pair<IdxType, ValType> newData(Config::kWithIdxIn ? idxIn[start + idx] : idx, valIn[start + idx]);
        pushNew<IdxType, ValType, Config::kThreadQLength, Config::kAscend>(newData, threadQ);
        ValType warpQTop = __shfl_val_sync<ValType>(0xffffffff, warpQ[warpQLength - 1].val, 31);
        bool flag = Compare<ValType, Config::kAscend>()(threadQ[Config::kThreadQLength - 1].val, warpQTop);
        if (__any_sync(0xffffffff, flag)) {
            sort<IdxType, ValType, Config::kAscend>(threadQ,
                                                    Config::kThreadQLength,
                                                    lane);
            merge<IdxType, ValType, Config::kAscend>(warpQ,
                                                     threadQ,
                                                     warpQLength,
                                                     Config::kThreadQLength,
                                                     lane);
        }
    }
    if (idx < n) {
        Pair<IdxType, ValType> newData(Config::kWithIdxIn ? idxIn[start + idx] : idx, valIn[start + idx]);
        pushNew<IdxType, ValType, Config::kThreadQLength, Config::kAscend>(newData, threadQ);
    }
    sort<IdxType, ValType, Config::kAscend>(threadQ,
                                            Config::kThreadQLength,
                                            lane);
    merge<IdxType, ValType, Config::kAscend>(warpQ,
                                             threadQ,
                                             warpQLength,
                                             Config::kThreadQLength,
                                             lane);

    // 2. write back
    start = blockIdx.y * gridDim.x * K + blockIdx.x * K;
    idx = lane;
    for (int i = 0; i < warpQLength - 1; ++i, idx += 32) {
        valOut[idx + start] = warpQ[i].val;
        idxOut[idx + start] = warpQ[i].idx;
    }
    if (idx < K) {
        valOut[idx + start] = warpQ[warpQLength - 1].val;
        idxOut[idx + start] = warpQ[warpQLength - 1].idx;
    }
}

};

template<typename Config>
__global__ void gridSelect(const typename Config::ValType *valIn,
                           const typename Config::IdxType *idxIn,
                           typename Config::ValType *valOut,
                           typename Config::IdxType *idxOut,
                           const int K,
                           const int n) {
    GridSelectKernel<Config>::KernelFn(valIn, idxIn, valOut,
                                       idxOut, K, n);
}

#endif  // TOPK_PQ_GRIDSELECT_CUH_
