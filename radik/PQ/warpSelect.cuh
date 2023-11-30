#ifndef TOPK_PQ_WARPSELECT_CUH_
#define TOPK_PQ_WARPSELECT_CUH_

#include "topK_PQ_config.h"
#include "utils.cuh"


template <typename Config>
struct WarpSelectKernel {

static_assert(Config::kBlockSize == 32, "invalid block size for warp select");

typedef typename Config::IdxType IdxType;
typedef typename Config::ValType ValType;

static __device__ __forceinline__ void
KernelFn(const ValType *valIn,
         const IdxType *idxIn,
         ValType *valOut,
         IdxType *idxOut,
         const int taskNum,
         const int K,
         const int n) {
    Pair<IdxType, ValType> threadQ[Config::kThreadQLength];
    Pair<IdxType, ValType> warpQ[Config::kWarpQCapacity];

    int taskId = blockIdx.x;
    const int warpQLength = (K + 31) / 32;
    const int lane = threadIdx.x;
    if (threadIdx.x == 0) {
    }
    for (; taskId < taskNum; taskId += gridDim.x) {
        // 1. each warp(block) deal with a task
        #pragma unroll
        for (int i = 0; i < Config::kThreadQLength; ++i) {
            threadQ[i].val = Config::kAscend ? getMax<ValType>() : getMin<ValType>();
        }
        #pragma unroll
        for (int i = 0; i < warpQLength; ++i) {
            warpQ[i].val = Config::kAscend ? getMax<ValType>() : getMin<ValType>();
        }

        int start = taskId * n;
        int idx = threadIdx.x;
        for (int i = 0; i < (n / 32); ++i, idx += 32) {
            Pair<IdxType, ValType> newData(
                    Config::kWithIdxIn ? idxIn[start + idx] : idx, valIn[start + idx]);
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
            Pair<IdxType, ValType> newData(
                    Config::kWithIdxIn ? idxIn[start + idx] : idx, valIn[start + idx]);
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
        start = taskId * K;
        idx = lane;
        for (int i = 0; i < warpQLength /*- 1*/; ++i, idx += 32) {
            valOut[idx + start] = warpQ[i].val;
            idxOut[idx + start] = warpQ[i].idx;
        }
        if (idx < K) {
            valOut[idx + start] = warpQ[warpQLength - 1].val;
            idxOut[idx + start] = warpQ[warpQLength - 1].idx;
        }
    }
}

};

template<typename Config>
__global__ void warpSelect(const typename Config::ValType *valIn,
                           const typename Config::IdxType *idxIn,
                           typename Config::ValType *valOut,
                           typename Config::IdxType *idxOut,
                           const int taskNum,
                           const int K,
                           const int n) {
    WarpSelectKernel<Config>::KernelFn(valIn, idxIn, valOut,
                                       idxOut, taskNum, K, n);
}

#endif  // TOPK_PQ_WARPSELECT_CUH_
