#ifndef TOPK_PQ_BLOCKSELECT_CUH_
#define TOPK_PQ_BLOCKSELECT_CUH_

#include "topK_PQ_config.h"
#include "utils.cuh"

template <typename Config>
struct BlockSelectKernel {

static_assert(Config::kBlockSize == 64 || Config::kBlockSize == 128,
              "invalid block size for block select.");

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

    __shared__ ValType valInter[Config::kBlockSize / 32][Config::kWarpQCapacity][32];
    __shared__ IdxType idxInter[Config::kBlockSize / 32][Config::kWarpQCapacity][32];

    int taskId = blockIdx.x;
    const int warpQLength = (K + 31) / 32;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    for (; taskId < taskNum; taskId += gridDim.x) {
        // 1. each warp deal with a part of task
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
        for (; idx - lane < n - Config::kBlockSize; idx += Config::kBlockSize) {
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

        // 2. merge every other warp's result into warp0
        #pragma unroll
        for (int i = 0; i < warpQLength; ++i) {
            valInter[warpId][i][lane] = warpQ[i].val;
            idxInter[warpId][i][lane] = warpQ[i].idx;
        }
        __syncthreads();

#define mergeShared(DSTWARPID) \
    for (int pos = threadIdx.x; pos < 32 * warpQLength; pos += Config::kBlockSize) { \
        if (Compare<ValType, Config::kThreadQLength>()(valR[pos], valL[32 * warpQLength - pos - 1])) { \
            valL[32 * warpQLength - pos - 1] = valR[pos]; \
            idxL[32 * warpQLength - pos - 1] = idxR[pos]; \
        } \
    } \
    __syncthreads(); \
    if (warpId == DSTWARPID) { \
        for (int i = step; i > 0; i /= 2) { \
            for (int j = 0; j < step; ++j) { \
                int idxH = (warpQLength - 1) - ((j % i) + (j / i) * i * 2); \
                int idxL = idxH - i; \
                if (idxH >= 0 && idxL >= 0) { \
                    if (Compare<ValType, Config::kAscend>()(valInter[warpId][idxL][lane], \
                         valInter[warpId][idxH][lane])) { \
                        ValType tmpVal = valInter[warpId][idxH][lane]; \
                        valInter[warpId][idxH][lane] = \
                                valInter[warpId][idxL][lane]; \
                        valInter[warpId][idxL][lane] = tmpVal; \
                        IdxType tmpIdx = idxInter[warpId][idxH][lane]; \
                        idxInter[warpId][idxH][lane] = \
                                idxInter[warpId][idxL][lane]; \
                        idxInter[warpId][idxL][lane] = tmpIdx; \
                    } \
                } \
            } \
        } \
    } \
    __syncthreads(); \
    for (int i = warpId; i < warpQLength; i += Config::kBlockSize / 32) { \
        Pair<IdxType, ValType> data(idxInter[DSTWARPID][i][lane], \
                                    valInter[DSTWARPID][i][lane]); \
        __syncwarp(0xffffffff); \
        warpMerge<IdxType, ValType, Config::kAscend>(data, lane); \
        idxInter[DSTWARPID][i][lane] = data.idx; \
        valInter[DSTWARPID][i][lane] = data.val; \
        __syncwarp(0xffffffff); \
    } \
    __syncthreads();

        int step = (warpQLength & (warpQLength - 1)) == 0 ? (warpQLength / 2) :
                                                            (1 << (31 - __clz(warpQLength)));
        if (Config::kBlockSize == 128) {
            // merge warp1's result into warp0's result
            ValType *valL = &valInter[0][0][0];
            IdxType *idxL = &idxInter[0][0][0];
            ValType *valR = &valInter[1][0][0];
            IdxType *idxR = &idxInter[1][0][0];
            mergeShared(0);
            // merge warp3's result into warp2's result
            valL = &valInter[2][0][0];
            idxL = &idxInter[2][0][0];
            valR = &valInter[3][0][0];
            idxR = &idxInter[3][0][0];
            mergeShared(2);
            // merge warp2's result into warp0's result
            valL = &valInter[0][0][0];
            idxL = &idxInter[0][0][0];
            valR = &valInter[2][0][0];
            idxR = &idxInter[2][0][0];
            mergeShared(0);
        }
        if (Config::kBlockSize == 64) {
            // merge warp1's result into warp0's result
            ValType *valL = &valInter[0][0][0];
            IdxType *idxL = &idxInter[0][0][0];
            ValType *valR = &valInter[1][0][0];
            IdxType *idxR = &idxInter[1][0][0];
            mergeShared(0);
        }
#undef mergeShared

        // 3. write back
        start = taskId * K;
        idx = threadIdx.x;
        ValType *valPtr = &valInter[0][0][0];
        IdxType *idxPtr = &idxInter[0][0][0];
        for (; idx < K; idx += Config::kBlockSize) {
            valOut[idx + start] = valPtr[idx];
            idxOut[idx + start] = idxPtr[idx];
        }
    }
}

};

template<typename Config>
__global__ void blockSelect(const typename Config::ValType *valIn,
                            const typename Config::IdxType *idxIn,
                            typename Config::ValType *valOut,
                            typename Config::IdxType *idxOut,
                            const int taskNum,
                            const int K,
                            const int n) {
    BlockSelectKernel<Config>::KernelFn(valIn, idxIn, valOut,
                                        idxOut, taskNum, K, n);
}

#endif  // TOPK_PQ_BLOCKSELECT_CUH_
