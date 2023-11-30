#ifndef TOPK_ABLATION_NAIVE_CUH_
#define TOPK_ABLATION_NAIVE_CUH_

#include "../RadixSelect/utils.cuh"

namespace radix_topk {

//===================================
// type- & scaling-aware kernels
//===================================
template <int BLOCK, int LEFT, int RIGHT, int PACKSIZE, bool WITHSCALE,
          bool LARGEST, typename T>
__global__ void __launch_bounds__(1024)
    countBinEx_naive(const T *dataIn, const int *taskOffsetPtr, int *histPtr,
               const int taskNum) {
    using InVec = VT<T, PACKSIZE>;
    using CompT = typename ComputeT<T>::type;

    constexpr int histLen = 1 << (8 * sizeof(CompT) - RIGHT);

    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int stepSize = BLOCK * PACKSIZE * gridDim.x;

    for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
        // lambda to update histogram
        auto updateHist = [&](const CompT &value) {
#ifndef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
            // for potential NaN, just return
            if (isNaN(value)) {
                return;
            }
#endif
            int binId = getBinId<LEFT, RIGHT>(value);
            atomicAdd(&histPtr[taskId * histLen + binId], 1);
            return;
        };

        int offset = taskOffsetPtr[taskId];
        const int pad = offset & (PACKSIZE - 1);
        offset -= pad;

        // scaling factor
        const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);

        const int taskLen = taskOffsetPtr[taskId + 1] - offset;
        const int step = taskLen / stepSize;

        if (step > 0) {
            int idx = tid;
            const InVec *readPtr =
                reinterpret_cast<const InVec *>(dataIn + offset);

            // first loop
            auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
            if (tid == 0) {
                #pragma unroll
                for (int j = pad; j < PACKSIZE; ++j) {
                    updateHist(val[j]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    updateHist(val[j]);
                }
            }
            idx += BLOCK * gridDim.x;

            // main loop
            for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    updateHist(val[j]);
                }
            }

            // tail
            for (idx = tid + step * stepSize; idx < taskLen;
                 idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
                updateHist(val);
            }
        } else {
            for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
                updateHist(val);
            }
        }
        __syncthreads();
    }
    return;
}

template <int BLOCK, int LEFT, int RIGHT, int PACKSIZE, bool WITHSCALE,
          bool LARGEST, typename T>
__global__ void __launch_bounds__(256)
    selectCandidateEx_naive(const T *dataIn, typename ComputeT<T>::type *dataOut,
                      int *globalCount, const int *binId,
                      const int *taskOffsetPtr, const int stride,
                      const int taskNum) {
    using InVec = VT<T, PACKSIZE>;
    using CompT = typename ComputeT<T>::type;

    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int stepSize = gridDim.x * BLOCK * PACKSIZE;

    for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {
        // check whether hit and stage to cache if necessary
        auto stageIfHit = [&](const CompT &val, const int &mask) {
            if (mask == getBinId<LEFT, RIGHT>(val)) {
                int pos = atomicAdd(globalCount + taskId, 1);
                dataOut[taskId * stride + pos] = val;
            }
            return;
        };

        int offset = taskOffsetPtr[taskId];
        const int pad = offset & (PACKSIZE - 1);
        offset -= pad;

        // scaling factor
        const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);
        const int mask = binId[taskId];

        const int taskLen = taskOffsetPtr[taskId + 1] - offset;
        const int step = taskLen / stepSize;

        if (step > 0) {
            int idx = tid;
            const InVec *readPtr =
                reinterpret_cast<const InVec *>(dataIn + offset);

            // first loop
            auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
            if (tid == 0) {
                #pragma unroll
                for (int j = pad; j < PACKSIZE; ++j) {
                    stageIfHit(val[j], mask);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    stageIfHit(val[j], mask);
                }
            }
            idx += BLOCK * gridDim.x;

            // main loop
            for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(readPtr, idx, scaler);
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    stageIfHit(val[j], mask);
                }
            }

            // tail
            for (idx = step * stepSize + tid; idx < taskLen;
                 idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
                stageIfHit(val, mask);
            }
        } else {
            for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
                auto val = loadScaling<WITHSCALE, LARGEST>(dataIn, offset + idx, scaler);
                stageIfHit(val, mask);
            }
        }
    }
    return;
}

#define RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j)       \
    do {                                                         \
        if (LARGEST && (val_s)[(j)] > kThEle) {                  \
            stagePair((val)[(j)], getPackIndex((idx), (j)));     \
        } else if (!LARGEST && (val_s)[(j)] < kThEle) {          \
            stagePair((val)[(j)], getPackIndex((idx), (j)));     \
        }                                                        \
        if ((val_s)[(j)] == kThEle) {                            \
            if (atomicSub(boundaryCount + taskId, 1) > 0) {      \
                stagePair((val)[(j)], getPackIndex((idx), (j))); \
            }                                                    \
        }                                                        \
    } while (0)

#define RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx)       \
    do {                                                    \
        if (LARGEST && (val_s) > kThEle) {                  \
            stagePair((val), getItemIndex((idx)));          \
        } else if (!LARGEST && (val_s) < kThEle) {          \
            stagePair((val), getItemIndex((idx)));          \
        }                                                   \
        if ((val_s) == kThEle) {                            \
            if (atomicSub(boundaryCount + taskId, 1) > 0) { \
                stagePair((val), getItemIndex((idx)));      \
            }                                               \
        }                                                   \
    } while (0)

template <bool LARGEST, int BLOCK, int PACKSIZE, bool WITHSCALE,
          bool WITHIDXIN = 0, typename IdxType, typename ValType>
__global__ void __launch_bounds__(256)
    filter_naive(const ValType *dataIn, const IdxType *idxIn,
                   const typename ComputeT<ValType>::type *kThElePtr,
                   ValType *valOut, IdxType *idxOut, int *globalCount,
                   int *boundaryCount, const int *taskOffsetPtr,
                   const int stride, const int K, const int taskNum) {
    using InVec = VT<ValType, PACKSIZE>;
    using CompT = typename ComputeT<ValType>::type;

    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int stepSize = PACKSIZE * BLOCK * gridDim.x;

    for (int taskId = blockIdx.y; taskId < taskNum; taskId += gridDim.y) {

        int offset = taskOffsetPtr[taskId];
        const int originalLen = taskOffsetPtr[taskId + 1] - offset;

        // padding if misaligned
        const int pad = offset & (PACKSIZE - 1);
        offset -= pad;
        const int taskLen = originalLen + pad;
        const int step = taskLen / stepSize;

        auto getPackIndex = [&](const int &idx, const int &j) {
            return WITHIDXIN ? idxIn[offset + idx * PACKSIZE + j]
                             : static_cast<IdxType>(idx * PACKSIZE + j - pad);
        };

        auto getItemIndex = [&](const int &idx) {
            return WITHIDXIN ? idxIn[offset + idx]
                             : static_cast<IdxType>(idx - pad);
        };

        auto stagePair = [&](const auto &value, const IdxType &index) {
            int pos = atomicAdd(globalCount + taskId, 1);
            valOut[taskId * K + pos] = static_cast<ValType>(value);
            idxOut[taskId * K + pos] = index;
            return;
        };

        /*
         * N <= K, just copy all
         */
        if (originalLen <= K) {
            if (step > 0) {
                int idx = tid;
                const InVec *readPtr =
                    reinterpret_cast<const InVec *>(dataIn + offset);

                // first loop
                InVec val = readPtr[idx];
                if (tid == 0) {
                    #pragma unroll
                    for (int j = pad; j < PACKSIZE; ++j) {
                        stagePair(val[j], getPackIndex(idx, j));
                    }
                } else {  // tid != 0
                    #pragma unroll
                    for (int j = 0; j < PACKSIZE; ++j) {
                        stagePair(val[j], getPackIndex(idx, j));
                    }
                }
                idx += BLOCK * gridDim.x;

                // main loop
                for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
                    InVec val = readPtr[idx];
                    #pragma unroll
                    for (int j = 0; j < PACKSIZE; ++j) {
                        stagePair(val[j], getPackIndex(idx, j));
                    }
                }

                // tail
                for (idx = tid + step * stepSize; idx < taskLen;
                     idx += BLOCK * gridDim.x) {
                    ValType val = dataIn[offset + idx];
                    stagePair(val, getItemIndex(idx));
                }
            } else {  // step == 0
                for (int idx = tid + pad; idx < taskLen;
                     idx += BLOCK * gridDim.x) {
                    ValType val = dataIn[offset + idx];
                    stagePair(val, getItemIndex(idx));
                }
            }
            continue;
        }

        /*
         * otherwise, filter by k-th element
         */
        // scaling factor
        const CompT scaler = sampleScaler<WITHSCALE>(dataIn, offset + pad);
        const CompT kThEle = *(kThElePtr + taskId * stride);

        if (step > 0) {
            int idx = tid;
            const InVec *readPtr =
                reinterpret_cast<const InVec *>(dataIn + offset);

            // first loop
            InVec val = readPtr[idx];
            auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);

            if (tid == 0) {
                #pragma unroll
                for (int j = pad; j < PACKSIZE; ++j) {
                    RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
                }
            } else {  // tid != 0
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
                }
            }
            idx += BLOCK * gridDim.x;

            // main loop
            for (int i = 1; i < step; i += 1, idx += BLOCK * gridDim.x) {
                InVec val = readPtr[idx];
                auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
                #pragma unroll
                for (int j = 0; j < PACKSIZE; ++j) {
                    RADIX_TOPK_FILTER_STAGE_PACKED(val, val_s, idx, j);
                }
            }

            // tail
            for (idx = tid + step * stepSize; idx < taskLen;
                 idx += BLOCK * gridDim.x) {
                ValType val = dataIn[offset + idx];
                auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
                RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
            }
        } else {  // step == 0
            for (int idx = tid + pad; idx < taskLen; idx += BLOCK * gridDim.x) {
                ValType val = dataIn[offset + idx];
                auto val_s = scaling<WITHSCALE, LARGEST>(val, scaler);
                RADIX_TOPK_FILTER_STAGE_ITEM(val, val_s, idx);
            }
        }
    }
    return;
}

#undef RADIX_TOPK_FILTER_STAGE_PACKED
#undef RADIX_TOPK_FILTER_STAGE_ITEM

}  // namespace radix_topk

#endif  // TOPK_ABLATION_NAIVE_CUH_
