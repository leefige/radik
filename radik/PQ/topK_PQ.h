#ifndef TOPK_PQ_TOPK_PQ_H_
#define TOPK_PQ_TOPK_PQ_H_

#include "warpSelect.cuh"
#include "gridSelect.cuh"
#include "blockSelect.cuh"

// default kernel launch config
static constexpr int MAX_GRID_SIZE = 1280;

// hyper-param for grid select: #items per block
static constexpr int N_ITEMS_PER_BLOCK = 4096;

template<typename IdxType, typename ValType, bool withIdxIn = 0, bool isSubByte = 0>
void topKPQWarpSelect(const ValType *valIn,
                      const IdxType *idxIn,
                      ValType *valOut,
                      IdxType *idxOut,
                      const int taskNum,
                      const int K,
                      const int n,
                      const bool largest,
                      cudaStream_t stream) {
    if (K > 1024 || K >= n) exit(-1);

    const int gridSize = std::min(MAX_GRID_SIZE, taskNum);
#define UNFOLD_LARGEST(largest) \
    if (K <= 32) { \
        typedef TopKPQConfig<IdxType, ValType, 1, 2, 32, withIdxIn, largest, isSubByte> Config; \
        warpSelect<Config><<<gridSize, 32, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else if (K <= 128) { \
        typedef TopKPQConfig<IdxType, ValType, 4, 3, 32, withIdxIn, largest, isSubByte> Config; \
        warpSelect<Config><<<gridSize, 32, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else if (K <= 256) { \
        typedef TopKPQConfig<IdxType, ValType, 8, 4, 32, withIdxIn, largest, isSubByte> Config; \
        warpSelect<Config><<<gridSize, 32, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else { \
        typedef TopKPQConfig<IdxType, ValType, 32, 8, 32, withIdxIn, largest, isSubByte> Config; \
        warpSelect<Config><<<gridSize, 32, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    }

    if (largest) {
        UNFOLD_LARGEST(1);
    } else {
        UNFOLD_LARGEST(0);
    }

#undef UNFOLD_LARGEST

    return;
}

template<typename IdxType, typename ValType, bool withIdxIn = 0, bool isSubByte = 0>
void topKPQBlockSelect(const ValType *valIn,
                       const IdxType *idxIn,
                       ValType *valOut,
                       IdxType *idxOut,
                       const int taskNum,
                       const int K,
                       const int n,
                       const bool largest,
                       cudaStream_t stream) {
    if (K > 1024 || K >= n) exit(-1);

    const int gridSize = std::min(MAX_GRID_SIZE, taskNum);
#define UNFOLD_LARGEST(largest) \
    if (K <= 32) { \
        typedef TopKPQConfig<IdxType, ValType, 1, 2, 128, withIdxIn, largest, isSubByte> Config; \
        blockSelect<Config><<<gridSize, 128, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else if (K <= 128) { \
        typedef TopKPQConfig<IdxType, ValType, 4, 3, 128, withIdxIn, largest, isSubByte> Config; \
        blockSelect<Config><<<gridSize, 128, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else if (K <= 256) { \
        typedef TopKPQConfig<IdxType, ValType, 8, 4, 128, withIdxIn, largest, isSubByte> Config; \
        blockSelect<Config><<<gridSize, 128, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    } else { \
        typedef TopKPQConfig<IdxType, ValType, 32, 8, 128, withIdxIn, largest, isSubByte> Config; \
        blockSelect<Config><<<gridSize, 128, 0, stream>>>(valIn, idxIn, valOut, idxOut, taskNum, K, n); \
    }

    if (largest) {
        UNFOLD_LARGEST(1);
    } else {
        UNFOLD_LARGEST(0);
    }

#undef UNFOLD_LARGEST

    return;
}

template<typename IdxType, typename ValType>
void getGridSelectWorkSpaceSize(const int taskNum,
                                const int K,
                                const int n,
                                size_t *sizeInBytes) {
    *sizeInBytes = 2 * (sizeof(IdxType) + sizeof(ValType)) * taskNum * (n / 1024) * K;
}

template<typename IdxType, typename ValType, bool withIdxIn = 0, bool isSubByte = 0>
void topKPQGridSelect(const ValType *valIn,
                      const IdxType *idxIn,
                      ValType *valOut,
                      IdxType *idxOut,
                      void *workSpace,
                      const int taskNum,
                      const int K,
                      const int n,
                      const bool largest,
                      cudaStream_t stream) {
    IdxType *idxBuffer[2];
    ValType *valBuffer[2];

    int blockNumPerTask = n / N_ITEMS_PER_BLOCK;
    int newTaskLen = blockNumPerTask * K;
    valBuffer[0] = static_cast<ValType *>(workSpace);
    valBuffer[1] = valBuffer[0] + (taskNum * newTaskLen);
    idxBuffer[0] = reinterpret_cast<IdxType *>(valBuffer[1] + (taskNum * newTaskLen));
    idxBuffer[1] = idxBuffer[0] + (taskNum * newTaskLen);

    int flag = 0;
#define UNFOLD_CONDITION(warpQCapacity, threadQLength, largest) \
    typedef TopKPQConfig<IdxType, ValType, warpQCapacity, threadQLength, 32, false, largest, isSubByte> Config0; \
    typedef TopKPQConfig<IdxType, ValType, warpQCapacity, threadQLength, 32, true, largest, isSubByte> Config1; \
    gridSelect<Config0><<<dim3(blockNumPerTask, taskNum), 32, 0, stream>>>(valIn, idxIn, valBuffer[0], idxBuffer[0], K, n); \
    while (newTaskLen > N_ITEMS_PER_BLOCK) { \
        blockNumPerTask = newTaskLen / N_ITEMS_PER_BLOCK; \
        gridSelect<Config1><<<dim3(blockNumPerTask, taskNum), 32, 0, stream>>>(valBuffer[flag], idxBuffer[flag], valBuffer[flag^1], idxBuffer[flag^1], K, newTaskLen); \
        newTaskLen = blockNumPerTask * K; \
        flag ^= 1; \
    }

    if (K <= 32) {
        if (largest) {
            UNFOLD_CONDITION(1, 2, 1);
        } else {
            UNFOLD_CONDITION(1, 2, 0);
        }
    } else if (K <= 128) {
        if (largest) {
            UNFOLD_CONDITION(4, 3, 1);
        } else {
            UNFOLD_CONDITION(4, 3, 0);
        }
    } else if (K <= 256) {
        if (largest) {
            UNFOLD_CONDITION(8, 4, 1);
        } else {
            UNFOLD_CONDITION(8, 4, 0);
        }
    } else {
        if (largest) {
            UNFOLD_CONDITION(32, 8, 1);
        } else {
            UNFOLD_CONDITION(32, 8, 0);
        }
    }

#undef UNFOLD_CONDITION

    if (newTaskLen > 1024) {
        topKPQBlockSelect<IdxType, ValType, true, isSubByte>(valBuffer[flag], idxBuffer[flag], valOut, idxOut, taskNum, K, newTaskLen, largest, stream);
    } else {
        topKPQWarpSelect<IdxType, ValType, true, isSubByte>(valBuffer[flag], idxBuffer[flag], valOut, idxOut, taskNum, K, newTaskLen, largest, stream);
    }

    return;
}

#endif  // TOPK_PQ_TOPK_PQ_H_
