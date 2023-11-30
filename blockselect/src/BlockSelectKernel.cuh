/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Select.cuh"
#include "FaissAssert.h"

namespace faiss {
namespace gpu {

template <
        typename K,
        typename IndexType,
        bool Dir,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void blockSelect(
        const K *in,
        K *outK,
        IndexType *outV,
        K initK,
        IndexType initV,
        int k,
        int taskNum,
        int taskLen) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<
            K,
            IndexType,
            Dir,
            Comparator<K>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;

    int i = threadIdx.x;
    const K* inStart = in + row * taskLen + i;

    // Whole warps must participate in the selection
    int limit = utils::roundDown(taskLen, kWarpSize);

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inStart, (IndexType)i);
        inStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < taskLen) {
        heap.addThreadQ(*inStart, (IndexType)i);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        outK[row * k + i] = smemK[i];
        outV[row * k + i] = smemV[i];
    }
}

void runBlockSelect(
        const float *in,
        float *outK,
        int *outV,
        bool dir,
        int k,
        int taskNum,
        int taskLen,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
