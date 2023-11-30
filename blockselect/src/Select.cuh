/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Comparators.cuh"
#include "DeviceDefs.cuh"
#include "StaticUtils.h"
#include "MergeNetworkBlock.cuh"
#include "MergeNetworkWarp.cuh"
#include "PtxUtils.cuh"

namespace faiss {
namespace gpu {

// Specialization for block-wide monotonic merges producing a merge sort
// since what we really want is a constexpr loop expansion
template <
        int NumWarps,
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge {};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<1, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        // no merge required; single warp
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<2, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<4, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp>(sharedK, sharedV);
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 4),
                NumWarpQ * 2,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int NumWarpQ,
        bool Dir,
        typename Comp>
struct FinalBlockMerge<8, NumThreads, K, V, NumWarpQ, Dir, Comp> {
    static inline __device__ void merge(K* sharedK, V* sharedV) {
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 2),
                NumWarpQ,
                !Dir,
                Comp>(sharedK, sharedV);
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 4),
                NumWarpQ * 2,
                !Dir,
                Comp>(sharedK, sharedV);
        // Final merge doesn't need to fully merge the second list
        blockMerge<
                NumThreads,
                K,
                V,
                NumThreads / (kWarpSize * 8),
                NumWarpQ * 4,
                !Dir,
                Comp,
                false>(sharedK, sharedV);
    }
};

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
template <
        typename K,
        typename V,
        bool Dir,
        typename Comp,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
struct BlockSelect {
    static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
    static constexpr int kTotalWarpSortSize = NumWarpQ;

    __device__ inline BlockSelect(
            K initKVal,
            V initVVal,
            K* smemK,
            V* smemV,
            int k)
            : initK(initKVal),
              initV(initVVal),
              numVals(0),
              warpKTop(initKVal),
              sharedK(smemK),
              sharedV(smemV),
              kMinus1(k - 1) {
        static_assert(
                utils::isPowerOf2(ThreadsPerBlock),
                "threads must be a power-of-2");
        static_assert(
                utils::isPowerOf2(NumWarpQ), "warp queue must be power-of-2");

        // Fill the per-thread queue keys with the default value
#pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }

        int laneId = getLaneId();
        int warpId = threadIdx.x / kWarpSize;
        warpK = sharedK + warpId * kTotalWarpSortSize;
        warpV = sharedV + warpId * kTotalWarpSortSize;

        // Fill warp queue (only the actual queue space is fine, not where
        // we write the per-thread queues for merging)
        for (int i = laneId; i < NumWarpQ; i += kWarpSize) {
            warpK[i] = initK;
            warpV[i] = initV;
        }

        warpFence();
    }

    __device__ inline void addThreadQ(K k, V v) {
        if (Dir ? Comp::gt(k, warpKTop) : Comp::lt(k, warpKTop)) {
            // Rotate right
#pragma unroll
            for (int i = NumThreadQ - 1; i > 0; --i) {
                threadK[i] = threadK[i - 1];
                threadV[i] = threadV[i - 1];
            }

            threadK[0] = k;
            threadV[0] = v;
            ++numVals;
        }
    }

    __device__ inline void checkThreadQ() {
        bool needSort = (numVals == NumThreadQ);

#if CUDA_VERSION >= 9000
        needSort = __any_sync(0xffffffff, needSort);
#else
        needSort = __any(needSort);
#endif

        if (!needSort) {
            // no lanes have triggered a sort
            return;
        }

        // This has a trailing warpFence
        mergeWarpQ();

        // Any top-k elements have been merged into the warp queue; we're
        // free to reset the thread queues
        numVals = 0;

#pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }

        // We have to beat at least this element
        warpKTop = warpK[kMinus1];

        warpFence();
    }

    /// This function handles sorting and merging together the
    /// per-thread queues with the warp-wide queue, creating a sorted
    /// list across both
    __device__ inline void mergeWarpQ() {
        int laneId = getLaneId();

        // Sort all of the per-thread queues
        warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

        constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;
        K warpKRegisters[kNumWarpQRegisters];
        V warpVRegisters[kNumWarpQRegisters];

#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpKRegisters[i] = warpK[i * kWarpSize + laneId];
            warpVRegisters[i] = warpV[i * kWarpSize + laneId];
        }

        warpFence();

        // The warp queue is already sorted, and now that we've sorted the
        // per-thread queue, merge both sorted lists together, producing
        // one sorted list
        warpMergeAnyRegisters<
                K,
                V,
                kNumWarpQRegisters,
                NumThreadQ,
                !Dir,
                Comp,
                false>(warpKRegisters, warpVRegisters, threadK, threadV);

        // Write back out the warp queue
#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i * kWarpSize + laneId] = warpKRegisters[i];
            warpV[i * kWarpSize + laneId] = warpVRegisters[i];
        }

        warpFence();
    }

    /// WARNING: all threads in a warp must participate in this.
    /// Otherwise, you must call the constituent parts separately.
    __device__ inline void add(K k, V v) {
        addThreadQ(k, v);
        checkThreadQ();
    }

    __device__ inline void reduce() {
        // Have all warps dump and merge their queues; this will produce
        // the final per-warp results
        mergeWarpQ();

        // block-wide dep; thus far, all warps have been completely
        // independent
        __syncthreads();

        // All warp queues are contiguous in smem.
        // Now, we have kNumWarps lists of NumWarpQ elements.
        // This is a power of 2.
        FinalBlockMerge<kNumWarps, ThreadsPerBlock, K, V, NumWarpQ, Dir, Comp>::
                merge(sharedK, sharedV);

        // The block-wide merge has a trailing syncthreads
    }

    // Default element key
    const K initK;

    // Default element value
    const V initV;

    // Number of valid elements in our thread queue
    int numVals;

    // The k-th highest (Dir) or lowest (!Dir) element
    K warpKTop;

    // Thread queue values
    K threadK[NumThreadQ];
    V threadV[NumThreadQ];

    // Queues for all warps
    K* sharedK;
    V* sharedV;

    // Our warp's queue (points into sharedK/sharedV)
    // warpK[0] is highest (Dir) or lowest (!Dir)
    K* warpK;
    V* warpV;

    // This is a cached k-1 value
    int kMinus1;
};


} // namespace gpu
} // namespace faiss
