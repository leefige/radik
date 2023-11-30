/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "blockselect/BlockSelectImpl.cuh"
#include "FaissAssert.h"
#include "DeviceDefs.cuh"

namespace faiss {
namespace gpu {

// warp Q to thread Q:
// 32, 2
// 64, 3
// 128, 3
// 256, 4
// 512, 8
// 1024, 8
// 2048, 8

BLOCK_SELECT_DECL(float, true, 32);
BLOCK_SELECT_DECL(float, true, 64);
BLOCK_SELECT_DECL(float, true, 128);
BLOCK_SELECT_DECL(float, true, 256);
BLOCK_SELECT_DECL(float, true, 512);
BLOCK_SELECT_DECL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(float, true, 2048);
#endif

BLOCK_SELECT_DECL(float, false, 32);
BLOCK_SELECT_DECL(float, false, 64);
BLOCK_SELECT_DECL(float, false, 128);
BLOCK_SELECT_DECL(float, false, 256);
BLOCK_SELECT_DECL(float, false, 512);
BLOCK_SELECT_DECL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_DECL(float, false, 2048);
#endif

void runBlockSelect(
        const float *in,
        float *outK,
        int *outV,
        bool dir,
        int k,
        int taskNum,
        int taskLen,
        cudaStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (dir) {
        if (k <= 32) {
            BLOCK_SELECT_CALL(float, true, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_CALL(float, true, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_CALL(float, true, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_CALL(float, true, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_CALL(float, true, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_CALL(float, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_CALL(float, true, 2048);
#endif
        }
    } else {
        if (k <= 32) {
            BLOCK_SELECT_CALL(float, false, 32);
        } else if (k <= 64) {
            BLOCK_SELECT_CALL(float, false, 64);
        } else if (k <= 128) {
            BLOCK_SELECT_CALL(float, false, 128);
        } else if (k <= 256) {
            BLOCK_SELECT_CALL(float, false, 256);
        } else if (k <= 512) {
            BLOCK_SELECT_CALL(float, false, 512);
        } else if (k <= 1024) {
            BLOCK_SELECT_CALL(float, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            BLOCK_SELECT_CALL(float, false, 2048);
#endif
        }
    }
}

} // namespace gpu
} // namespace faiss
