/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>

namespace faiss {
namespace gpu {

template <typename T>
struct Limits {};

// Unfortunately we can't use constexpr because there is no
// constexpr constructor for half
// FIXME: faiss CPU uses +/-FLT_MAX instead of +/-infinity
constexpr float kFloatMax = std::numeric_limits<float>::max();
constexpr float kFloatMin = std::numeric_limits<float>::lowest();

template <>
struct Limits<float> {
    static __device__ __host__ inline float getMin() {
        return kFloatMin;
    }
    static __device__ __host__ inline float getMax() {
        return kFloatMax;
    }
};

} // namespace gpu
} // namespace faiss
