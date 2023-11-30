#ifndef TOPK_PQ_UTILS_CUH_
#define TOPK_PQ_UTILS_CUH_

#include <cstdint>
#include <cuda_fp16.h>

template<typename IdxType, typename ValType>
class Pair{
 public:
    IdxType idx;
    ValType val;

    __device__ Pair() {
        idx = 0;
        val = 0;
    }
    __device__ Pair(IdxType first, ValType second) {
        idx = first;
        val = second;
    }

    __device__ __forceinline__ void set(const Pair<IdxType, ValType> &p) {
        idx = p.idx;
        val = p.val;
    }
};

template<typename T, int N>
struct alignas(sizeof(T) * N) VT {

T data[N];

__device__ __forceinline__ T operator[] (int idx) {
    return data[idx];
}

};

template<typename T>
__device__ __forceinline__
T getMax() {
    return 0x7fffffff;
}

template<>
__device__ __forceinline__
__half getMax() {
    return 0x7fff;
}

template<>
__device__ __forceinline__
int8_t getMax() {
    return 127;
}

template<typename T>
__device__ __forceinline__
T getMin() {
    return 0xffffffff;
}

template<>
__device__ __forceinline__
__half getMin() {
    return 0xffff;
}

template<>
__device__ __forceinline__
int8_t getMin() {
    return -128;
}

template<typename T, bool ASCEND>
struct Compare {
__device__ __forceinline__
bool operator() (const T &pre, const T &post) {
    if (ASCEND) {
        return pre < post;
    } else {
        return pre > post;
    }
}
};

template<typename IdxType, typename ValType, bool ASCEND>
struct CAS {
__device__ __forceinline__
void operator() (Pair<IdxType, ValType> &a, Pair<IdxType, ValType> &b) {
    if (Compare<ValType, ASCEND>()(b.val, a.val)) {
        ValType valTmp = a.val;
        a.val = b.val;
        b.val = valTmp;
        IdxType idxTmp = a.idx;
        a.idx = b.idx;
        b.idx = idxTmp;
    }
}
};

template<typename T>
__device__ __forceinline__
T  __shfl_val_sync(int mask, T val, int srcLane) {
    return __shfl_sync(mask, val, srcLane);
}

template<>
__device__ __forceinline__
int16_t __shfl_val_sync(int mask, int16_t val, int srcLane) {
    int32_t reg0 = static_cast<int32_t>(val);
    int16_t reg1 = static_cast<int16_t>(__shfl_sync(mask, reg0, srcLane));
    return reg1;
}

template<>
__device__ __forceinline__
int8_t __shfl_val_sync(int mask, int8_t val, int srcLane) {
    int32_t reg0 = static_cast<int32_t>(val);
    int8_t reg1 = static_cast<int8_t>(__shfl_sync(mask, reg0, srcLane));
    return reg1;
}

template<typename IdxType, typename ValType, bool ASCEND>
__device__ __forceinline__ void warpMerge(Pair<IdxType, ValType> &data, int lane) {
    bool flag = lane & 16;
    int srcLane = flag ? lane - 16 : lane + 16;
    ValType new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    IdxType new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    flag = lane & 8;
    srcLane = flag ? lane - 8 : lane + 8;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    flag = lane & 4;
    srcLane = flag ? lane - 4 : lane + 4;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    flag = lane & 2;
    srcLane = flag ? lane - 2 : lane + 2;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    flag = lane & 1;
    srcLane = flag ? lane - 1 : lane + 1;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
}

template<typename IdxType, typename ValType, bool ASCEND>
__device__ __forceinline__
void merge(Pair<IdxType, ValType> *LA, Pair<IdxType, ValType> *RA, int l, int r, int lane) {
    // step1. merge LA && RA
    int common = l > r ? r : l;
    #pragma unroll
    for (int i = 0; i < common; ++i) {
        ValType tmpR_val = __shfl_val_sync(0xffffffff, RA[i].val, 31 - lane);
        IdxType tmpR_idx = __shfl_val_sync(0xffffffff, RA[i].idx, 31 - lane);
        ValType tmpL_val = __shfl_val_sync(0xffffffff, LA[l - 1 - i].val, 31 - lane);
        IdxType tmpL_idx = __shfl_val_sync(0xffffffff, LA[l - 1 - i].idx, 31 - lane);
        if (Compare<ValType, ASCEND>()(tmpR_val, LA[l - 1 - i].val)) {
            LA[l - 1 - i].val = tmpR_val;
            LA[l - 1 - i].idx = tmpR_idx;
        }
        if (Compare<ValType, ASCEND>()(RA[i].val, tmpL_val)) {
            RA[i].val = tmpL_val;
            RA[i].idx = tmpL_idx;
        }
    }

    // step2. sort LA
    //int L_step = (l & (l - 1)) == 0 ? 31 - __clz(l) : 32 - __clz(l);
    int L_step = (l & (l - 1)) == 0 ? (l / 2) : (1 << (31 - __clz(l)));
    // stepsize >= 32
    #pragma unroll
    for (int i = L_step; i > 0; i /= 2) {
        for (int j = 0; j < L_step; ++j) {
            int idxH = (l - 1) - ((j % i) + (j / i) * i * 2);
            int idxL = idxH - i;
            if (idxH >= 0 && idxL >= 0) {
                CAS<IdxType, ValType, ASCEND>()(LA[idxL], LA[idxH]);
            }
        }
    }
    // stepsize < 32
    #pragma unroll
    for (int i = 0; i < l; ++i) {
        warpMerge<IdxType, ValType, ASCEND>(LA[i], lane);
    }

    // step3. sort RA
    //int R_step = (r & (r - 1)) == 0 ? 31 - __clz(r) : 32 - __clz(r);
    int R_step = (r & (r - 1)) == 0 ? (r / 2) : (1 << (31 - __clz(r)));
    // stepsize >= 32
    #pragma unroll
    for (int i = R_step; i > 0; i /= 2) {
        for (int j = 0; j < R_step; ++j) {
            int idxL = (j % i) + (j / i) * i * 2;
            int idxH = idxL + i;
            if (idxL < r && idxH < r) {
                CAS<IdxType, ValType, ASCEND>()(RA[idxL], RA[idxH]);
            }
        }
    }
    // stepsize < 32
    #pragma unroll
    for (int i = 0; i < r; ++i) {
        warpMerge<IdxType, ValType, ASCEND>(RA[i], lane);
    }
}

template<typename IdxType, typename ValType, bool ASCEND>
__device__ __forceinline__ void warpSort(Pair<IdxType, ValType> &data, int lane) {
    // 1. merge every two neighbouring sequence of length 1
    bool flag = lane & 1;
    int srcLane = flag ? lane - 1 : lane + 1;
    ValType new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    IdxType new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    // 2. merge every two neighbouring sequence of length 2
    flag = lane & 2;
    srcLane = (lane / 4) * 8 + 3 - lane;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    // 2.1 sort every sequence of length 2
    flag = lane & 1;
    srcLane = flag ? lane - 1 : lane + 1;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    // 3. merge every two neighbouring sequence of length 4
    flag = lane & 4;
    srcLane = (lane / 8) * 16 + 7 - lane;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    // 3.1 sort every sequence of length 4
    flag = lane & 2;
    srcLane = flag ? lane - 2 : lane + 2;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 1;
    srcLane = flag ? lane - 1 : lane + 1;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    // 4. merge every two neighbouring sequence of length 8
    flag = lane & 8;
    srcLane = (lane / 16) * 32 + 15 - lane;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    // 4.1 sort every sequence of length 8
    flag = lane & 4;
    srcLane = flag ? lane - 4 : lane + 4;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 2;
    srcLane = flag ? lane - 2 : lane + 2;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 1;
    srcLane = flag ? lane - 1 : lane + 1;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }

    // 5. merge every two neighbouring sequence of length 16
    flag = lane & 16;
    srcLane = 31 - lane;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    // 5.1 sort every sequence of length 16
    flag = lane & 8;
    srcLane = flag ? lane - 8 : lane + 8;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 4;
    srcLane = flag ? lane - 4 : lane + 4;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 2;
    srcLane = flag ? lane - 2 : lane + 2;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
    flag = lane & 1;
    srcLane = flag ? lane - 1 : lane + 1;
    new_val = __shfl_val_sync(0xffffffff, data.val, srcLane);
    new_idx = __shfl_val_sync(0xffffffff, data.idx, srcLane);
    if (flag ? Compare<ValType, ASCEND>()(data.val, new_val) :
               Compare<ValType, ASCEND>()(new_val, data.val)) {
        data.val = new_val;
        data.idx = new_idx;
    }
}

template<typename IdxType, typename ValType, bool ASCEND>
__device__ __forceinline__ void sort(Pair<IdxType, ValType> *Arr, int len, int lane) {
    // step1. divide Array into segments of 32 elements,
    //        and sort every segment
    #pragma unroll
    for (int i = 0; i < len; ++i) {
        warpSort<IdxType, ValType, ASCEND>(Arr[i], lane);
    }

    // step2. merge segment
    // lenClip = 2 ^ floor(log2(len))
    int lenClip = 1 << (31 - __clz(len));
    #pragma unroll
    for (int i = lenClip; i < len; ++i) {
        merge<IdxType, ValType, ASCEND>(Arr + lenClip - 1, Arr + i, i - lenClip + 1, 1, lane);
    }

    int32_t tail = len - lenClip;
    for (int i = lenClip / 2, stepSize = 1; i > 0; i /= 2, stepSize *= 2) {
        for (int j = 0; j < i - 1; ++j) {
            Pair<IdxType, ValType> *LA = Arr + j * stepSize * 2;
            Pair<IdxType, ValType> *RA = LA + stepSize;
            merge<IdxType, ValType, ASCEND>(LA, RA, stepSize, stepSize, lane);
        }
        Pair<IdxType, ValType> *LA = Arr + (i - 1) * stepSize * 2;
        Pair<IdxType, ValType> *RA = LA + stepSize;
        merge<IdxType, ValType, ASCEND>(LA, RA, stepSize, stepSize + tail, lane);
    }
}

template <typename IdxType, typename ValType, int t, bool ASCEND>
__device__ __forceinline__ void pushNew(Pair<IdxType, ValType> &ele, Pair<IdxType, ValType> *threadQ) {
    if (Compare<ValType, ASCEND>()(ele.val, threadQ[t - 1].val)) {
        threadQ[t - 1].set(ele);
        for (int i = t - 2; i >= 0; --i) {
            if (!Compare<ValType, ASCEND>()(ele.val, threadQ[i].val)) break;
            threadQ[i + 1].set(threadQ[i]);
            threadQ[i].set(ele);
        }
    }
}

#endif  // TOPK_PQ_UTILS_CUH_
