#ifndef TOPK_PQ_CONFIG_H_
#define TOPK_PQ_CONFIG_H_

#include <cmath>

template <typename IdxType_,         // int64 int32 int16
          typename ValType_,         // fp32 fp16 int8
          int kWarpQCapacity_,       //
          int kThreadQLength_,       //
          int kBlockSize_,           //
          bool kWithIdxIn_,          //
          bool kLargest_,            //
          bool kIsSubByte_ = 0>      // for int4
struct TopKPQConfig {
    typedef ValType_ ValType;
    typedef IdxType_ IdxType;
    
    static constexpr int kWarpQCapacity = kWarpQCapacity_;
    static constexpr int kThreadQLength = kThreadQLength_;
    static constexpr bool kWithIdxIn = kWithIdxIn_;
    static constexpr int kBlockSize = kBlockSize_;
    static constexpr bool kAscend = !kLargest_;
    static constexpr bool kIsSubByte = !kIsSubByte_;
    static constexpr int kPackSize = kWithIdxIn ? 16 / std::min(sizeof(ValType), sizeof(IdxType)) : 16 / sizeof(ValType);

};

#endif  // TOPK_PQ_CONFIG_H_
