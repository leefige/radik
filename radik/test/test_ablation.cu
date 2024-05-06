#include <curand.h>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <numeric>

#include "../RadixSelect/topk_radixselect.h"
#include "../ablation/baseline.h"
#include "error_check.hpp"

// #define CORRECTNESS_CHECK

// Default: smallest top-k, no index
static constexpr bool LARGEST = 0;
static constexpr bool ASCEND = 1;
static constexpr bool WITHIDXIN = 0;
static constexpr bool WITHSCALE = 0;

enum AblationType {
    ABLATION_BASELINE = 0,
    ABLATION_ONLY_1 = 1,    // only (1)
    ABLATION_ONLY_2 = 2,    // only (2)
    ABLATION_ONLY_3 = 3,    // only (3)
    ABLATION_EXCEPT_1 = 4,  // except (1)
    ABLATION_EXCEPT_2 = 5,  // except (2)
    ABLATION_EXCEPT_3 = 6,  // except (3)
    ABLATION_ALL = 7        // all
};

// =========================================================

#define RECORD_START()                              \
    do {                                            \
        CHECK_CUDA(cudaEventCreate(&start));        \
        CHECK_CUDA(cudaEventCreate(&end));          \
        CHECK_CUDA(cudaEventRecord(start, stream)); \
    } while (0)

#define RECORD_END()                                        \
    do {                                                    \
        CHECK_CUDA(cudaEventRecord(end, stream));           \
        CHECK_CUDA(cudaEventSynchronize(end));              \
        CHECK_CUDA(cudaEventElapsedTime(&time, start, end));\
        printf("elapsed: %f ms\n", time);                   \
        CHECK_CUDA(cudaEventDestroy(start));                \
        CHECK_CUDA(cudaEventDestroy(end));                  \
    } while (0)

// Ablation wrapper

// base
template <typename IdxType, typename ValType>
void ablation_baseline(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &TASKOFFSET,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    for (int i = 0; i < BATCHSIZE; ++i) {
        topKRadixSelectL_baseline<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
            valIn_dev + TASKOFFSET[i],
            idxIn_dev + TASKOFFSET[i],
            valOut_dev + K * i,
            idxOut_dev + K * i,
            workSpace,
            TASKLEN.data() + i,
            1,
            K,
            stream);
    }
    RECORD_END();
}

// only atomics + buffer
template <typename IdxType, typename ValType>
void ablation_only_1(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &TASKOFFSET,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    for (int i = 0; i < BATCHSIZE; ++i) {
        topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
            valIn_dev + TASKOFFSET[i],
            idxIn_dev + TASKOFFSET[i],
            valOut_dev + K * i,
            idxOut_dev + K * i,
            workSpace,
            TASKLEN.data() + i,
            1,
            K,
            stream);
    }
    RECORD_END();
}

// only rescheduling
template <typename IdxType, typename ValType>
void ablation_only_2(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &/* TASKOFFSET */,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    topKRadixSelectL_baseline<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
        valIn_dev,
        idxIn_dev,
        valOut_dev,
        idxOut_dev,
        workSpace,
        TASKLEN.data(),
        BATCHSIZE,
        K,
        stream);
    RECORD_END();
}

// only padding
template <typename IdxType, typename ValType>
void ablation_only_3(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> & TASKOFFSET,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    for (int i = 0; i < BATCHSIZE; ++i) {
        // still false, for all offsets are not aligned
        topKRadixSelectL_baseline<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
            valIn_dev + TASKOFFSET[i],
            idxIn_dev + TASKOFFSET[i],
            valOut_dev + K * i,
            idxOut_dev + K * i,
            workSpace,
            TASKLEN.data() + i,
            1,
            K,
            stream);
    }
    RECORD_END();
}

// except atomics + buffer
template <typename IdxType, typename ValType>
void ablation_except_1(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &/* TASKOFFSET */,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    topKRadixSelectL_baseline<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN>(
        valIn_dev,
        idxIn_dev,
        valOut_dev,
        idxOut_dev,
        workSpace,
        TASKLEN.data(),
        BATCHSIZE,
        K,
        stream);
    RECORD_END();
}

// except rescheduling
template <typename IdxType, typename ValType>
void ablation_except_2(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &TASKOFFSET,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    for (int i = 0; i < BATCHSIZE; ++i) {
        // still false, for all offsets are not aligned
        topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
            valIn_dev + TASKOFFSET[i],
            idxIn_dev + TASKOFFSET[i],
            valOut_dev + K * i,
            idxOut_dev + K * i,
            workSpace,
            TASKLEN.data() + i,
            1,
            K,
            stream);
    }
    RECORD_END();
}

// except padding
template <typename IdxType, typename ValType>
void ablation_except_3(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &/* TASKOFFSET */,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, false>(
        valIn_dev,
        idxIn_dev,
        valOut_dev,
        idxOut_dev,
        workSpace,
        TASKLEN.data(),
        BATCHSIZE,
        K,
        stream);
    RECORD_END();
}

// all
template <typename IdxType, typename ValType>
void ablation_all(
        const ValType *valIn_dev,
        const IdxType *idxIn_dev,
        ValType *valOut_dev,
        IdxType *idxOut_dev,
        void *workSpace,
        const std::vector<int> &TASKLEN,
        const std::vector<int64_t> &/* TASKOFFSET */,
        const int &BATCHSIZE,
        const int &K,
        cudaStream_t stream) {
    float time = 0.;
    cudaEvent_t start, end;

    RECORD_START();
    topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN>(
        valIn_dev,
        idxIn_dev,
        valOut_dev,
        idxOut_dev,
        workSpace,
        TASKLEN.data(),
        BATCHSIZE,
        K,
        stream);
    RECORD_END();
}

// =========================================================

template <typename IdxType, typename ValType>
void runAblation(
        const int ABLATION_TYPE,
        const int BATCHSIZE,
        const int N,
        const int K) {
    // prepare data CPU

    // N is power of 2, ensure missaligned
    std::vector<int> TASKLEN(BATCHSIZE, N);
    TASKLEN[0] = N - 1;

    const int64_t TOTALLEN = std::accumulate(TASKLEN.begin(), TASKLEN.end(), 0);
    const int64_t MAXLEN = *std::max_element(TASKLEN.begin(), TASKLEN.end());
    std::vector<int64_t> TASKOFFSET(BATCHSIZE + 1, 0);
    for (int i = 0; i < BATCHSIZE; ++i) {
        TASKOFFSET[i + 1] = TASKLEN[i] + TASKOFFSET[i];
    }

    std::vector<ValType> valIn(TOTALLEN);
    std::vector<IdxType> idxIn(TOTALLEN);
    std::cout<<"batchSize: "<<BATCHSIZE<<", K: "<<K<<std::endl;
    for (int i = 0; i < BATCHSIZE; ++i) {
        std::cout<<"["<<i<<", "<<TASKLEN[i]<<"]"<<std::endl;
    }

    // prepare data GPU
    ValType *valIn_dev = 0;
    CHECK_CUDA(cudaMalloc(&valIn_dev, sizeof(ValType) * TOTALLEN));
    // using U[0, 1]
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CHECK_CURAND(curandGenerateUniform(gen, valIn_dev, TOTALLEN));
#ifdef CORRECTNESS_CHECK
    CHECK_CUDA(cudaMemcpy(valIn.data(), valIn_dev, sizeof(ValType) * TOTALLEN, cudaMemcpyDeviceToHost));
#endif
    CHECK_CURAND(curandDestroyGenerator(gen));

    IdxType *idxIn_dev = 0;
    if (WITHIDXIN) {
        CHECK_CUDA(cudaMalloc(&idxIn_dev, sizeof(IdxType) * TOTALLEN));
        CHECK_CUDA(cudaMemcpy(idxIn_dev, idxIn.data(), sizeof(ValType) * TOTALLEN, cudaMemcpyHostToDevice));
    }
    ValType *valOut_dev = 0;
    CHECK_CUDA(cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K));
    IdxType *idxOut_dev = 0;
    CHECK_CUDA(cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K));
    size_t workSpaceSize = 0;
    getRadixSelectLWorkSpaceSize<ValType>(K, MAXLEN, BATCHSIZE, &workSpaceSize);
    void *workSpace = 0;
    CHECK_CUDA(cudaMalloc(&workSpace, workSpaceSize));
    cudaStream_t stream0;
    CHECK_CUDA(cudaStreamCreate(&stream0));

    // call wrapper
    switch (ABLATION_TYPE) {
    case ABLATION_BASELINE:
        ablation_baseline(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_ONLY_1:
        ablation_only_1(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_ONLY_2:
        ablation_only_2(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_ONLY_3:
        ablation_only_3(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_EXCEPT_1:
        ablation_except_1(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_EXCEPT_2:
        ablation_except_2(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_EXCEPT_3:
        ablation_except_3(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    case ABLATION_ALL:
        ablation_all(valIn_dev, idxIn_dev, valOut_dev, idxOut_dev, workSpace,
                          TASKLEN, TASKOFFSET, BATCHSIZE, K, stream0);
        break;
    default:
        throw std::runtime_error("Bad ABLATION_TYPE");
    }

#ifdef CORRECTNESS_CHECK
    std::vector<ValType> valOut_host(BATCHSIZE * K);
    CHECK_CUDA(cudaMemcpy(valOut_host.data(), valOut_dev, sizeof(ValType) * BATCHSIZE * K, cudaMemcpyDeviceToHost));
    std::vector<IdxType> idxOut_host(BATCHSIZE * K);
    CHECK_CUDA(cudaMemcpy(idxOut_host.data(), idxOut_dev, sizeof(IdxType) * BATCHSIZE * K, cudaMemcpyDeviceToHost));
    std::vector<std::pair<ValType, IdxType>> res_gpu(BATCHSIZE * K);
    for (int i = 0; i < BATCHSIZE * K; ++i) {
        res_gpu[i] = std::make_pair(valOut_host[i], idxOut_host[i]);
    }
    std::vector<std::pair<ValType, IdxType>> res_cpu(TOTALLEN);
    for (int i = 0; i < BATCHSIZE; ++i) {
        for (int j = 0; j < TASKLEN[i]; ++j) {
            res_cpu[TASKOFFSET[i] + j] = std::make_pair(valIn[TASKOFFSET[i] + j], WITHIDXIN ? idxIn[TASKOFFSET[i] + j] : j);
        }
    }
    for (int i = 0; i < BATCHSIZE; ++i) {
        std::sort(res_cpu.begin() + TASKOFFSET[i], res_cpu.begin() + TASKOFFSET[i + 1], [](std::pair<ValType, IdxType>& a, std::pair<ValType, IdxType>& b) { return a.first < b.first; });
        int offset = LARGEST ? TASKLEN[i] - K : 0;
        for (int j = 0; j < K; ++j) {
            int cpu_off = ASCEND ? (TASKOFFSET[i] + offset + j) : (TASKOFFSET[i] + K - j + 1);
            if (res_cpu[cpu_off].first != res_gpu[i * K + j].first) {
                std::cout<<"error at ["<<i<<", "<<j<<"], CPU:"<<"("<<res_cpu[cpu_off].first<<", "<<res_cpu[cpu_off].second<<")"<<", GPU:"<<"("<<res_gpu[i * K + j].first<<", "<<res_gpu[i * K + j].second<<")"<<std::endl;
            }
        }
    }
#endif

    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaFree(valIn_dev));
    if (WITHIDXIN) {
        CHECK_CUDA(cudaFree(idxIn_dev));
    }
    CHECK_CUDA(cudaFree(valOut_dev));
    CHECK_CUDA(cudaFree(idxOut_dev));
    CHECK_CUDA(cudaFree(workSpace));
}

// ==================================================================

int main(int argc, char *argv[]) {
    int TYPE;
    int BATCHSIZE;
    int N;
    int K;

    if (argc >= 5) {
        TYPE = atoi(argv[1]);
        BATCHSIZE = atoi(argv[2]);
        N = (1<<atoi(argv[3]));
        K = atoi(argv[4]);
    } else {
        printf("Please enter ABLATION_TYPE:\n");
        std::cin>>TYPE;
        printf("Please enter BATCHSIZE:\n");
        std::cin>>BATCHSIZE;
        printf("Please enter power of N(max val 29):\n");
        std::cin>>N;
        N = (1<<N);
        printf("Please enter K:\n");
        std::cin>>K;
    }

    runAblation<int32_t, float>(TYPE, BATCHSIZE, N, K);

    return 0;
}
