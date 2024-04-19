#include <curand.h>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <thread>

#include "../RadixSelect/topk_radixselect.h"
#include "zipf.hpp"

// #define CORRECTNESS_CHECK

// Default: smallest top-k, no index
static constexpr bool LARGEST = 0;
static constexpr bool ASCEND = 1;
static constexpr bool WITHIDXIN = 0;

template<typename T>
void generate_random_idx(T *idx, int n) {
    for (int i = 0; i < n; ++i) {
        idx[i] = i;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idx, idx + n, std::default_random_engine(seed));
}

void generate_random_task(int *taskLen, int batchSize, int order) {
    const int lowerBound = order * 9 / 10;
    const int upperBound = order * 11 / 10;
    std::uniform_real_distribution<float> dis(lowerBound, upperBound);
    std::default_random_engine generator;
    generator.seed(1);
    std::generate(taskLen, taskLen + batchSize,
            [&]() { return static_cast<int>(dis(generator));});
}

template <typename T>
void generate_uniform_val(T *valIn, int len, T lowerBound, T upperBound) {
    std::uniform_real_distribution<T> dis(lowerBound, upperBound);
    std::default_random_engine generator;
    generator.seed(1);
    std::generate(valIn, valIn + len, [&]() { return dis(generator);});
}

template <typename T>
void generate_zipf_val(T *valIn, int len, double skew, double scale = 1.0) {
    std::default_random_engine generator;
    generator.seed(1);
    ZipfRejectionSampler<std::default_random_engine> zipf(generator, len, skew);
    std::generate(valIn, valIn + len, [&]() { return static_cast<T>(zipf.getSample() * scale);});
}

template <bool BATCHED, bool PADDING, bool WITHSCALE, typename IdxType, typename ValType>
void profRadixSelectL(const int BATCHSIZE,
                      const int N,
                      const int K,
                      const int RANDOM_TASK,
                      const int DISTRIBUTION_TYPE = 0,
                      const bool SORTING = true) {
    // prepare data CPU
    std::vector<int> TASKLEN(BATCHSIZE, N);
    if (RANDOM_TASK) {
        generate_random_task(TASKLEN.data(), BATCHSIZE, N);
    }
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
        if (WITHIDXIN) {
            generate_random_idx(&idxIn[TASKOFFSET[i]], TASKLEN[i]);
        }
        std::cout<<"["<<i<<", "<<TASKLEN[i]<<"]"<<std::endl;
    }

    // prepare data GPU
    ValType *valIn_dev = 0;
    cudaMalloc(&valIn_dev, sizeof(ValType) * TOTALLEN);
    if (DISTRIBUTION_TYPE == 0) {
        // using U[0, 1]
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, valIn_dev, TOTALLEN);
#ifdef CORRECTNESS_CHECK
        cudaMemcpy(valIn.data(), valIn_dev, sizeof(ValType) * TOTALLEN, cudaMemcpyDeviceToHost);
#endif
        curandDestroyGenerator(gen);
    } else if (DISTRIBUTION_TYPE == 1) {
        // using U[0.6, 0.7]
        generate_uniform_val<ValType>(valIn.data(), TOTALLEN, 0.6, 0.7);
        cudaMemcpy(valIn_dev, valIn.data(), sizeof(ValType) * TOTALLEN, cudaMemcpyDefault);
    } else if (DISTRIBUTION_TYPE == 2) {
        // using U[128.6, 128.7]
        generate_uniform_val<ValType>(valIn.data(), TOTALLEN, 128.6, 128.7);
        cudaMemcpy(valIn_dev, valIn.data(), sizeof(ValType) * TOTALLEN, cudaMemcpyDefault);
    } else if (DISTRIBUTION_TYPE == 3) {
        std::vector<std::thread> thds;
        // using Zipf(N, 1.1)
        for (int i = 0; i < BATCHSIZE; i++) {
            thds.emplace_back([&](int idx) {
                generate_zipf_val<ValType>(valIn.data() + TASKOFFSET[idx], TASKLEN[idx], 1.1, 1.0 / TASKLEN[idx]);
            }, i);
        }
        for (auto& t: thds) {
            t.join();
        }
        cudaMemcpy(valIn_dev, valIn.data(), sizeof(ValType) * TOTALLEN, cudaMemcpyDefault);
    } else if (DISTRIBUTION_TYPE == 4) {
        // all zero
        cudaMemset(valIn_dev, 0, sizeof(ValType) * TOTALLEN);
    } else {
        throw std::runtime_error("Bad distributiion");
    }

    IdxType *idxIn_dev = 0;
    if (WITHIDXIN) {
        cudaMalloc(&idxIn_dev, sizeof(IdxType) * TOTALLEN);
        cudaMemcpy(idxIn_dev, idxIn.data(), sizeof(ValType) * TOTALLEN, cudaMemcpyHostToDevice);
    }
    ValType *valOut_dev = 0;
    cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K);
    IdxType *idxOut_dev = 0;
    cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K);
    size_t workSpaceSize = 0;
    getRadixSelectLWorkSpaceSize<ValType>(K, MAXLEN, BATCHSIZE, &workSpaceSize);
    void *workSpace = 0;
    cudaMalloc(&workSpace, workSpaceSize);
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    float time = 0.;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (BATCHED) {
        // using batched query
        cudaEventRecord(start, stream0);
        topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN, ValType, PADDING>(
            valIn_dev,
            idxIn_dev,
            valOut_dev,
            idxOut_dev,
            workSpace,
            TASKLEN.data(),
            BATCHSIZE,
            K,
            stream0,
            SORTING);
        cudaEventRecord(end, stream0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
    } else {
        // using a loop over tasks
        cudaEventRecord(start, stream0);
        for (int i = 0; i < BATCHSIZE; ++i) {
            topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, WITHIDXIN>(
                valIn_dev + TASKOFFSET[i],
                idxIn_dev + TASKOFFSET[i],
                valOut_dev + K * i,
                idxOut_dev + K * i,
                workSpace,
                TASKLEN.data() + i,
                1,
                K,
                stream0,
                SORTING);
        }
        cudaEventRecord(end, stream0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
    }
    printf("elapsed: %f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

#ifdef CORRECTNESS_CHECK
    std::vector<ValType> valOut_host(BATCHSIZE * K);
    cudaMemcpy(valOut_host.data(), valOut_dev, sizeof(ValType) * BATCHSIZE * K, cudaMemcpyDeviceToHost);
    std::vector<IdxType> idxOut_host(BATCHSIZE * K);
    cudaMemcpy(idxOut_host.data(), idxOut_dev, sizeof(IdxType) * BATCHSIZE * K, cudaMemcpyDeviceToHost);
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

    cudaStreamDestroy(stream0);
    cudaFree(valIn_dev);
    if (WITHIDXIN) {
        cudaFree(idxIn_dev);
    }
    cudaFree(valOut_dev);
    cudaFree(idxOut_dev);
    cudaFree(workSpace);
}

int main(int argc, char *argv[]) {
    int BATCHSIZE;
    int N;
    int K;
    int RANDOM_TASK;
    int SCALING;
    int DISTRIBUTION = 0;
    int USING_BATCH = 1;
    int USING_PADDING = 1;
    int USING_SORTING = 1;

    if (argc >= 6) {
        BATCHSIZE = atoi(argv[1]);
        N = (1<<atoi(argv[2]));
        K = atoi(argv[3]);
        RANDOM_TASK = atoi(argv[4]);
        SCALING = atoi(argv[5]);
        if (argc >= 7) {
            DISTRIBUTION = atoi(argv[6]);
        }
        if (argc >= 8) {
            USING_BATCH = atoi(argv[7]);
        }
        if (argc >= 9) {
            USING_PADDING = atoi(argv[8]);
        }
        if (argc >= 10) {
            USING_SORTING = atoi(argv[9]);
        }
    } else {
        printf("Please enter BATCHSIZE:\n");
        std::cin>>BATCHSIZE;
        printf("Please enter power of N(max val 29):\n");
        std::cin>>N;
        N = (1<<N);
        printf("Please enter K:\n");
        std::cin>>K;
        printf("Please enter RANDOM_TASK:\n0-each task length in this batch will be N\n1-each task length in this batch will fluctuate randomly between 0.9 * N and 1.1 * N\n");
        std::cin>>RANDOM_TASK;
        printf("Please choose whether scaling or not (0: N, 1: Y)\n");
        std::cin>>SCALING;
    }

    if (USING_PADDING == 0) {
        // just use batched query in ablation study
        if (SCALING == 0) {
                profRadixSelectL<true, false, 0, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            } else {
                profRadixSelectL<true, false, 1, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            }
    } else {
        if (USING_BATCH == 1) {
            if (SCALING == 0) {
                profRadixSelectL<true, true, 0, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            } else {
                profRadixSelectL<true, true, 1, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            }
        } else {
            if (SCALING == 0) {
                profRadixSelectL<false, true, 0, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            } else {
                profRadixSelectL<false, true, 1, int32_t, float>(BATCHSIZE, N, K, RANDOM_TASK, DISTRIBUTION, USING_SORTING);
            }
        }
    }

    return 0;
}
