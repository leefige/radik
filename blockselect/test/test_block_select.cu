#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>

#include "../src/BlockSelectKernel.cuh"
#include "zipf.hpp"

#include <curand.h>

// #define CORRECTNESS_CHECK

static constexpr bool LARGEST = 0;

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

template <typename IdxType, typename ValType>
void profBlockSelect(const int BATCHSIZE,
                     const int N,
                     const int K,
                     const int DISTRIBUTION_TYPE = 0) {
    // prepare data CPU
    std::vector<int> TASKLEN(BATCHSIZE, N);
    const int64_t TOTALLEN = std::accumulate(TASKLEN.begin(), TASKLEN.end(), 0);
    const int64_t MAXLEN = *std::max_element(TASKLEN.begin(), TASKLEN.end());
    std::vector<int64_t> TASKOFFSET(BATCHSIZE + 1, 0);
    for (int i = 0; i < BATCHSIZE; ++i) {
        TASKOFFSET[i + 1] = TASKLEN[i] + TASKOFFSET[i];
    }

    // prepare data GPU
    ValType *valIn_dev = 0;
    cudaMalloc(&valIn_dev, sizeof(ValType) * BATCHSIZE * N);

    std::vector<ValType> valIn(TOTALLEN);
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

    // IdxType *idxIn_dev = 0;
    ValType *valOut_dev = 0;
    cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K);
    IdxType *idxOut_dev = 0;
    cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K);

    float time = 0.;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    faiss::gpu::runBlockSelect(valIn_dev,
                               valOut_dev,
                               idxOut_dev,
                               LARGEST,
                               K,
                               BATCHSIZE,
                               N,
                               0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);


    printf("elapsed: %f ms\n", time);

#ifdef CORRECTNESS_CHECK
    std::vector<ValType> res(BATCHSIZE * K);
    cudaMemcpy(res.data(), valOut_dev, sizeof(ValType) * BATCHSIZE * K, cudaMemcpyDeviceToHost);
    int offset = LARGEST ? N - K : 0;
    for (int i = 0; i < BATCHSIZE; ++i) {
        std::sort(res.begin() + i * K, res.begin() + (i + 1) * K);
        std::sort(dataIn.begin() + i * N, dataIn.begin() + (i + 1) * N);
        for (int j = 0; j < K; ++j) {
            if (dataIn[i * N + j + offset] != res[i * K + j]) {
                std::cout<<"error at ["<<i<<", "<<j<<"], CPU:"<<dataIn[i * N + j + offset]<<", GPU:"<<res[i * K + j]<<std::endl;
            }
        }
    }
#endif
}

int main(int argc, char *argv[]) {
    int BATCHSIZE;
    int K;
    int N;
    int DISTRIBUTION = 0;

    if (argc >= 4) {
        BATCHSIZE = atoi(argv[1]);
        N = (1<<atoi(argv[2]));
        K = atoi(argv[3]);
        if (argc >= 5) {
            DISTRIBUTION = atoi(argv[4]);
        }
    } else {
        printf("Please enter BATCHSIZE:\n");
        std::cin>>BATCHSIZE;
        printf("Please enter power of N(max val 29):\n");
        std::cin>>N;
        N = (1<<N);
        printf("Please enter K:\n");
        std::cin>>K;
    }

    profBlockSelect<int32_t, float>(BATCHSIZE, N, K, DISTRIBUTION);

    return 0;
}
