#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include "../PQ/topK_PQ.h"

#include <curand.h>

// #define CORRECTNESS_CHECK

static constexpr bool LARGEST = 0;

template <typename IdxType, typename ValType>
void profWarpSelect(const int BATCHSIZE,
                    const int N,
                    const int K) {
    // prepare data CPU
    const int64_t TOTALLEN = BATCHSIZE * N;

    // prepare data GPU
    ValType *valIn_dev = 0;
    cudaMalloc(&valIn_dev, sizeof(ValType) * BATCHSIZE * N);

    // using U[0, 1]
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, valIn_dev, TOTALLEN);
#ifdef CORRECTNESS_CHECK
    std::vector<ValType> dataIn(BATCHSIZE * N);
    cudaMemcpy(dataIn.data(), valIn_dev, sizeof(ValType) * TOTALLEN, cudaMemcpyDeviceToHost);
#endif
    curandDestroyGenerator(gen);

    // cudaMemcpy(valIn_dev, dataIn.data(), sizeof(ValType) * BATCHSIZE * N, cudaMemcpyHostToDevice);
    IdxType *idxIn_dev = 0;
    ValType *valOut_dev = 0;
    cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K);
    IdxType *idxOut_dev = 0;
    cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K);

    float time = 0.;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    topKPQWarpSelect<IdxType, ValType>(valIn_dev,
                                       idxIn_dev,
                                       valOut_dev,
                                       idxOut_dev,
                                       BATCHSIZE,
                                       K,
                                       N,
                                       LARGEST,
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

template <typename IdxType, typename ValType>
void profBlockSelect(const int BATCHSIZE,
                     const int N,
                     const int K) {
    // prepare data CPU
    const int64_t TOTALLEN = BATCHSIZE * N;

    // prepare data GPU
    ValType *valIn_dev = 0;
    cudaMalloc(&valIn_dev, sizeof(ValType) * BATCHSIZE * N);

    // using U[0, 1]
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, valIn_dev, TOTALLEN);
#ifdef CORRECTNESS_CHECK
    std::vector<ValType> dataIn(BATCHSIZE * N);
    cudaMemcpy(dataIn.data(), valIn_dev, sizeof(ValType) * TOTALLEN, cudaMemcpyDeviceToHost);
#endif
    curandDestroyGenerator(gen);

    IdxType *idxIn_dev = 0;
    ValType *valOut_dev = 0;
    cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K);
    IdxType *idxOut_dev = 0;
    cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K);

    float time = 0.;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    topKPQBlockSelect<IdxType, ValType>(valIn_dev,
                                        idxIn_dev,
                                        valOut_dev,
                                        idxOut_dev,
                                        BATCHSIZE,
                                        K,
                                        N,
                                        LARGEST,
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

template<typename IdxType, typename ValType>
void profGridSelect(const int BATCHSIZE,
                    const int N,
                    const int K) {
    // prepare data CPU
    const int64_t TOTALLEN = BATCHSIZE * N;

    // prepare data GPU
    ValType *valIn_dev = 0;
    cudaMalloc(&valIn_dev, sizeof(ValType) * BATCHSIZE * N);

    // using U[0, 1]
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, valIn_dev, TOTALLEN);
#ifdef CORRECTNESS_CHECK
    std::vector<ValType> dataIn(BATCHSIZE * N);
    cudaMemcpy(dataIn.data(), valIn_dev, sizeof(ValType) * TOTALLEN, cudaMemcpyDeviceToHost);
#endif
    curandDestroyGenerator(gen);

    IdxType *idxIn_dev = 0;
    ValType *valOut_dev = 0;
    cudaMalloc(&valOut_dev, sizeof(ValType) * BATCHSIZE * K);
    IdxType *idxOut_dev = 0;
    cudaMalloc(&idxOut_dev, sizeof(IdxType) * BATCHSIZE * K);
    size_t bufferSize = 0;
    getGridSelectWorkSpaceSize<IdxType, ValType>(BATCHSIZE, K, N, &bufferSize);
    void *buffer_dev = 0;
    cudaMalloc(&buffer_dev, bufferSize);

    float time = 0.;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    topKPQGridSelect<IdxType, ValType>(valIn_dev,
                                       idxIn_dev,
                                       valOut_dev,
                                       idxOut_dev,
                                       buffer_dev,
                                       BATCHSIZE,
                                       K,
                                       N,
                                       LARGEST,
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
    // 0: warp, 1: block, 2: grid
    int ALGO;
    int BATCHSIZE;
    int K;
    int N;

    if (argc >= 5) {
        ALGO = atoi(argv[1]);
        BATCHSIZE = atoi(argv[2]);
        N = (1<<atoi(argv[3]));
        K = atoi(argv[4]);
    } else {
        printf("Please choose algorithm (0: warp select, 1: block select, 2: grid select):\n");
        std::cin>>ALGO;
        printf("Please enter BATCHSIZE:\n");
        std::cin>>BATCHSIZE;
        printf("Please enter power of N(max val 29):\n");
        std::cin>>N;
        N = (1<<N);
        printf("Please enter K:\n");
        std::cin>>K;
    }

    switch (ALGO) {
    case 0:
        profWarpSelect<int32_t, float>(BATCHSIZE, N, K);
        break;
    case 1:
        profBlockSelect<int32_t, float>(BATCHSIZE, N, K);
        break;
    case 2:
        profGridSelect<int32_t, float>(BATCHSIZE, N, K);
        break;
    default:
        printf("Bad algorithm type\n");
        return 1;
    }

    return 0;
}
