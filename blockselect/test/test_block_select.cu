#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

#include "../src/BlockSelectKernel.cuh"

#include <curand.h>

// #define CORRECTNESS_CHECK

static constexpr bool LARGEST = 0;

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

    if (argc >= 4) {
        BATCHSIZE = atoi(argv[1]);
        N = (1<<atoi(argv[2]));
        K = atoi(argv[3]);
    } else {
        printf("Please enter BATCHSIZE:\n");
        std::cin>>BATCHSIZE;
        printf("Please enter power of N(max val 29):\n");
        std::cin>>N;
        N = (1<<N);
        printf("Please enter K:\n");
        std::cin>>K;
    }

    profBlockSelect<int32_t, float>(BATCHSIZE, N, K);

    return 0;
}
