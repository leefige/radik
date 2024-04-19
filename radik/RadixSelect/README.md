# RadiK

RadiK provides a header-only library for GPU radix top-k.

## Usage

Just put this `RadixSelect` folder into your project.
Here is a simple example.

```cpp
#include <RadixSelect/topk_radixselect.h>

void foo(...) {
  size_t workspaceSize = 0;
  getRadixSelectLWorkSpaceSize<ValType>(k, maxTaskLength, batchSize,
                                        &workspaceSize);

  // allocate workspace
  // it can also be pre-allocated, but make sure the size is sufficient
  void *workspace = nullptr;
  cudaMalloc(&workspace, workspaceSize);

  // launch kernel on a specific CUDA stream
  topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, true>(
    valIn, idxIn, valOut, idxOut, workspace, taskLengthArray,
    batchSize, k, stream);

  // alternatively, if there is no input index, just pass nullptr
  topKRadixSelectL<IdxType, LARGEST, ASCEND, WITHSCALE, false>(
    valIn, nullptr, valOut, idxOut, workspace, taskLengthArray,
    batchSize, k, stream);
}
```

For detailed usage and explanation of the parameters, please refer to the header file [`topk_radixselect.h`](topk_radixselect.h).
For more examples, please refer to the test cases in [`../test/test_radik.cu`](../test/test_radik.cu).

## Citation

If you find our work helpful, feel free to cite our [paper](https://doi.org/10.1145/3650200.3656596).

```bibtex
@inproceedings{radik2024,
  title = {RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection},
  author = {Li, Yifei and Zhou, Bole and Zhang, Jiejing and Wei, Xuechao and Li, Yinghan and Chen, Yingda},
  booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
  year = {2024}
}
```

## License

The entire RadiK project, including the RadiK source codes in this folder, is licensed under the MIT license.
