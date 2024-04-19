RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection
======

[![DOI](https://zenodo.org/badge/725482731.svg)](https://zenodo.org/doi/10.5281/zenodo.10224741)

RadiK is a highly optimized GPU-parallel radix top-k selection that is scalable with *k*, input length, and batch size.
It is also robust against adversarial input distributions.

This artifact provides a docker container image to reproduce evaluations in our paper.

For the source codes of RadiK, please refer to the [`radik/RadixSelect`](radik/RadixSelect) folder.

# Table of Contents

- [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](#radik-scalable-and-optimized-gpu-parallel-radix-top-k-selection)
- [Table of Contents](#table-of-contents)
- [Getting Started Guide](#getting-started-guide)
  - [Platform](#platform)
  - [Set `SM_VERSION` (IMPORTANT)](#set-sm_version-important)
  - [`runme.sh`](#runmesh)
  - [Run without GNU `make`](#run-without-gnu-make)
    - [Build the Image](#build-the-image)
    - [Run a Fast Test](#run-a-fast-test)
- [Step-by-Step Instructions](#step-by-step-instructions)
  - [`runme_full.sh`](#runme_fullsh)
  - [Run Evaluations Manually](#run-evaluations-manually)
    - [Start and Log in the Container](#start-and-log-in-the-container)
    - [Run Evaluations and Plot Figures](#run-evaluations-and-plot-figures)
- [Additional Information](#additional-information)
  - [Compatibility and JIT Compilation](#compatibility-and-jit-compilation)
  - [Project Structure](#project-structure)
    - [RadiK and Grid Select](#radik-and-grid-select)
    - [Bitonic Select (Externel)](#bitonic-select-externel)
    - [Block Select (Extracted from Faiss)](#block-select-extracted-from-faiss)
    - [Image Building](#image-building)
    - [Evaluations](#evaluations)
  - [References](#references)
- [Citation](#citation)
- [License](#license)

# Getting Started Guide

Overall, all that is needed is a Docker compatible with NVIDIA GPUs.
An available GNU `make` will greatly facilitate the evaluation.

## Platform

We test our container image on a machine with the following setup:

- CPU: 2x Intel Xeon Platinum 8163 @ 2.50 GHz (Skylake with 24 cores, 48 threads)
- GPU: NVIDIA T4 @ 1.59 GHz with VRAM @ 5.00 GHz (Turing TU104 with Compute Capability 7.5, 16 GiB VRAM)
- Memory: 512 GiB
- OS: CentOS 7.4 (kernel version 4.9)
- NVIDIA driver: version 525.60.13
- Docker: version 20.10.6
- NVIDIA Container Toolkit: version 1.5.1

Internet connection is required for building the image.

Other platforms with NVIDIA GPU whose Compute Capability is greater than or equal to 7.5 should also be able to run the evaluations, but the results may be slightly different from what we report in the paper.

We recommend running the evalutaion on a Linux machine using the provided `runme.sh` and `runme_full.sh`.
The container image should also work on Windows, but it may require many extra efforts.
So in this guide, we only cover running evalutaions on Linux.

## Set `SM_VERSION` (IMPORTANT)

This step is a must.
If the `SM_VERSION` is set incorrectly, the evaluation results may be badly interfered by the [JIT compilation](#compatibility-and-jit-compilation).

Before running, please check the Streaming Multiprocessor (SM) version, also known as the Compute Capability, of your NVIDIA GPU.
First, check the name(s) of your GPU(s):

```sh
nvidia-smi --query-gpu=name --format=csv,noheader
```

If you have multiple GPUs and their names are different, by default we will use the GPU 0 (the first GPU), and the following steps only apply to GPU 0.
With the GPU name, please check the SM version (Compute Capability) from the website [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).
For example, the Compute Capability of NVIDIA T4 is 7.5, and we set the envariable `SM_VERSION` to 75:

```sh
export SM_VERSION=75
```

Here is a table for some common GPUs:

| GPU Name         | Compute Capability | `SM_VERSION` |
| ---------------- | -----------------: | -----------: |
| NVIDIA T4        |                7.5 |           75 |
| Geforce RTX 2080 |                7.5 |           75 |
| NVIDIA A100      |                8.0 |           80 |
| NVIDIA A30       |                8.0 |           80 |
| NVIDIA A10       |                8.6 |           86 |
| GeForce RTX 3090 |                8.6 |           86 |
| NVIDIA L40       |                8.9 |           89 |
| GeForce RTX 4090 |                8.9 |           89 |
| NVIDIA H100      |                9.0 |           90 |

The default value of `SM_VERSION` is 75 and is defined in `Makefile`.
You can alse directly set its value in `Makefile` instead of setting the envariable.

## `runme.sh`

The `runme.sh` script includes building the image and running a basic test.
To run this script, GNU `make` is required.

```sh
./runme.sh
```

This will build the image, run a fast test corresponding to Section 5.3.2, and plot Figure 11 (a). Building takes about 20 minutes, depending on your network bandwidth and the number of CPU cores.

In many cases, running Docker commands requires privilege, and you may need to input your password when running `runme.sh`.

You can change `PIP_SOURCE` in `Dockerfile` if you have trouble installing Python packages when building the image.

By default, `plot` folder in this directory is mounted to the container, and all results are saved in that folder.
So to check the result of the fast test, see `plot/2-batched-a.png`.
The variable `PLOT_DIR` in `Makefile` defines where to put the results in your host machine, and you may change this variable.

## Run without GNU `make`

Without `make`, you will need to manually run the following commands in a shell.
The working directory should be this directory.

### Build the Image

```sh
docker build --build-arg SM_VERSION=$SM_VERSION -t radik .
```

Depending on your platform, you may need privilege to run Docker commands.
If the command above fails due to permission issue, try:

```sh
sudo docker build --build-arg SM_VERSION=$SM_VERSION -t radik .
```

This also applies to other Docker commands.

### Run a Fast Test

```sh
docker run --name $USER-radik --rm --gpus '"device=0"' \
    --mount type=bind,src=`pwd`/plot,dst=/radik/plot radik \
    eval/2-batched-a.py
docker run --name $USER-radik --rm --gpus '"device=0"' \
    --mount type=bind,src=`pwd`/plot,dst=/radik/plot radik \
    eval/plot-2-batched-a.py
```

You can also run the test in an interactive manner.
First, start and log in the container.

```sh
docker run -it --name $USER-radik --rm --gpus '"device=0"' \
    --mount type=bind,src=`pwd`/plot,dst=/radik/plot radik
```

Then, run the following commands in the container.

```sh
cd eval
# run the evaluation
./2-batched-a.py
# plot the results
./plot-2-batched-a.py
```

# Step-by-Step Instructions

## `runme_full.sh`

The easiest way to run the full evaluation is to directly run the `runme_full.sh` script.
By default, the output figures will be saved in `plot` folder.
The correspondence between the output figures and the figures in our paper is shown in the following table.

| Output Figure Name        | Figure in Paper |
| ------------------------- | --------------- |
| `1-single-query-all.png`  | Figure 7        |
| `1-single-query-last.png` | Figure 8        |
| `2-batched-a.png`         | Figure 9 (a)    |
| `2-batched-b.png`         | Figure 9 (b)    |
| `2-batched-c.png`         | Figure 9 (c)    |
| `4-ablation.png`          | Figure 10       |
| `ex1-median.png`          | Figure 11       |
| `3-skewed-a.png`          | Figure 12 (a)   |
| `3-skewed-b.png`          | Figure 12 (b)   |
| `ex2-skewed-a.png`        | Figure 12 (c)   |
| `ex2-skewed-b.png`        | Figure 12 (d)   |
| `ex3-zipf-a.png`          | Figure 13 (a)   |
| `ex3-zipf-b.png`          | Figure 13 (b)   |

## Run Evaluations Manually

If `runme_full.sh` does not work, you can run all the evaluations manually.
As stated before, GNU `make` will facilitate the whole process.

To build the image, please refer to [Build the Image](#build-the-image) section in the [Getting Started Guide](#getting-started-guide) above.

### Start and Log in the Container

```sh
make run
```

Alternatively, run Docker commands directly if `make` is unavailable.

```sh
docker run -it --name $USER-radik --rm --gpus '"device=0"' \
    --mount type=bind,src=`pwd`/plot,dst=/radik/plot radik
```

As stated before, by default, `plot` folder in this directory is mounted to the container and the results are saved there.
You can also mount another directory or docker volumes.
If so, you need to change the `MNT_FLAGS` in the `Makefile`.
For more details, please refer to [Manage data in Docker](https://docs.docker.com/storage/).

To specify the GPU(s) you want to use in the container, change the `RUN_FLAGS` in the `Makefile`. **Note**: this may involve re-building the image with the `SM_VERSION` of the GPU you specified.

### Run Evaluations and Plot Figures

The evaluations in Section 5 of our paper are provided as Python scripts in `/radik/eval` directory in the container, e.g., `2-batched-a.py` runs one of the evaluations in Section 5.3.2.

Inside the container, the default working directory is `/radik`.
You can run the scripts in `eval` folder, and the results are saved to `/radik/plot`, which is mapped to `./plot` in the host machine by default.

After running the evaluation scripts, to produce the figures in the paper, run `plot-*.py` scripts.
The scripts will indicate where the results are saved.
For the mapping between the output figures and the figures in the paper, please refer to [`runme_full.sh`](#runme_fullsh) section above.

```sh
# go into eval folder
cd eval
```

1. Section 5.3.1 (Figure 7 and 8)

    ```sh
    # run tests
    ./1-simple-topk.py
    # plot Figure 7
    ./plot-1-single-query-all.py
    # plot Figure 8
    ./plot-1-single-query-part.py
    ```

2. Section 5.3.2 (Figure 9)

    ```sh
    # run tests
    ./2-batched-a.py
    # plot Figure 9 (a)
    ./plot-2-batched-a.py
    ```

    ```sh
    # run tests
    ./2-batched-b.py
    # plot Figure 9 (b)
    ./plot-2-batched-b.py
    ```

    ```sh
    # run tests
    ./2-batched-c.py
    # plot Figure 9 (c)
    ./plot-2-batched-c.py
    ```

3. Section 5.3.3 (Figure 10)

    ```sh
    # run tests
    ./4-ablation.py
    # plot Figure 10
    ./plot-4-ablation.py
    ```

4. Section 5.4 (Figure 11)

    ```sh
    # run tests
    ./ex1-median.py
    # plot Figure 11
    ./plot-ex1-median-fig.py
    ```

5. Section 5.5 (Figure 12 and 13)

    ```sh
    # run tests
    ./3-skewed.py
    # plot Figure 12 (a) and (b)
    ./plot-3-skewed.py

    ./ex2-skewed.py
    # plot Figure 12 (c) and (d)
    ./plot-ex2-skewed-fig.py

    ./ex3-zipf-a.py
    # plot Figure 13 (a)
    ./plot-ex3-zipf-fig-a.py

    ./ex3-zipf-b.py
    # plot Figure 13 (b)
    ./plot-ex3-zipf-fig-b.py
    ```

# Additional Information

## Compatibility and JIT Compilation

Building your container image from the `Dockerfile` with proper `SM_VERSION` ensures expected behavior on your platform.

Without setting `SM_VERSION`, the image is built with Compute Capability 7.5 by default.
It should also work on GPUs of Compute Capability greater than 7.5, because we generate PTX codes for 7.5 and JIT compilation takes effect on newer architectures.
However, JIT may interfere the evaluation results, resulting in abnormally large time consumption for some test cases.

So, please make sure to set the correct `SM_VERSION`.
Otherwise, you will need to run each evaluation script (e.g., `1-simple-topk.py`) TWICE, so that the results of the second run are not influenced by JIT.

## Project Structure

This artifact contains our own implementation and some external codes, together constructing a docker container image to reproduce the evaluation results in our paper.

### RadiK and Grid Select

In addition to our RadiK, we also implement grid select [3] by ourselves.

```plain
radik/
   ablation     # codes for ablation study
   PQ/          # our implementation of grid select (PQ-grid)
   RadixSelect/ # our implementation of RadiK
   test/        # benchmark RadiK and PQ-grid
   Makefile
```

### Bitonic Select (Externel)

The sub-module `bitonic` contains the open-source implementation of bitonic sort by [the original author](https://github.com/anilshanbhag/gpu-topk) [2].

### Block Select (Extracted from Faiss)

We compare our algorithm with the open-source implementation of block select by [Faiss](https://github.com/facebookresearch/faiss) [1].
To make it easier to reproduce the evaluations, we extract a minimum code package to run block select from Faiss and put it in the directory `blockselect`.
We slightly modify the original codes to work around the `Tensor` data structure of Faiss.

### Image Building

See [Build the Image](#build-the-image).

```plain
patches/          # minor patch for bitonic select
scripts/          # useful scripts for building the image
Dockerfile
Makefile
requirements.txt
```

### Evaluations

See [Run Evaluations and Plot Figures](#run-evaluations-and-plot-figures).

```plain
eval/         # scripts for running evaluations and reproducing figures
plot/         # default directory for output tables and figures
runme.sh      # getting-started script
runme_full.sh # script for full evaluation
```

## References

[1] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7, 3, 535–547. https://doi.ieeecomputersociety.org/10.1109/TBDATA.2019.2921572

[2] Anil Shanbhag, Holger Pirk, and Samuel Madden. 2018. Efficient Top-K Query Processing on Massively Parallel Hardware. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18). Association for Computing Machinery, New York, NY, USA, 1557–1570. https://doi.org/10.1145/3183713.3183735

[3] Christina Zhang and Yong Wang. 2020. Accelerating top-k computation on GPU. NVIDIA. https://live.nvidia.cn/gtc-od/attachments/CNS20315.pdf.

# Citation

If you find our work helpful, feel free to cite our [paper](https://doi.org/10.1145/3650200.3656596).

```bibtex
@inproceedings{radik2024,
  title = {RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection},
  author = {Li, Yifei and Zhou, Bole and Zhang, Jiejing and Wei, Xuechao and Li, Yinghan and Chen, Yingda},
  booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
  year = {2024}
}
```

# License

This project, including the RadiK source codes in the [`radik/RadixSelect`](radik/RadixSelect) folder, is licensed under the [MIT license](LICENSE).
