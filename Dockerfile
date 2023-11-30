# NVIDIA GPU SM version (Compute Capability)
ARG SM_VERSION=75
# Mirror PyPi repository
ARG PIP_SOURCE_URL=https://mirrors.aliyun.com/pypi/simple/

# ===========================
# Build image
FROM nvidia/cuda:11.6.2-devel-centos7 as build-image
ARG SM_VERSION

RUN yum clean all && yum makecache
RUN yum install -y wget patch openssl-devel
RUN yum install -y centos-release-scl && yum install -y devtoolset-10

WORKDIR /root/building

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz
RUN tar xzf cmake-3.26.3.tar.gz
RUN cd cmake-3.26.3 && source /opt/rh/devtoolset-10/enable \
    && ./configure --parallel=`nproc` && make -j`nproc` && make install

# Clean up
RUN rm -rf /root/building

ENV CUDA_SM_VERSION ${SM_VERSION}
ENV CUDA_HOME /usr/local/cuda

WORKDIR /radik

# Build bitonic select
COPY bitonic bitonic
COPY patches/bitonic/Makefile.patch bitonic
RUN cd bitonic && patch Makefile Makefile.patch
RUN cd bitonic && source /opt/rh/devtoolset-10/enable \
    && make -j`nproc` CUDA_PATH=$CUDA_HOME GENCODE_FLAGS=-arch=sm_$CUDA_SM_VERSION

# Build block select (PQ-block)
COPY blockselect blockselect
RUN cd blockselect && source /opt/rh/devtoolset-10/enable \
    && make -j`nproc` all && make clean

# Build RadiK & grid select (PQ-grid)
COPY radik radik
RUN cd radik && source /opt/rh/devtoolset-10/enable && make -j`nproc` all

# ===========================
# Release image
FROM nvidia/cuda:11.6.2-runtime-centos7
ARG PIP_SOURCE_URL

RUN yum install -y python3 python3-pip

# Copy built binaries
COPY --from=build-image /radik /radik
WORKDIR /radik

# Install Python requirements
ENV PIP_SOURCE ${PIP_SOURCE_URL}
RUN python3 -m pip install pip -U -i $PIP_SOURCE
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt -i $PIP_SOURCE

# Evaluation scripts
COPY eval eval

COPY scripts/entry.sh /opt
RUN chmod +x /opt/entry.sh
ENTRYPOINT ["/opt/entry.sh"]
