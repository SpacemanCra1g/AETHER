#!/usr/bin/zsh

# cmake -S . -B build-cuda-3D \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DKokkos_ARCH_AMPERE86=OFF \
#     -DKokkos_ARCH_ADA89=ON \
#     -DKokkos_ENABLE_DEBUG=OFF \
#     -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
#     -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
#     -DKokkos_ENABLE_OPENMP=ON \
#     -DKokkos_ENABLE_CUDA=ON \
#     -DKokkos_ENABLE_THREADS=OFF \
#     -DKokkos_ENABLE_SERIAL=OFF \
#     -DKokkos_ENABLE_CUDA_LAMBDA=ON \
#     -DAETHER_DIM=3 \
#     -DAETHER_PHYSICS=Euler \
#     -DENABLE_OPENMP=ON \
#     -DENABLE_CUDA=ON

# cmake -S . -B build-cuda-2D \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DKokkos_ARCH_AMPERE86=OFF \
#     -DKokkos_ARCH_ADA89=ON \
#     -DKokkos_ENABLE_DEBUG=OFF \
#     -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
#     -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
#     -DKokkos_ENABLE_OPENMP=ON \
#     -DKokkos_ENABLE_CUDA=ON \
#     -DKokkos_ENABLE_THREADS=OFF \
#     -DKokkos_ENABLE_SERIAL=OFF \
#     -DKokkos_ENABLE_CUDA_LAMBDA=ON \
#     -DAETHER_DIM=2 \
#     -DAETHER_PHYSICS=Euler \
#     -DENABLE_OPENMP=ON \
#     -DENABLE_CUDA=ON

cmake -S . -B build-cuda-1D \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ARCH_AMPERE86=OFF \
    -DKokkos_ARCH_ADA89=ON \
    -DKokkos_ENABLE_DEBUG=OFF \
    -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
    -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_THREADS=OFF \
    -DKokkos_ENABLE_SERIAL=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DAETHER_DIM=1 \
    -DAETHER_PHYSICS=Euler \
    -DENABLE_OPENMP=ON \
    -DENABLE_CUDA=ON \
    -DAETHER_FORCE_DIM=2


# cmake -S . -B build-OPENMP-1D \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DKokkos_ARCH_AMPERE86=OFF \
#     -DKokkos_ARCH_ADA89=OFF \
#     -DKokkos_ENABLE_DEBUG=OFF \
#     -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
#     -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
#     -DKokkos_ENABLE_OPENMP=ON \
#     -DKokkos_ENABLE_CUDA=OFF \
#     -DKokkos_ENABLE_THREADS=OFF \
#     -DKokkos_ENABLE_SERIAL=OFF \
#     -DKokkos_ENABLE_CUDA_LAMBDA=OFF \
#     -DAETHER_DIM=1 \
#     -DAETHER_PHYSICS=Euler \
#     -DENABLE_OPENMP=ON \
#     -DENABLE_CUDA=OFF

# cmake -S . -B build-OPENMP-2D \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DKokkos_ARCH_AMPERE86=OFF \
#     -DKokkos_ARCH_ADA89=OFF \
#     -DKokkos_ENABLE_DEBUG=OFF \
#     -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
#     -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
#     -DKokkos_ENABLE_OPENMP=ON \
#     -DKokkos_ENABLE_CUDA=OFF \
#     -DKokkos_ENABLE_THREADS=OFF \
#     -DKokkos_ENABLE_SERIAL=OFF \
#     -DKokkos_ENABLE_CUDA_LAMBDA=OFF \
#     -DAETHER_DIM=1 \
#     -DAETHER_PHYSICS=Euler \
#     -DENABLE_OPENMP=ON \
#     -DENABLE_CUDA=OFF

# cmake -S . -B build-debug \
#     -DCMAKE_BUILD_TYPE=Debug\
#     -DKokkos_ENABLE_DEBUG=ON \
#     -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
#     -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF \
#     -DKokkos_ENABLE_OPENMP=ON \
#     -DKokkos_ENABLE_CUDA=OFF \
#     -DKokkos_ENABLE_THREADS=OFF \
#     -DKokkos_ENABLE_SERIAL=OFF \
#     -DAETHER_DIM=1 \
#     -DAETHER_PHYSICS=Euler \
#     -DENABLE_OPENMP=ON \
#     -DENABLE_CUDA=OFF