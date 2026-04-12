#!/usr/bin/zsh

# cmake -S . -B build-cuda-2D \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DKokkos_ARCH_AMPERE86=ON \
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


cmake -S . -B build-omp-prof \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DKokkos_ENABLE_DEBUG=OFF \
    -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
    -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_THREADS=OFF \
    -DKokkos_ENABLE_SERIAL=OFF \
    -DAETHER_DIM=2 \
    -DAETHER_PHYSICS=Euler \
    -DENABLE_OPENMP=ON \
    -DENABLE_CUDA=OFF