# cuda_matrix_mult
Simple project that shows the effect of different matrix multiplication speed-ups for CUDA

# Results
This code executed on a RTX3080 yields the following times:
  - CPU Time: 4324.901000 ms. Traditional matrix multiplication using the CPU without any parallelization.
  - CUDA Time: 1.128256 ms. Utilizes CUDA for parallelization of matrix multiplication on the GPU.
  - CUDA Time with Shared Memory: 0.871296 ms. Further optimizes CUDA by utilizing shared memory, reducing access times.
  - CUDA Time with Shared Memory and Prefetching to GPU: 0.866304 ms. Includes both shared memory optimization and prefetching data to the GPU, allowing for concurrent computation and data transfer.

