#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE (16)

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}

/**
 * \brief Kernel for in-place GEMM opration.
 *
 * See myGEMM for more details.
 */
__global__ void myGEMM_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N, int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;


    if ((row < M) && (col < N))
    {
        double product = 0;
        double old_val = C[M*col + row];

        for(int k = 0; k < K; k++)
        {
           product += A[k*M + row] * B[col*K + k];
        }

        C[M*col + row] = alpha*product + beta*old_val;
    }

}

/*
 * \brief Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B +
 * beta*C. 
 * 
 * This function performs the in-place GEMM operation accelerated by the GPU.
 * The arguments are as follows:
 * 
 * A is an MxK matrix of doubles in col-major format.
 * B is an KxN matrix of doubles in col-major format.
 * C is an MxN matrix of doubles in col-major format.
 * alpha is the address of a scalar to multiply AB by before adding it to the
 * final result.
 * beta is the address of a scalar to multiply C by before adding alpha*AB to
 * it.
 *
 * Note that A, B, and C are pointers to device memory whereas alpha and beta
 * are pointers to host memory.
 */
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K) {

    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize ((M + blockSize.x - 1)/blockSize.x,
                   (N + blockSize.y - 1)/blockSize.y);

    myGEMM_kernel<<<gridSize, blockSize>>>(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}
