#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE (16)


/*****************************************************************************\
 * Section 1: Helper Structs                                                 *
\*****************************************************************************/

struct Identity
{
    __device__
    static double func(double x) {return x;}
};


/*****************************************************************************\
 * Section 2: General Functions                                              *
\*****************************************************************************/


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
 * See myGEMM for more details. We have used this Naive GEMM implementation for
 * the time being. The operation PostOp::func is applied to each element in C
 * after the product has been found. In some cases, the offset is not necessary
 * so we use set the IncludeOffset to false so we do not have to waste a memory
 * access to C.
 */
template<class PostOp, bool IncludeOffset>
__global__ void myGEMM_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N, int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;


    if ((row < M) && (col < N))
    {
        double product = 0;
        double old_val = IncludeOffset ? C[M*col + row] : 0;

        for(int k = 0; k < K; k++)
        {
           product += A[k*M + row] * B[col*K + k];
        }

        C[M*col + row] = PostOp::func(alpha*product + beta*old_val);
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

    myGEMM_kernel<Identity, true><<<gridSize, blockSize>>>(
        A, B, C, *alpha, *beta, M, N, K);

    check_launch("myGEMM_kernel");

    return 0;
}


/**
 * \brief Kernel for myTranspose.
 */
void myTranspose_kernel(double *A, double *AT, int M, int N)
{
    // TODO
}


/**
 * \brief Efficient transpose of an MxN col-major matrix in device memory.
 * 
 * Arguments are as follows:
 * A:  A device pointer pointing to the row-major matrix to be transposed.
 * AT: A device pointer pointing to memory where A Transpose will be stored.
 * M:  The number of rows in A
 * N:  The number of columns in A
 */
int myTranspose(double *A, double *AT, int M, int N)
{
    // TODO
}



void mySpecialHadamard_kernel(double *dA1, double *A1, double *dZ1, 
                              int M, int N) {
    // TODO
}


/**
 * \brief Function for carrying out the special Hadamard product present in our
 * back propogation process.
 *
 * Arguments are as follows:
 * dA1: The Jacobian of the objective function with respect to A1.
 * A1:  The matrix A1 cached from the feed forward process.
 * dZ1: Space where we can store the Jacobian of the objective function with
 *      respect to Z1 (i.e. our result).
 * M:   The number of rows in all of these matrices.
 * N:   The number of columns in all of these matrices.
 */
int mySpecialHadamard(double *dA1, double *A1, double *dZ1, int M, int N)
{
    // TODO
}






