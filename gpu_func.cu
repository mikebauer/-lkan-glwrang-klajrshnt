#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#include <stdio.h>

#define BLOCK_SIZE (16)
#define RESULT_BLOCK_Y    (16)
#define SUBMATRIX_K   (4)
#define MAX_GRID_SIZE (65535) 
/******************************************************************************\
 * Section 1: Helper Structs                                                  *
\******************************************************************************/

/**
 * \brief Identity operation.
 * 
 * This struct contains the identity operation so this operation can be accessed
 * via template.
 */
struct Identity
{
    __device__
    static double func(double x) {return x;}
};

/**
 * \brief Sigmoid operation.
 * 
 * This struct contains the sigmoid operation so this operation can be accessed
 * via template.
 */
struct Sigmoid
{
    __device__
    static double func(double x){return 1 / (1 + exp(-x));}
};

/******************************************************************************\
 * Section 2: General Functions                                               *
\******************************************************************************/

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

__global__ void myGEMM_fast_kernel(double *A, double *B, double *C,
                                   double alpha, double beta,
                                   int M, int N, int K)
{
    // Step 1: Find the block root
    int i  =  blockDim.x*threadIdx.y + threadIdx.x;
    int i0 = (blockDim.x*blockDim.y)*blockIdx.x + i;
    int j0 = (RESULT_BLOCK_Y)*blockIdx.y;


    // Step 2: Initialize shared and register memory:
    __shared__ double B_block[SUBMATRIX_K*RESULT_BLOCK_Y];
    double A_block[SUBMATRIX_K];
    double C_out[RESULT_BLOCK_Y];

    // Step 3: Initialize C_out:
    for(int j = 0; j < RESULT_BLOCK_Y; j++)
        C_out[j] = 0;

    int num_iters = ((N - j0 < RESULT_BLOCK_Y) ? (N - j0) : RESULT_BLOCK_Y);
    int B_col = (i/SUBMATRIX_K) + j0;

    // Step 4: Iterate through all but the last blocks of A and B
    for(int k0 = 0; k0 < K; k0 += SUBMATRIX_K)
    {
        // Step 4a: Load the A block into shared memory
        if(i0 < M)
        {
            for(int k = 0; k < SUBMATRIX_K; k++)
            {
                if((k0 + k) < K)
                    A_block[k] = A[M*(k0 + k) + i0];
            }
        }

        __syncthreads();

        // Step 4b: Load the B block into shared memory
        if((B_col < N) && (k0 + (i % SUBMATRIX_K) < K))
        {
            B_block[i] = B[K*B_col + k0 + (i % SUBMATRIX_K)];
        }

        __syncthreads();

        // Step 4c: Compute the results:
        for(int j = 0; j < num_iters; j++)
        {
            for(int k = 0; k < SUBMATRIX_K; k++)
            {
                if((k0 + k) < K) 
                    C_out[j] += A_block[k] * B_block[SUBMATRIX_K*j + k];
            }
        }
    }
    // Step 5: Accumulate results in C
    for(int j = 0; j < num_iters; j++)
    {
        if(i0 < M)
        {
            C[(j0 + j)*M + i0] = alpha*C_out[j] + beta*C[(j0 + j)*M + i0]; 
        }
    }

}

template<bool IncludeOffset, bool TransposeA, bool TransposeB>
__global__ void
__launch_bounds__(256)
myGEMM_shared_kernel(double const * const __restrict__ A, 
                     double const * const __restrict__ B, 
                     double * __restrict__ C,
                     const double alpha, const double beta,
                     const int M, const int N, const int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    double old_val, product;

    __shared__ double A_submat[BLOCK_SIZE*(BLOCK_SIZE + 1)];
    __shared__ double B_submat[BLOCK_SIZE*(BLOCK_SIZE + 1)];


    product  = 0;
    old_val = (IncludeOffset ? C[row + M*col] : 0);
    
    for(int k0 = 0; k0 < K; k0 += blockDim.x)
    {
        __syncthreads();
        // Step 1: Update A_submat:
        if(TransposeA) 
        {
            A_submat[threadIdx.y + (blockDim.x + 1)*threadIdx.x] 
                = ((k0 + threadIdx.x < K) ?
                   ((row - threadIdx.x + threadIdx.y < M) ? 
                    A[(k0 + threadIdx.x) + 
                      K*(row - threadIdx.x + threadIdx.y)] : 0) : 0);
        } else {
            A_submat[threadIdx.x + (blockDim.x + 1)*threadIdx.y] 
                = ((k0 + threadIdx.y < K) ?
                   ((row < M) ? A[row + M*(k0 + threadIdx.y)] : 0) : 0);
        }
    
        // Step 2: Update B_submat:
        if(TransposeB)
        {
            B_submat[threadIdx.y + (blockDim.x+1)*threadIdx.x] 
                = ((k0 + threadIdx.y < K) ?
                   ((col - threadIdx.y + threadIdx.x < N) ? 
                    B[(col - threadIdx.y + threadIdx.x) + 
                      N*(k0 + threadIdx.y)] : 0) : 0);
            
        } else {
            B_submat[threadIdx.x + (blockDim.x+1)*threadIdx.y]
                = ((k0 + threadIdx.x < K) ? 
                ((col < N) ? B[(k0 + threadIdx.x) + K*col] : 0) : 0);
        }
        __syncthreads();
        
        // Step 3: Accumulate the results:
        int num_iters = min(K - k0, blockDim.x);
        for(int k = 0; k < num_iters; k++) 
        {
            product += A_submat[threadIdx.x + k*(blockDim.x+1)] *
                       B_submat[k + threadIdx.y*(blockDim.x+1)];
        }
    }
    if((row < M) && (col < N))
        C[row + M*col] = alpha*product + beta*old_val;
}


template<bool IncludeOffset, bool TransposeA, bool TransposeB>
__global__ void 
__launch_bounds__(256)
myGEMM_tile_kernel(double const * const __restrict__ A, 
                   double const * const __restrict__ B, 
                   double * __restrict__ C,
                   const double alpha, const double beta,
                   const int M, const int N, const int K)
{
    const int block_root_i0 = 64*blockIdx.x;
    const int block_root_j0 = 64*blockIdx.y;

    __shared__ double A_submat[65][4]; // TODO
    __shared__ double B_submat[5][64];

    double product[4][4] = {{0}};
    double A_frag[4];
    double B_frag[4];

    const int tile_root_i = (threadIdx.x/32) * 32 + 2*(threadIdx.x % 8);
    const int tile_root_j = threadIdx.y*16 + ((threadIdx.x % 32)/8) * 2;

    for(int k0 = 0; k0 < K; k0 += 4)
    {

        // Update the shared memory
        __syncthreads();
        if(TransposeA)
        {
            A_submat[threadIdx.x/4 + 16*threadIdx.y][threadIdx.x % 4] = 
              ((block_root_i0 + threadIdx.x/4 + 16*threadIdx.y < M) ? 
              ((k0 + (threadIdx.x % 4) < K) ?
               A[(k0 + (threadIdx.x % 4)) + K*(block_root_i0 + threadIdx.x/4 +
                   16*threadIdx.y)] : 0) : 0);

        } else {
            A_submat[threadIdx.x][threadIdx.y] = 
              ((block_root_i0 + threadIdx.x < M) ? ((k0 + threadIdx.y < K) ?
               A[(block_root_i0 + threadIdx.x) + M*(k0 + threadIdx.y)] : 0): 0);
        }

        if(TransposeB)
        {
            B_submat[threadIdx.y][threadIdx.x] = 
              ((block_root_j0 + threadIdx.x < N) ? ((k0 + threadIdx.y < K) ?
               B[(block_root_j0 + threadIdx.x) + N*(k0 + threadIdx.y)] : 0): 0);
        } else {
            B_submat[threadIdx.x % 4][threadIdx.x/4 + 16*threadIdx.y] =
              ((block_root_j0 + threadIdx.x/4 + 16*threadIdx.y < N) ? 
              ((k0 + (threadIdx.x % 4) < K) ?
               B[(k0 + (threadIdx.x % 4)) + K*(block_root_j0 + threadIdx.x/4 +
                   16*threadIdx.y)] : 0) : 0);
        }

        __syncthreads();


        // Loop through the tiles accumulating the product at each
#pragma unroll 3
        for(int k = 0; k < 4; k++)
        {
            // Step 1: Copy the fragments of A and B for this thread.
            for(int i = 0; i < 2; i++)
                for(int l = 0; l < 2; l++)
                {
                    A_frag[2*i + l] = A_submat[tile_root_i + 16*i + l][k];
                    B_frag[2*i + l] = B_submat[k][tile_root_j +  8*i + l];
                }
            // Step 2: Accumulate the product:
            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    product[i][j] += A_frag[i]*B_frag[j];
        }
    }
    // Now update C
#pragma unroll 4
    for(int i = 0; i < 2; i++)
      for(int j = 0; j < 2; j++)
        for(int l = 0; l < 2; l++)
          for(int m = 0; m < 2; m++)
          {
              if((block_root_i0 + tile_root_i + 16*i + l < M) &&
                 (block_root_j0 + tile_root_j + 8*j + m < N))
              {
                  C[(block_root_i0 + tile_root_i + 16*i + l) 
                    + M*(block_root_j0 + tile_root_j + 8*j + m)] =
                    alpha*product[2*i + l][2*j + m] +
                    (IncludeOffset ? 
                    beta *
                    C[(block_root_i0 + tile_root_i + 16*i + l) 
                      + M*(block_root_j0 + tile_root_j + 8*j + m)] : 0);
              }

          }
}


/**
 * \brief Kernel for in-place GEMM opration.
 *
 * See myGEMM for more details. We have used this Naive GEMM implementation for
 * the time being. The operation PostOp::func is applied to each element in C
 * after the product has been found. In some cases, the offset is not necessary
 * so we use set the IncludeOffset to false so we do not have to waste a memory
 * access to C. Furthermore, we have additional template arguments which tell us
 * whether to transpose matrix A before multiplication and whether to transpose
 * matrix B before multiplication.
 *
 * arguments:
 *     A, B, C, alpha, beta: Matrices and scalars in the general GEMM framework:
 *     C <- alpha*A*B + beta*C
 *
 *     M: Number of rows in A and C
 *     N: Number of columns in C and B
 *     K: Number of columns in A and rows in B
 */
template<class PostOp, bool IncludeOffset, bool TransposeA, bool TransposeB>
__global__ void myGEMM_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N, int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;

            double product = 0;
            double old_val = IncludeOffset ? C[M*curr_col + curr_row] : 0;
            double Apart, Bpart;

            for(int k = 0; k < K; k++)
            {
               Apart = (TransposeA ? A[curr_row*K + k] : A[k*M + curr_row]);
               Bpart = (TransposeB ? B[k*N + curr_col] : B[curr_col*K + k]);
               product += Apart * Bpart;
            }

            C[M*curr_col + curr_row] = PostOp::func(
                                           alpha*product + beta*old_val);
        }
    }
}


/**
 * \brief Kernel for Matrix multiplication with vector accumulator.
 *
 * We have used this Naive GEMM implementation for the time being. The operation
 * PostOp::func is applied to each element in C after the product has been
 * found.
 *
 * Arguments:
 *     A, B, C, v, alpha, beta: Matrices, vectors, and scalars in the framework:
 *     C <- alpha*A*B + beta*[ v v v ... v]
 *
 *     M: Number of rows in A, C and v
 *     N: Number of columns in C and B and the matrix [ v v v ... v ]
 *     K: Number of columns in A and rows in B
 */
template<class PostOp>
__global__ void GEMM_vector_kernel(double *A, double *B, double *C, double *v,
                                   int M, int N, int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;

            double product = 0;
            double old_val = v[curr_row];

            for(int k = 0; k < K; k++)
            {
               product += A[k*M + curr_row] * B[curr_col*K + k];
            }

            C[M*curr_col + curr_row] = 
                PostOp::func(product + old_val);
        }
    }
}

template<class PostOp>
__global__ void
__launch_bounds__(256)
GEMM_vector_shared_kernel(double *A, double *B, double *C, double *v,
                          int M, int N, int K)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    double old_val, product;

    __shared__ double A_submat[BLOCK_SIZE*(BLOCK_SIZE+1)];
    __shared__ double B_submat[BLOCK_SIZE*(BLOCK_SIZE+1)];


    product  = 0;
    old_val = v[row];
    
    for(int k0 = 0; k0 < K; k0 += blockDim.x)
    {
        __syncthreads();
        // Step 1: Update A_submat:
        A_submat[threadIdx.x + (blockDim.x+1)*threadIdx.y] 
            = ((k0 + threadIdx.y < K) ?
               ((row < M) ? A[row + M*(k0 + threadIdx.y)] : 0) : 0);
    
        // Step 2: Update B_submat:
        B_submat[threadIdx.x + (blockDim.x+1)*threadIdx.y]
            = ((k0 + threadIdx.x < K) ? 
            ((col < N) ? B[(k0 + threadIdx.x) + K*col] : 0) : 0);

        __syncthreads();
        
        // Step 3: Accumulate the results:
        int num_iters = min(K - k0, blockDim.x);
        for(int k = 0; k < num_iters; k++) 
        {
            product += A_submat[threadIdx.x + k*(blockDim.x+1)] *
                       B_submat[k + threadIdx.y*(blockDim.x+1)];
        }
    }
    if((row < M) && (col < N))
        C[row + M*col] = PostOp::func(product + old_val);
}


template<class PostOp>
__global__ void 
__launch_bounds__(256)
GEMM_vector_tile_kernel(double *A, double *B, double *C, double *v,
                                   int M, int N, int K)
{
    int block_root_i0 = 64*blockIdx.x;
    int block_root_j0 = 64*blockIdx.y;

    __shared__ double A_submat[65][4];
    __shared__ double B_submat[5][64];

    double product[4][4];
    double A_frag[4];
    double B_frag[4];

    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            product[i][j] = 0;

    int tile_root_i = (threadIdx.x/32) * 32 + 2*(threadIdx.x % 8);
    int tile_root_j = threadIdx.y*16 + ((threadIdx.x % 32)/8) * 2;

    for(int k0 = 0; k0 < K; k0 += 4)
    {

        // Update the shared memory
        __syncthreads();
        A_submat[threadIdx.x][threadIdx.y] = 
          ((block_root_i0 + threadIdx.x < M) ? ((k0 + threadIdx.y < K) ?
           A[(block_root_i0 + threadIdx.x) + M*(k0 + threadIdx.y)] : 0): 0);

        B_submat[threadIdx.x % 4][threadIdx.x/4 + 16*threadIdx.y] =
          ((block_root_j0 + threadIdx.x/4 + 16*threadIdx.y < N) ? 
          ((k0 + (threadIdx.x % 4) < K) ?
           B[(k0 + (threadIdx.x % 4)) + K*(block_root_j0 + threadIdx.x/4 +
               16*threadIdx.y)] : 0) : 0);
        __syncthreads();


        // Loop through the tiles accumulating the product at each
#pragma unroll 3
        for(int k = 0; k < 4; k++)
        {
            // Step 1: Copy the fragments of A and B for this thread.
            for(int i = 0; i < 2; i++)
                for(int l = 0; l < 2; l++)
                {
                    A_frag[2*i + l] = A_submat[tile_root_i + 16*i + l][k];
                    B_frag[2*i + l] = B_submat[k][tile_root_j +  8*i + l];
                }
            // Step 2: Accumulate the product:
            for(int i = 0; i < 4; i++)
                for(int j = 0; j < 4; j++)
                    product[i][j] += A_frag[i]*B_frag[j];
        }
    }
    // Now update C
#pragma unroll 4
    for(int i = 0; i < 2; i++)
      for(int j = 0; j < 2; j++)
        for(int l = 0; l < 2; l++)
          for(int m = 0; m < 2; m++)
          {
              if((block_root_i0 + tile_root_i + 16*i + l < M) &&
                 (block_root_j0 + tile_root_j + 8*j + m < N))
              {
                  C[(block_root_i0 + tile_root_i + 16*i + l) 
                    + M*(block_root_j0 + tile_root_j + 8*j + m)] =
                    PostOp::func(
                    product[2*i + l][2*j + m] +
                    v[(block_root_i0 + tile_root_i + 16*i + l)]); 
              }

          }
}

template<class PostOp, bool IncludeOffset, bool TransposeA, bool TransposeB,
         bool Vector>
void smart_GEMM_wrapper(double *A, double *B, double *C, double *v,
                        double alpha, double beta,
                        int M, int N, int K)
{
    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize  ((M + BLOCK_SIZE - 1)/BLOCK_SIZE, 
                    (N + BLOCK_SIZE - 1)/BLOCK_SIZE);
    if(Vector) 
    {
        if((M < 64) || (N < 64))
        {
            GEMM_vector_shared_kernel<PostOp><<<gridSize, blockSize>>>(
                    A, B, C, v, M, N, K);
        } else {
            blockSize.x = 64;
            blockSize.y = 4;
            gridSize.x  = (M + 63)/64;
            gridSize.y  = (N + 63)/64;
            GEMM_vector_tile_kernel<PostOp><<<gridSize, blockSize>>>(
                    A, B, C, v, M, N, K);
        }

    } else {
        if(false)
        {
            myGEMM_shared_kernel<IncludeOffset, TransposeA, 
                           TransposeB><<<gridSize, blockSize>>>(
                                   A, B, C, alpha, beta, M, N, K);
        } else {
            blockSize.x = 64;
            blockSize.y = 4;
            gridSize.x  = (M + 63)/64;
            gridSize.y  = (N + 63)/64;
            myGEMM_tile_kernel<IncludeOffset, TransposeA, 
                               TransposeB><<<gridSize, blockSize>>>(
                                   A, B, C, alpha, beta, M, N, K);
        }

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

    smart_GEMM_wrapper<Identity, true, false, false, false>(
            A, B, C, NULL, *alpha, *beta, M, N, K);

    check_launch("myGEMM_kernel");

    return 0;
}




/******************************************************************************\
 * Section 3: Feed Forward Special Functions                                  *
\******************************************************************************/


/**
 * \brief Kernel for finding the softmax
 *
 * Although there's likely a smarter algorithm than this, we know that the
 * matrix we will be processing only has 10 rows for this use case, so we can
 * have a thread iterate through each row without too much of an issue.
 *
 * Furthermore, we store our results back in Z2 because we no longer need Z2
 * after we take the softmax, only yhat is neccesary in future steps.
 *
 * Z2 has L rows and N columns.
 *
 */
__global__ void softmax_kernel(double *Z2, int L, int N)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;


    // Number of columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_cols = (N - col + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);
    int curr_col;

    for(int c = 0; c < num_cols; c++)
    {
        curr_col = col + c*blockDim.x*gridDim.x;

        double sum = 0;
        for(int i = 0; i < L; i++)
            sum += exp(Z2[curr_col*L + i]);

        for(int i = 0; i < L; i++)
            Z2[curr_col*L + i] = exp(Z2[curr_col*L + i]) / sum;
    }

}

/**
 * \brief Wrapper to perform the feed-forward operation on device memory.
 *
 * arguments:
 *     d - A deviceCache (see gpu_func.h) containing pointers to all of the
 *         relevant neural network parameters.
 *     X - A pointer to a matrix of training images.
 *     N - The number of images (i.e. columns in X)
 */
int myFeedForward(deviceCache &d, double* X, int N)
{
    double *A1 = d.A1;
    double *W1 = d.W1;
    double *b1 = d.b1;

    double *A2 = d.A2;
    double *W2 = d.W2;
    double *b2 = d.b2;

    int K = d.K;
    int L = d.L;
    int M = d.M;

    // Step 1: First layer. We want to compute A1 which is M by N
    dim3 blockSize (64, 4);
    dim3 gridSize ((M + 63)/64, (N + 63)/64);

    // Use our vector-accumulating GEM to compute A1
    smart_GEMM_wrapper<Sigmoid, true, false, false, true>(
            W1, X, A1, b1, 1, 1, M, N, K);

    // Step 2a: Second layer. We want to compute A2. We start by storing Z2 in
    // the space of A2
    smart_GEMM_wrapper<Identity, true, false, false, true>(
            W2, A1, A2, b2, 1, 1, L, N, M);

    // Step 2b: Now we want to apply the softmax kernel to Z2 (which is stored
    // in A2) to get the correct A2 = yhat.
    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x  = std::min((int)((N + blockSize.x - 1)/blockSize.x),
                           MAX_GRID_SIZE);
    gridSize.y  = 1;
    softmax_kernel<<<gridSize, blockSize>>>(A2, L, N);

    return 0;
}

/**
 * \brief Computes the difference yhat - y in the back propogation and stores
 * the result in yhat.
 *  
 * Use blocks of 256x1 for this kernel.
 * 
 * arguments:
 *     yhat - The result of the neural network feed forward.
 *     y    - The correct results for each image
 *     L    - The number of rows in y and yhat
 *     N    - The number of columns in y and yhat
 */
__global__
void backPropDiff_kernel(double *yhat, double *y, int L, int N)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // Number of columns this thread must compute so that the whole matrix ends
    // up getting computed.
    int num_cols = (N - col + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int curr_col;

    for(int c = 0; c < num_cols; c++)
    {
        curr_col = col + c*blockDim.x*gridDim.x;
        for(int i = 0; i < L; i++)
            yhat[curr_col*L + i] = (yhat[curr_col*L + i] 
                                  - y[curr_col*L + i])/N;
    }
}


/**
 * \brief Scales a vector by an integer scale.
 *
 * This kernel is useful for scaling the gradients of b1 and b2 before they are
 * accumulated with gradients from other processes.
 * 
 * arguments:
 *     v     - pointer to the vector to scale
 *     M     - the number of rows in v
 *     scale - the integer to scale v by
 */
__global__
void vector_scale_kernel(double *v, int M, int scale)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    // Number of rows this thread must compute so that the whole matrix ends up
    // getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);
    int curr_row;
    
    for(int c = 0; c < num_rows; c++)
    {
        curr_row = row + c*blockDim.x*gridDim.x;
        v[curr_row] *= scale;
    }
}


/**
 * \brief Scales a matrix by an integer scale.
 *
 * This kernel is useful for scaling the gradients of W1 and W2 before they are
 * accumulated with gradients from other processes.
 * 
 * arguments:
 *     A     - pointer to the matrix to scale
 *     M     - the number of rows in A
 *     N     - the number of rows in A
 *     scale - the integer to scale v by
 */
__global__
void matrix_scale_kernel(double *A, int M, int N, int scale)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;
            A[curr_row + M*curr_col] *= scale;
        }
    }
}

/**
 * \brief Kernel for summing by row to get a column vector in the first column
 * of the array A. Does only part of the sum: must be called multiply times (see
 * myRowSum below) to compute the row sum.
 *
 * arguments:
 *     A        - the array to sum the rows of
 *     M        - the number of rows in A
 *     N        - the number of columns in A
 *     stride   - the stride between adjacent terms summed by a single thread
 *     num_iter - the number of terms for each thread to sum
 */
__global__
void myRowSum_kernel(double *A, int M, int N, int stride, int num_iter)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    // The number of rows that this thread must compute to ensure all rows get
    // covered.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);
    int curr_row;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;

        // Compute the first and last column that this column includes in the
        // sum.
        int start_col = (blockIdx.y*blockDim.y + threadIdx.y)*stride*num_iter;
        int end_col   = (blockIdx.y*blockDim.y + threadIdx.y+1)*stride*num_iter;

        // Sum from the last column to the first with the appropriate stride.
        double sum = 0; 
        for(int col = end_col-stride; col >= start_col; col -= stride)
        {
            if(col < N)
                sum += A[col*M + curr_row];
        }
        if(start_col < N)
            A[start_col*M + curr_row] = sum;
    }
}


/**
 * \brief Kernel which performs the special Hadamard product present in the back
 * propogation.
 *
 * Note that dZ1 is stored in the memory occupied initially by dA1.
 *
 * arguments:
 *     dA1 - dA1 in the Hadamard product
 *     A1  - A1 in the Hadamard product
 *     M   - the number of rows in dA1 and A1
 *     N   - the number of columns in dA1 and A1
 */
__global__
void mySpecialHadamard_kernel(double *dA1, double *A1, 
                              int M, int N) 
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;
            double A1_elem   =  A1[M*curr_col + curr_row];
            dA1[M*curr_col + curr_row] = dA1[M*curr_col + curr_row]
                                       * A1_elem * (1 - A1_elem);
        }
    }
}


/**
 * \brief Copies a matrix B to the area pointed to by A.
 *
 * arguments:
 *     A - the place to copy to
 *     B - the matrix to copy from
 *     M - the number of rows in A and B
 *     N - the number of columns in A and B
 */
__global__
void onDeviceCopy_kernel(double *A, double *B, int M, int N)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;
            A[curr_col*M + curr_row] = B[curr_col*M + curr_row];
        }
    }
}

/**
 * \brief Wrapper which copies a matrix from B to A using onDeviceCopy_kernel.
 * 
 * arguments:
 *     A - the place to copy to
 *     B - the matrix to copy from
 *     M - the number of rows in A and B
 *     N - the number of columns in A and B
 */
void onDeviceCopy(double *A, double *B, int M, int N)
{
    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize (std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE),
                   std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE));

    onDeviceCopy_kernel<<<gridSize,blockSize>>>(A, B, M, N);
    //check_launch("onDeviceCopy kernel");
}



/**
 * \brief Sums by row to get a column vector in the first column of the array A.
 *
 * Uses myRowSum_kernel repeatedly to compute the row sum.
 *
 * arguments:
 *     A - matrix to sum the rows of
 *     M - number of rows in A
 *     N - number of columns in A
 */
void myRowSum(double *A, double *out, int M, int N)
{
    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);

    // Define num_iters to ensure that our gridsize is not larger than the max
    // possible.
    int num_iters = std::max(4, (N + MAX_GRID_SIZE - 1) / MAX_GRID_SIZE);
    int stride = 1;
    
    dim3 gridSize (std::min((int)((M + blockSize.x - 1)/blockSize.x),
                   MAX_GRID_SIZE),
                   (N + (blockSize.y*stride*num_iters) - 1) /
                        (blockSize.y*stride*num_iters));


    // Call myRowSum_kernel repeatedly until all values have been summed into
    // the first row. We need to adjust the gridSize as we go because the number
    // of columns we need to sum is reduced on each iteration.
    myRowSum_kernel<<<gridSize, blockSize>>>(A, M, N, stride, num_iters);
    //check_launch("myRowSum_kernel");

    while (stride*num_iters < N)
    {
      stride *= num_iters;
      gridSize.y = (N + (blockSize.y*stride*num_iters) - 1) /
                        (blockSize.y*stride*num_iters);
      
      myRowSum_kernel<<<gridSize, blockSize>>>(A, M, N, stride, num_iters);
      //check_launch("myRowSum_kernel");
    }

    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x = std::min((int)((M + blockSize.x - 1)/blockSize.x),
                          MAX_GRID_SIZE);
    gridSize.y = 1;

    vector_scale_kernel<<<gridSize, blockSize>>>(A, M, N);
    //check_launch("vector_scale_kernel");

    onDeviceCopy_kernel<<<gridSize, blockSize>>>(out, A, M, 1);
    //check_launch("onDeviceCopy_kernel");
}









/**
 * \brief Function for carrying out the back propogation
 *
 * arguments:
 *     d   - a deviceCache (see gpu_func.h) containing pointers to all of the
 *           relevant neural network parameters
 *     X   - a pointer to a matrix of training images
 *     y   - a pointer to the matrix of image labels
 *     N   - the number of images (i.e. columns in X and y)
 *     reg - the regularization term
 */
int myBackPropogation(deviceCache &d, double *X, double *y, int N, double reg)
{
    // Set up aliases
    double *A1 = d.A1;
    double *W1 = d.W1;
    double *W2 = d.W2;
    double *dA1 = d.dA1;
    double *dW1 = d.dW1;
    double *dW2 = d.dW2;
    int L = d.L;
    int M = d.M;
    int K = d.K;

    double *diff = d.A2;
    double *dZ1 = d.dA1;

    // Step 1: Find the difference yhat - y
    dim3 blockSize (256, 1); 
    dim3 gridSize (std::min((int)((N + blockSize.x - 1)/blockSize.x),
                   MAX_GRID_SIZE), 1);

    backPropDiff_kernel<<<gridSize,blockSize>>>(diff, y, L, N);            
    //check_launch("backPropDiff kernel");

    // Step 2: Compute dW2
    onDeviceCopy(dW2, W2, L, M);

    // Include offset, transpose A1
    smart_GEMM_wrapper<Identity, true, false, true, false>(
            diff, A1, dW2, NULL, 1, reg, L, M, N);

    blockSize.x = BLOCK_SIZE;
    blockSize.y = BLOCK_SIZE;
    gridSize.x  = std::min((int)((L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((M + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);

    matrix_scale_kernel<<<gridSize, blockSize>>>(dW2, L, M, N);
    //check_launch("matrix_scale_kernel");

    // Step 3: Compute dA1
    // Do not include offset, transpose W2
    smart_GEMM_wrapper<Identity, false, true, false, false>(
        W2, diff, dA1, NULL, 1, 1, M, N, L);

    // Step 4: Compute db2
    myRowSum(diff, d.db2, L, N);

    // Step 5: Compute dZ1
    blockSize.x = BLOCK_SIZE;
    blockSize.y = BLOCK_SIZE;
    gridSize.x  = std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    mySpecialHadamard_kernel<<<gridSize, blockSize>>>(dZ1, A1, M, N);      
    //check_launch("mySpecialHadamard_kernel");

    // Step 6: Compute dW1
    onDeviceCopy(dW1, W1, M, K);
    smart_GEMM_wrapper<Identity, true, false, true, false>(
            dZ1, X, dW1, NULL, 1, reg, M, K, N);


    blockSize.x = BLOCK_SIZE;
    blockSize.y = BLOCK_SIZE;
    gridSize.x  = std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((K + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);

    matrix_scale_kernel<<<gridSize, blockSize>>>(dW1, M, K, N);

    // Step 7: Compute db1
    myRowSum(dZ1, d.db1, M, N);
    
    return 0;
}

/******************************************************************************\
 * Section 5: Gradient Descent Special Functions                              *
\******************************************************************************/

/**
 * \brief GPU kernel for performing the gradient descent update.
 *
 * The normalization factor allows us to renormalize the gradients after all of
 * our processes exchange and sum gradients. 
 *
 * arguments:
 *     W_or_b        - the parameter that we subtracting the gradient from
 *     grad          - the gradient that we are subtracting from W_or_b
 *     learning_rate - the learning rate
 *     M             - the number of rows in W_or_b
 *     N             - the number of columns in W_or_b
 *     normalization - a normalization integer to divide the gradeint by.
 */
__global__
void grad_descent_kernel(double *W_or_b, double *grad, double learning_rate,
                         int M, int N, int normalization)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    // Number of rows and columns this thread must compute so that the whole
    // matrix ends up getting computed.
    int num_rows = (M - row + (blockDim.x * gridDim.x - 1)) /
                   (blockDim.x * gridDim.x);

    int num_cols = (N - col + (blockDim.y * gridDim.y - 1)) /
                   (blockDim.y * gridDim.y);

    int curr_row, curr_col;

    for(int i = 0; i < num_rows; i++)
    {
        curr_row = row + i*blockDim.x*gridDim.x;
        for(int j = 0; j < num_cols; j++)
        {
            curr_col = col + j*blockDim.y*gridDim.y;
            W_or_b[curr_col*M + curr_row] -= learning_rate 
                * (grad[curr_col*M + curr_row]/normalization);
        }
    }
}

/**
 * \brief Performs the gradient descent update accelrated by the gpu.
 * 
 * arguments:
 *     d             - a deviceCache (see gpu_func.h) containing pointers to 
 *                     all of the
 *     N             - the number of images in the whole batch
 *     learning_rate - the learning rate for this training
 */
void myGradientDescent(deviceCache &d, double learning_rate, int N)
{
    // Step 1: Update W1
    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize  (std::min((int)((d.M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE),
                    std::min((int)((d.K + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE)); 
    grad_descent_kernel<<<gridSize, blockSize>>>(d.W1, d.dW1, learning_rate,
                                                 d.M, d.K, N);
    //check_launch("grad_descent_kernel");

    // Step 2: Update W2
    gridSize.x = std::min((int)((d.L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y = std::min((int)((d.M + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    grad_descent_kernel<<<gridSize, blockSize>>>(d.W2, d.dW2, learning_rate,
                                                 d.L, d.M, N);
    //check_launch("grad_descent_kernel");

    // Step 3: Update b1
    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x = std::min((int)((d.M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y = 1;
    grad_descent_kernel<<<gridSize, blockSize>>>(d.b1, d.db1, learning_rate,
                                                 d.M, 1, N);
    //check_launch("grad_descent_kernel");


    // Step 4: Update b2
    gridSize.x = std::min((int)((d.L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    grad_descent_kernel<<<gridSize, blockSize>>>(d.b2, d.db2, learning_rate,
                                                 d.L, 1, N);

    //check_launch("grad_descent_kernel");
}
