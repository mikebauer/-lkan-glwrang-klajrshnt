#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE (16)
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
    dim3 gridSize (std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE),
                   std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE));

    myGEMM_kernel<Identity, true, false, false><<<gridSize, blockSize>>>(
        A, B, C, *alpha, *beta, M, N, K);

    check_launch("myGEMM_kernel");

    return 0;
}

/******************************************************************************\
 * Section 3: Feed Forward Special Functions                                  *
\******************************************************************************/

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
    dim3 blockSize (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize (std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE),
                   std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE));


    // Use our vector-accumulating GEM to compute A1
    GEMM_vector_kernel<Sigmoid><<<gridSize, blockSize>>>(
        W1, X, A1, b1, M, N, K);

    check_launch("GEMM_vector_kernel layer1");

    // Step 2a: Second layer. We want to compute A2. We start by storing Z2 in
    // the space of A2
    gridSize.x  = std::min((int)((L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    GEMM_vector_kernel<Identity><<<gridSize, blockSize>>>(
        W2, A1, A2, b2, L, N, M);
    check_launch("GEMM_vector_kernel layer2");

    // Step 2b: Now we want to apply the softmax kernel to Z2 (which is stored
    // in A2) to get the correct A2 = yhat.
    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x  = std::min((int)((N + blockSize.x - 1)/blockSize.x),
                           MAX_GRID_SIZE);
    gridSize.y  = 1;
    softmax_kernel<<<gridSize, blockSize>>>(A2, L, N);
    check_launch("softmax_kernel");

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
    check_launch("onDeviceCopy kernel");
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
    check_launch("myRowSum_kernel");

    while (stride*num_iters < N)
    {
      stride *= num_iters;
      gridSize.y = (N + (blockSize.y*stride*num_iters) - 1) /
                        (blockSize.y*stride*num_iters);
      
      myRowSum_kernel<<<gridSize, blockSize>>>(A, M, N, stride, num_iters);
      check_launch("myRowSum_kernel");
    }

    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x = std::min((int)((M + blockSize.x - 1)/blockSize.x),
                          MAX_GRID_SIZE);
    gridSize.y = 1;

    vector_scale_kernel<<<gridSize, blockSize>>>(A, M, N);
    check_launch("vector_scale_kernel");

    onDeviceCopy_kernel<<<gridSize, blockSize>>>(out, A, M, 1);
    check_launch("onDeviceCopy_kernel");
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
    check_launch("backPropDiff kernel");

    // Step 2: Compute dW2
    onDeviceCopy(dW2, W2, L, M);
    blockSize.x = BLOCK_SIZE;
    blockSize.y = BLOCK_SIZE;
    gridSize.x  = std::min((int)((L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((M + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);

    // Include offset, transpose A1
    myGEMM_kernel<Identity, true, false, true><<<gridSize, blockSize>>>(
        diff, A1, dW2, 1, reg, L, M, N);                                   
    check_launch("myGEMM_kernel");

    matrix_scale_kernel<<<gridSize, blockSize>>>(dW2, L, M, N);
    //check_launch("matrix_scale_kernel");

    // Step 3: Compute dA1
    gridSize.x  = std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((N + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    // Do not include offset, transpose W2
    myGEMM_kernel<Identity, false, true, false><<<gridSize, blockSize>>>(
        W2, diff, dA1, 1, 1, M, N, L);                                     
    check_launch("myGEMM_kernel");

    // Step 4: Compute db2
    myRowSum(diff, d.db2, L, N);

    // Step 5: Compute dZ1
    // Gridsize is still fine
    mySpecialHadamard_kernel<<<gridSize, blockSize>>>(dZ1, A1, M, N);      
    check_launch("mySpecialHadamard_kernel");

    // Step 6: Compute dW1
    onDeviceCopy(dW1, W1, M, K);
    gridSize.x  = std::min((int)((M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y  = std::min((int)((K + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    // Include offset, transpose X
    myGEMM_kernel<Identity, true, false, true><<<gridSize, blockSize>>>(
        dZ1, X, dW1, 1, reg, M, K, N);                                    
    check_launch("myGEMM_kernel");

    matrix_scale_kernel<<<gridSize, blockSize>>>(dW1, M, K, N);
    //check_launch("matrix_scale_kernel");

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
    check_launch("grad_descent_kernel");

    // Step 2: Update W2
    gridSize.x = std::min((int)((d.L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y = std::min((int)((d.M + blockSize.y - 1)/blockSize.y), MAX_GRID_SIZE);
    grad_descent_kernel<<<gridSize, blockSize>>>(d.W2, d.dW2, learning_rate,
                                                 d.L, d.M, N);
    check_launch("grad_descent_kernel");

    // Step 3: Update b1
    blockSize.x = 256;
    blockSize.y = 1;
    gridSize.x = std::min((int)((d.M + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    gridSize.y = 1;
    grad_descent_kernel<<<gridSize, blockSize>>>(d.b1, d.db1, learning_rate,
                                                 d.M, 1, N);
    check_launch("grad_descent_kernel");


    // Step 4: Update b2
    gridSize.x = std::min((int)((d.L + blockSize.x - 1)/blockSize.x), MAX_GRID_SIZE);
    grad_descent_kernel<<<gridSize, blockSize>>>(d.b2, d.db2, learning_rate,
                                                 d.L, 1, N);

    check_launch("grad_descent_kernel");
}
