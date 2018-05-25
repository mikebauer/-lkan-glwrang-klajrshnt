#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);


class deviceCache {
  public:
    double *X, *y;
    double *A1, *W1, *b1;
    double *A2, *W2, *b2;
    double *dA1, *dW1, *dW2;

    int K, L, M, N;

    deviceCache(const double *host_X, const double *host_y, 
                int _K, int _L, int _M, int _N, int batch_size)
        : K(_K), L(_L), M(_M), N(_N)
    {
        // Allocate all of the cache space
        checkCudaErrors(
            cudaMalloc((void **)&X, K * N * sizeof(double)));
        checkCudaErrors(
            cudaMalloc((void **)&y, L * N * sizeof(double)));

        checkCudaErrors(
            cudaMalloc((void **)&A1, M * batch_size * sizeof(double)));
        checkCudaErrors(
            cudaMalloc((void **)&dA1, M * batch_size * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&W1, M * K * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&dW1, M * K * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&b1, M * sizeof(double)));

        checkCudaErrors(
            cudaMalloc((void **)&A2, L * batch_size * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&W2, L * M * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&dW2, L * M * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&b2, L * sizeof(double)));

        // Load the host X and y data
        checkCudaErrors(
            cudaMemcpy(X, host_X, sizeof(double) * K * N,
                    cudaMemcpyHostToDevice));

        checkCudaErrors(
            cudaMemcpy(y, host_y, sizeof(double) * L * N,
                    cudaMemcpyHostToDevice));
    }

    ~deviceCache()
    {
        cudaFree(X);
        cudaFree(y); 
        cudaFree(A1); 
        cudaFree(W1); 
        cudaFree(b1); 
        cudaFree(A2); 
        cudaFree(W2); 
        cudaFree(b2); 
        cudaFree(dA1); 
        cudaFree(dW1); 
        cudaFree(dW2); 
    }
};

int myFeedForward(deviceCache &d, double* X, int N);

void myRowSum(double *A, int M, int N);







#endif
