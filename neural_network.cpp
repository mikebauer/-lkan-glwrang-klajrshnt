#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include <cuda_runtime.h> // TODO

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}


void update_nn_from_deviceCache(NeuralNetwork& nn, deviceCache& dCache)
{
    checkCudaErrors(
        cudaMemcpy(nn.W[0].memptr(),
                   dCache.W1, sizeof(double)*dCache.M*dCache.K,
                   cudaMemcpyDeviceToHost));

    checkCudaErrors(
        cudaMemcpy(nn.b[0].memptr(),
                   dCache.b1, sizeof(double)*dCache.M,
                   cudaMemcpyDeviceToHost));

    checkCudaErrors(
        cudaMemcpy(nn.W[1].memptr(),
                   dCache.W2, sizeof(double)*dCache.L*dCache.M,
                   cudaMemcpyDeviceToHost));

    checkCudaErrors(
        cudaMemcpy(nn.b[1].memptr(),
                   dCache.b2, sizeof(double)*dCache.L,
                   cudaMemcpyDeviceToHost));
}


void update_deviceCache_from_nn(NeuralNetwork& nn, deviceCache& dCache)
{
    checkCudaErrors(
        cudaMemcpy(dCache.W1, nn.W[0].memptr(), sizeof(double) * dCache.M * 
        dCache.K, cudaMemcpyHostToDevice)); 
    checkCudaErrors(
        cudaMemcpy(dCache.b1, nn.b[0].memptr(), sizeof(double) * dCache.M,
        cudaMemcpyHostToDevice)); 
    checkCudaErrors(
        cudaMemcpy(dCache.W2, nn.W[1].memptr(), sizeof(double) * dCache.L * 
        dCache.M, cudaMemcpyHostToDevice)); 
    checkCudaErrors(
        cudaMemcpy(dCache.b2, nn.b[1].memptr(), sizeof(double) * dCache.L,
        cudaMemcpyHostToDevice)); 
}


/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    int K = nn.H[0];
    int M = nn.H[1];
    // N already defined
    int L = nn.H[2];

    deviceCache dCache (K, L, M, N, batch_size);
    update_deviceCache_from_nn(nn, dCache);

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    double *dW1;// = (double *)malloc(M*K*sizeof(double));
    double *db1;// = (double *)malloc(M*sizeof(double));
    double *dW2;// = (double *)malloc(L*M*sizeof(double));
    double *db2;// = (double *)malloc(L*sizeof(double));

    double *dW1_g;// = (double *)malloc(M*K*sizeof(double));
    double *db1_g;// = (double *)malloc(M*sizeof(double));
    double *dW2_g;// = (double *)malloc(L*M*sizeof(double));
    double *db2_g;// = (double *)malloc(L*sizeof(double));

    checkCudaErrors(cudaMallocHost((void **)&dW1, M*K*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&db1, M*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&dW2, L*M*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&db2, L*sizeof(double)));

    checkCudaErrors(cudaMallocHost((void **)&dW1_g, M*K*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&db1_g, M*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&dW2_g, L*M*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&db2_g, L*sizeof(double)));

    // Find the maximum possible number of images per process per batch. 
    int N_proc = (batch_size + (num_procs) - 1)/num_procs;

    double *X_proc;// = (double *)malloc(K*N_proc*sizeof(double));
    double *y_proc;// = (double *)malloc(L*N_proc*sizeof(double));
    checkCudaErrors(cudaMallocHost((void **)&X_proc, K*N_proc*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void **)&y_proc, L*N_proc*sizeof(double)));

    double *dX_proc;
    double *dy_proc;

    checkCudaErrors(cudaMalloc((void **)&dX_proc, K*N_proc*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dy_proc, L*N_proc*sizeof(double)));

    int X_sendcounts[num_procs];
    int X_displs[num_procs];
    int y_sendcounts[num_procs];
    int y_displs[num_procs];

    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            
            // Find the X and y values from this batch 
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            int N_batch  = std::min((batch + 1)*batch_size, N) 
                         - batch*batch_size;

            double *X_batch = (double *)X.memptr() + batch*batch_size*K;
            double *y_batch = (double *)y.memptr() + batch*batch_size*L;

            // Find the max number of images per proc
            N_proc = (N_batch + (num_procs - 1)) / 
                      num_procs;

            // Find the sendcounts and displacements for each process
            for(int i = 0; i < num_procs; i++)
            {
                X_sendcounts[i] = (std::min((i + 1) * N_proc, N_batch) 
                                - i*N_proc)*K;
                y_sendcounts[i] = (std::min((i + 1) * N_proc, N_batch) 
                                - i*N_proc)*L;
                X_displs[i] = i*N_proc*K;
                y_displs[i] = i*N_proc*L;
            }
            // Find N_proc for this process
            N_proc = std::min((rank + 1) * N_proc, N_batch) - rank*N_proc;

            // Scatter this batch of X and y values to all processes
            MPI_Scatterv(X_batch, X_sendcounts, X_displs, MPI_DOUBLE, X_proc, 
                         X_sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Scatterv(y_batch, y_sendcounts, y_displs, MPI_DOUBLE, y_proc, 
                         y_sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Copy this batches X and y values to the device
            checkCudaErrors(cudaMemcpy(dX_proc, X_proc, K*N_proc*sizeof(double),
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMemcpy(dy_proc, y_proc, L*N_proc*sizeof(double),
                                       cudaMemcpyHostToDevice));

            // Feed forward and back propogate on the device
            myFeedForward(dCache, dX_proc, N_proc); 
            myBackPropogation(dCache, dX_proc, dy_proc, N_proc, reg);

            // Copy the gradients back to the host
            checkCudaErrors(cudaMemcpy(dW1, dCache.dW1, M*K*sizeof(double), 
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(db1, dCache.db1, M*sizeof(double),   
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(dW2, dCache.dW2, L*M*sizeof(double), 
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(db2, dCache.db2, L*sizeof(double),   
                            cudaMemcpyDeviceToHost));

            // Allreduce the gradients
            MPI_SAFE_CALL(MPI_Allreduce(dW1, dW1_g, M*K, MPI_DOUBLE, MPI_SUM, 
                                        MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(db1, db1_g, M,   MPI_DOUBLE, MPI_SUM, 
                                        MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(dW2, dW2_g, L*M, MPI_DOUBLE, MPI_SUM, 
                                        MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(db2, db2_g, L,   MPI_DOUBLE, MPI_SUM, 
                                        MPI_COMM_WORLD));

            // Copy the reduced gradients back to the device
            checkCudaErrors(cudaMemcpy(dCache.dW1, dW1_g, M*K*sizeof(double), 
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dCache.db1, db1_g, M*sizeof(double),   
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dCache.dW2, dW2_g, L*M*sizeof(double), 
                                       cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dCache.db2, db2_g, L*sizeof(double),   
                                       cudaMemcpyHostToDevice));

            // Perform gradient descent on the device
            myGradientDescent(dCache, learning_rate, N_batch);

            // Update the host data in the neural network on proc 0 
            if(rank == 0)
                update_nn_from_deviceCache(nn, dCache);

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the
             * arma matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    checkCudaErrors(cudaFreeHost(X_proc));
    checkCudaErrors(cudaFreeHost(y_proc));
    checkCudaErrors(cudaFreeHost(dW1));
    checkCudaErrors(cudaFreeHost(db1));
    checkCudaErrors(cudaFreeHost(dW2));
    checkCudaErrors(cudaFreeHost(db2));
    checkCudaErrors(cudaFreeHost(dW1_g));
    checkCudaErrors(cudaFreeHost(db1_g));
    checkCudaErrors(cudaFreeHost(dW2_g));
    checkCudaErrors(cudaFreeHost(db2_g));

    error_file.close();
}
