#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
#include "linear.cuh"

// ----------------------------------------------------------------------------
// GPU kernels

// copy the matmul forward and backward kernels from llm.c

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a cuBLAS function that does this
__global__ void add_bias(float* out, const float* bias, int N, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// kernel 2 calls cuBLAS, which should be very efficient
void matmul_forward2(cublasHandle_t cublas_handle,
                     float* out,
                     const float* inp, const float* weight, const float* bias,
                     int N, int C, int OC,
                     const int block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, N, C, &alpha, weight, C, inp, C, &beta, out, OC));
    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int grid_size = ceil_div(OC * N, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, N, OC);
        cudaCheck(cudaGetLastError());
    }
}

// use shared memory and coarsening + reductions
__global__ void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int N, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < N; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = shared[0];
    }
}

// version1: simple cuBLAS calls
void matmul_backward1(cublasHandle_t cublas_handle,
                      float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight,
                      int N, int C, int OC) {
    float alpha = 1.0f;
    float beta = 0.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // backward to input
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, N, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));
    // backward to weight
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, N, &alpha, inp, C, dout, OC, &beta, dweight, C));
    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        dim3 block_dim(block_size);
        dim3 grid_dim(OC);
        size_t shared_mem_size = block_size * sizeof(float);
        matmul_backward_bias_kernel_faster<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, N, OC);
    }
}

// ----------------------------------------------------------------------------

void linear_set_param_ptrs(
    LinearParams* params,
    float* params_memory,
    int C, int OC
) {
    params->w = params_memory;
    params_memory += OC * C;
    params->b = params_memory;
}


#ifndef LINKING
int main(int argc, char **argv) {
    int N = 32;
    int C = 64;
    int OC = 128;

    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));


    // create host memory
    float *inp = (float*)malloc(N * C * sizeof(float));
    float *weight = (float*)malloc(OC * C * sizeof(float));
    float *bias = (float*)malloc(OC * sizeof(float));
    float *out = (float*)malloc(N * OC * sizeof(float));
    float *dout = (float*)malloc(N * OC * sizeof(float));
    float *dinp = (float*)malloc(N * C * sizeof(float));
    float *dweight = (float*)malloc(OC * C * sizeof(float));
    float *dbias = (float*)malloc(OC * sizeof(float));

    FILE *file = fopen("linear.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(inp, sizeof(float), N * C, file);
    freadCheck(weight, sizeof(float), OC * C, file);
    freadCheck(bias, sizeof(float), OC, file);
    freadCheck(out, sizeof(float), N * OC, file);
    freadCheck(dout, sizeof(float), N * OC, file);
    freadCheck(dinp, sizeof(float), N * C, file);
    freadCheck(dweight, sizeof(float), OC * C, file);
    freadCheck(dbias, sizeof(float), OC, file);
    fclose(file);


    // allocate device memory
    float *d_inp, *d_weight, *d_bias, *d_out, *d_dout, *d_dinp, *d_dweight, *d_dbias;
    cudaCheck(cudaMalloc(&d_inp, N * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dinp, N * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, OC * sizeof(float)));

    cudaCheck(cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * OC * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {128, 256, 512, 1024};

    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("\nBlock size: %d\n", block_size);
        matmul_forward2(cublas_handle, d_out, d_inp, d_weight, d_bias, N, C, OC, block_size);
        validate_result(d_out, out, "out", N * OC);
    }

    printf("Forward pass successful\n");

    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("\nBlock size: %d\n", block_size);
        matmul_backward1(cublas_handle, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, N, C, OC);
        printf("Checking dbias\n");
        validate_result(d_dbias, dbias, "dbias", OC);
        printf("Checking dinp\n");
        validate_result(d_dinp, dinp, "dinp", N * C);
        printf("Checking dweight\n");
        validate_result(d_dweight, dweight, "dweight", OC * C);
    }
    printf("Backward pass successful\n");

    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward2,
                                              cublas_handle, d_out, d_inp, d_weight, d_bias,
                                              N, C, OC, block_size);

        float tflops = (float)N * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    }

    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_backward1,
                                              cublas_handle, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                                              N, C, OC);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(inp);
    free(weight);
    free(bias);
    free(out);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);

    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));

    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
#endif