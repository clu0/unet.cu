#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
#include "conv2d_k1.cuh"



__global__ void permute_x_kernel(
    const float* x_in,
    float* x_out,
    int B, int C_in, int H, int W
) {
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx < B * C_in * H * W) {
        int img_size = H * W;
        int b = flat_idx / (C_in * img_size);
        int c = (flat_idx / img_size) % C_in;
        int h = (flat_idx / W) % H;
        int w = flat_idx % W;

        // load from adjacent entries in x_in to enable coalescing
        int out_ind = c * B * img_size + b * img_size + h * W + w;
        x_out[out_ind] = x_in[flat_idx]; 
    }
}


__global__ void permute_out_add_bias_kernel(
    const float* out_perm, const float* bias,
    float* out,
    int B, int C_out, int H, int W
) {
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx < B * C_out * H * W) {
        int img_size = H * W;
        int c = flat_idx / (B * img_size);
        int b = (flat_idx / img_size) % B;
        int h = (flat_idx / W) % H;
        int w = flat_idx % W;

        int out_ind = b * C_out * img_size + c * img_size + h * W + w;
        float val = out_perm[flat_idx];
        if (bias != NULL) {
            val += bias[c];
        }
        out[out_ind] = val;
    }
}


// kernel that uses cuBLAS to do the 1D conv, then adds bias by hand
void conv2d_k1_forward1(
    cublasHandle_t cublas_handle,
    float* out,
    const float* x, const float* weight, const float* bias,
    int B, int C_in, int H, int W, int C_out,
    const int block_size,
    float* t1, float* t2, float* t3
) {
    // x shape (B, C_in, H, W)
    // weight (C_out, C_in, 1, 1), which is essentially (C_out, C_in)
    // bias (C_out,)
    // out (B, C_out, H, W)

    // cublas stores matrices in column-major order
    // so plan is as follows:
    // 1. permute x to shape (C_in, B, H, W)
    // 2. do x @ weight -> out with cublas, where out ends up with shape (C_out, B, H, W)
    // 3. permute out to shape (B, C_out, H, W), and add bias at the same time.

    // 1. allocate x_perm and permute x
    cudaEvent_t start, end1, end2, end3;
    if (t1 != nullptr) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end1));
        cudaCheck(cudaEventCreate(&end2));
        cudaCheck(cudaEventCreate(&end3));
        cudaCheck(cudaEventRecord(start, nullptr));
    }
    float *x_perm, *out_perm;
    cudaCheck(cudaMalloc(&x_perm, C_in * B * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&out_perm, C_out * B * H * W * sizeof(float)));
    int total_threads1 = B * C_in * H * W;
    int num_blocks1 = ceil_div(total_threads1, block_size);
    permute_x_kernel<<<num_blocks1, block_size>>>(x, x_perm, B, C_in, H, W);
    if (t1 != nullptr) {
        cudaCheck(cudaEventRecord(end1, nullptr));
    }

    // 2. x @ weight with cublas
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int non_C_dim = B * H * W;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, non_C_dim, C_out, C_in, &alpha, x_perm, non_C_dim, weight, C_in, &beta, out_perm, non_C_dim));
    if (t2 != nullptr) {
        cudaCheck(cudaEventRecord(end2, nullptr));
    }

    // 3. permute back and add bias
    int total_threads2 = B * C_out * H * W;
    int num_blocks2 = ceil_div(total_threads2, block_size);
    permute_out_add_bias_kernel<<<num_blocks2, block_size>>>(out_perm, bias, out, B, C_out, H, W);

    // free memory
    cudaCheck(cudaFree(x_perm));
    cudaCheck(cudaFree(out_perm));
    if (t3 != nullptr) {
        cudaCheck(cudaEventRecord(end3, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end1));
        cudaCheck(cudaEventSynchronize(end2));
        cudaCheck(cudaEventSynchronize(end3));
        float elapsed_time1, elapsed_time2, elapsed_time3;
        cudaCheck(cudaEventElapsedTime(&elapsed_time1, start, end1));
        cudaCheck(cudaEventElapsedTime(&elapsed_time2, end1, end2));
        cudaCheck(cudaEventElapsedTime(&elapsed_time3, end2, end3));
        *t1 += elapsed_time1;
        *t2 += elapsed_time2;
        *t3 += elapsed_time3;
    }
}

template <const int BO, const int BH, const int TO, const int TH>
__global__ void conv2d_k1_forward_kernel(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C, int H, int W, int O
) {
    // constants
    const int BC = 16;
    assert(O % BO == 0);
    assert(C % BC == 0);
    assert(H * W % BH == 0);
    // n_threads per block = (BO * BH) / (TO * TH) = 256
    // n_blocks = B * (O / BO) * (H * W / BH)
    // set gridDim to (BxHxW / BH, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = H * W / BH;
    assert(gridDim.x == B * blocks_per_col);
    const int block_col = blockIdx.x % blocks_per_col;
    const int batch = blockIdx.x / blocks_per_col;

    const int threads_per_col = BH / TH;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // shared mem for weight and x
    const int weight_s_sz = BO * BC;
    const int x_s_sz = BC * BH;
    __shared__ float w_s[weight_s_sz];
    __shared__ float x_s[x_s_sz];

    // local registers for results;
    float thread_results[TO * TH] = {0.0};
    float reg_w[TO] = {0.0};
    float reg_x[TH] = {0.0};

    // get offsets to x and weight
    int x_block_offset = batch * C * H * W + block_col * BH;
    int w_block_offset = block_row * BO * C;

    for (int c_idx = 0; c_idx < C; c_idx += BC) {
        int x_inner_offset = c_idx * H * W;
        int w_inner_offset = c_idx;
        // load weights to smem
        for (int i = threadIdx.x; i < weight_s_sz; i += blockDim.x) {
            int w_i = i / BC;
            int w_j = i % BC;
            int w_abs_idx = w_block_offset + w_inner_offset + w_i * C + w_j;
            w_s[w_i * BC + w_j] = weight[w_abs_idx];
        }
        for (int i = threadIdx.x; i < x_s_sz; i += blockDim.x) {
            int x_i = i / BH;
            int x_j = i % BH;
            int x_abs_idx = x_block_offset + x_inner_offset + x_i * H * W + x_j;
            x_s[x_i * BH + x_j] = x[x_abs_idx];
        }
        __syncthreads();


        // calculate thread results
        for (int dot_idx = 0; dot_idx < BC; dot_idx++) {
            for (int i = 0; i < TO; i++) {
                reg_w[i] = w_s[(thread_row * TO + i) * BC + dot_idx];
            }
            for (int j = 0; j < TH; j++) {
                reg_x[j] = x_s[dot_idx * BH + thread_col * TH + j];
            }
            for (int i = 0; i < TO; i++) {
                for (int j = 0; j < TH; j++) {
                    thread_results[i * TH + j] += reg_w[i] * reg_x[j];
                }
            }
        }
        __syncthreads();
    }

    // write out results
    // set out and bias directly to thread offset
    int out_block_offset = batch * O * H * W + block_row * BO * H * W + block_col * BH;
    int bias_offset = block_row * BO;
    for (int i = 0; i < TO; i++) {
        for (int j = 0; j < TH; j++) {
            int bias_idx = bias_offset + thread_row * TO + i;
            float out_val = thread_results[i * TH + j] + bias[bias_idx];
            int out_idx = out_block_offset + thread_row * TO * H * W + thread_col * TH + i * H * W + j;
            atomicAdd(out + out_idx, out_val);
        }
    }
}

void conv2d_k1_forward2(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int H, int W, int C_out
) {
    cudaCheck(cudaMemset(out, 0, B * C_out * H * W * sizeof(float)));
    const int BO = 64;
    const int BH = 64;
    const int TO = 4;
    const int TH = 4;
    dim3 gridDim(B * H * W / BH, C_out / BO);
    dim3 blockDim((BO * BH) / (TO * TH));
    conv2d_k1_forward_kernel<BO, BH, TO, TH><<<gridDim, blockDim>>>(x, weight, bias, out, B, C_in, H, W, C_out);
}



// --------------------------------------------------------------------------------
// backward pass


__global__ void permute_dout_x_kernel(
    const float* x, const float* dout,
    float* x_perm, float* dout_perm,
    int B, int C_in, int C_out, int H, int W
) {
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = H * W;
    int h = (flat_idx / W) % H;
    int w = flat_idx % W;

    if (flat_idx < B * C_in * H * W) {
        // permute x
        int b = flat_idx / (C_in * img_size);
        int c = (flat_idx / img_size) % C_in;

        int out_ind = c * B * img_size + b * img_size + h * W + w;
        x_perm[out_ind] = x[flat_idx];
    }

    if (flat_idx < B * C_out * H * W) {
        // permute dout
        int b = flat_idx / (C_out * img_size);
        int c = (flat_idx / img_size) % C_out;

        int out_ind = c * B * img_size + b * img_size + h * W + w;
        dout_perm[out_ind] = dout[flat_idx];
    }
}


__global__ void permute_dx_kernel(
    const float* dx_perm,
    float* dx,
    int B, int C_in, int H, int W
) {
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = H * W;
    if (flat_idx < B * C_in * img_size) {
        int c = flat_idx / (B * img_size);
        int b = (flat_idx / img_size) % B;
        int h = (flat_idx / W) % H;
        int w = flat_idx % W;
        int out_ind = b * C_in * img_size + c * img_size + h * W + w;
        dx[out_ind] = dx_perm[flat_idx];
    }
}


// we want to sum the B, H, W dimensions of dout
// since dout_perm is shaped as (C_out, B, H, W)
// we will have each block take care of one entry of C_out
// and the threads in each block will use coorperative_groups to do the sum
__global__ void dbias_kernel(
    const float* dout_perm, // shape (C_out, B, H, W)
    float* dbias,
    int B, int C_out, int H, int W
) {
    int bhw = B * H * W;
    extern __shared__ float shared[];

    // move dout_perm to the C_out index
    if (threadIdx.x < bhw) {
        dout_perm += blockIdx.x * B * H * W;
        float th_sum = 0.0f;
        for (int i = threadIdx.x; i < bhw; i += blockDim.x) {
            th_sum += dout_perm[i];
        }
        shared[threadIdx.x] = th_sum;
    }
    __syncthreads();

    // reduce the shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dbias[blockIdx.x] = shared[0];
    }
}


// uses cublas
void conv2d_k1_backward1(
    cublasHandle_t cublas_handle,
    const float* dout, const float* x, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C_in, int C_out, int H, int W,
    int block_size
) {
    // shapes:
    // x, dx: (B, C_in, H, W)
    // weight: (C_out, C_in)
    // bias: (C_out,)
    // dout: (B, C_out, H, W)

    // plan (the shapes are raw array shapes, note that cublas interprets the arrays as transposed):
    // 1. transpose x to x_perm of shape (C_in, B*H*W), and dout to dout_perm of shape (C_out, B*H*W)
    // 2. do one sgemm call to get dw = x.T @ dout_perm, and another call to get dx_perm = dout @ w.T
    // 3. dx_perm is of shape (C_in, B*H*W), so call a kernel to permute it
    // 4. reduce to get dbias
    float *x_perm, *dout_perm, *dx_perm;
    cudaCheck(cudaMalloc(&x_perm, C_in * B * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&dout_perm, C_out * B * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&dx_perm, C_in * B * H * W * sizeof(float)));

    // 1. permute x and dout
    int total_th1 = (C_in > C_out) ? C_in * B * H * W : C_out * B * H * W;
    int n_blk1 = ceil_div(total_th1, block_size);
    permute_dout_x_kernel<<<n_blk1, block_size>>>(x, dout, x_perm, dout_perm, B, C_in, C_out, H, W);

    // 2.a sgemm call for dw = x_perm.T @ dout_perm
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int non_C_dim = B * H * W;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C_in, C_out, non_C_dim, &alpha, x_perm, non_C_dim, dout_perm, non_C_dim, &beta, dweight, C_in));

    // 2.b sgemm call for dx_perm = dout_perm @ weight.T
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, non_C_dim, C_in, C_out, &alpha, dout_perm, non_C_dim, weight, C_in, &beta, dx_perm, non_C_dim));

    // 3. permute dx_perm from (C_out, B, H, W) to (B, C_out, H, W)
    int total_th2 = C_in * B * H * W;
    int n_blk2 = ceil_div(total_th2, block_size);
    permute_dx_kernel<<<n_blk2, block_size>>>(dx_perm, dx, B, C_in, H, W);

    // 4. reduce dout to get dbias
    dbias_kernel<<<C_out, block_size, block_size * sizeof(float)>>>(dout_perm, dbias, B, C_out, H, W);

    // free memory
    cudaCheck(cudaFree(x_perm));
    cudaCheck(cudaFree(dout_perm));
    cudaCheck(cudaFree(dx_perm));
}


#ifndef LINKING
int main(int argc, char **argv) {
    srand(0);
    int B = 32;
    int C_in = 64;
    int C_out = 128;
    int H = 64;
    int W = 64;

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    // create host memory to load data
    float* x = (float*)malloc(B * C_in * H * W * sizeof(float));
    float* weight = (float*)malloc(C_out * C_in * sizeof(float));
    float* bias = (float*)malloc(C_out * sizeof(float));
    float* out = (float*)malloc(B * C_out * H * W * sizeof(float));
    float* dout = (float*)malloc(B * C_out * H * W * sizeof(float));
    float* dx = (float*)malloc(B * C_in * H * W * sizeof(float));
    float* dweight = (float*)malloc(C_out * C_in * sizeof(float));
    float* dbias = (float*)malloc(C_out * sizeof(float));
    
    // read saved output
    FILE *file = fopen("conv2d_k1.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), B * C_in * H * W, file);
    freadCheck(weight, sizeof(float), C_out * C_in, file);
    freadCheck(bias, sizeof(float), C_out, file);
    freadCheck(out, sizeof(float), B * C_out * H * W, file);
    freadCheck(dout, sizeof(float), B * C_out * H * W, file);
    freadCheck(dx, sizeof(float), B * C_in * H * W, file);
    freadCheck(dweight, sizeof(float), C_out * C_in, file);
    freadCheck(dbias, sizeof(float), C_out, file);
    fclose(file);

    // allocate device memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_dout, *d_dx, *d_dweight, *d_dbias;
    cudaCheck(cudaMalloc(&d_x, B * C_in * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C_out * C_in * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C_out * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * C_out * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * C_out * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, B * C_in * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, C_out * C_in * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, C_out * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C_out * C_in * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C_out * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * C_out * H * W * sizeof(float), cudaMemcpyHostToDevice));

    //int block_sizes[] = {128, 256, 512, 1024};
    int block_sizes[] = {512};
    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        printf("Checking block size %d\n", block_sizes[j]);
        int block_size = block_sizes[j];
        conv2d_k1_forward1(cublas_handle, d_out, d_x, d_weight, d_bias, B, C_in, H, W, C_out, block_size);
        validate_result(d_out, out, "out", B * C_out * H * W);
    }
    printf("\nChecking new kernel\n");
    conv2d_k1_forward2(d_x, d_weight, d_bias, d_out, B, C_in, H, W, C_out);
    validate_result(d_out, out, "new_out", B * C_out * H * W);
    printf("Running our own forward benchmark for new kernel\n");
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));
    int repeat_times = 1000;
    float new_time = 0.0f;
    for (int i = 0; i < repeat_times; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        cudaEvent_t start, end;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
        cudaCheck(cudaEventRecord(start, nullptr));
        conv2d_k1_forward2(d_x, d_weight, d_bias, d_out, B, C_in, H, W, C_out);
        cudaCheck(cudaEventRecord(end, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        new_time += elapsed_time;
    }
    new_time /= repeat_times;
    printf("new kernel time: %.4f ms\n", new_time);
    
    printf("Forward pass successful\n\n");

    printf("Checking backward pass\n");
    float accuracy = 1e-1;
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("\nChecking block size %d\n", block_size);
        conv2d_k1_backward1(cublas_handle, d_dout, d_x, d_weight, d_dx, d_dweight, d_dbias, B, C_in, C_out, H, W, block_size);
        printf("\nChecking dweight\n");
        validate_result(d_dweight, dweight, "dweight", C_out * C_in, accuracy);
        printf("\nchecking dbias\n");
        validate_result(d_dbias, dbias, "dbias", C_out, accuracy);
        printf("\nchecking dx\n");
        validate_result(d_dx, dx, "dx", B * C_in * H * W);
    }
    printf("Backward pass successful\n\n");

    printf("All results match. Starting benchmarks.\n\n");
    //printf("Forward pass benchmarks:\n");
    //for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    //    int block_size = block_sizes[j];

    //    int repeat_times = 100;
    //    float elapsed_time = benchmark_kernel(repeat_times, conv2d_k1_forward1,
    //                                          cublas_handle, d_out, d_x, d_weight, d_bias,
    //                                          B, C_in, H, W, C_out, block_size);

    //    // napkin math: estimate the flops achieved
    //    // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
    //    float tflops = (float)B * H * W * C_in * C_out * 2 / elapsed_time * 1e3f / 1e12f;
    //    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    //}
    printf("Running our own forward benchmarks\n");
    float t1 = 0.0f;
    float t2 = 0.0f;
    float t3 = 0.0f;
    float t_total = 0.0f;
    for (int i = 0; i < repeat_times; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        conv2d_k1_forward1(cublas_handle, d_out, d_x, d_weight, d_bias, B, C_in, H, W, C_out, 512, &t1, &t2, &t3);
    }
    cudaCheck(cudaFree(flush_buffer));
    t1 /= repeat_times;
    t2 /= repeat_times;
    t3 /= repeat_times;
    t_total = t1 + t2 + t3;

    printf("t1: %.4f ms\n", t1);
    printf("t2: %.4f ms\n", t2);
    printf("t3: %.4f ms\n", t3);
    printf("t_total: %.4f ms\n", t_total);

    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, conv2d_k1_backward1,
                                              cublas_handle, d_dout, d_x, d_weight, d_dx, d_dweight, d_dbias,
                                              B, C_in, C_out, H, W, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }


    // free memory
    free(x);
    free(weight);
    free(bias);
    free(out);
    free(dout);
    free(dx);
    free(dweight);
    free(dbias);
    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dx));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
#endif