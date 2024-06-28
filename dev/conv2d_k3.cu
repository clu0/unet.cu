#include <cstdio>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "common.h"
#include "conv2d_k3.cuh"


__global__ void permute_w_tile_x_kernel1(
    const float* x, const float* weight,
    float* x_tiled, float* weight_perm,
    int B, int C_in, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < C_out * C_in * 9) {
        // permute weight
        int c_out = idx / (C_in * 9);
        int c_in = (idx / 9) % C_in;
        int k = idx % 9;
        int out_ind = k * C_out * C_in + c_out * C_in + c_in;
        weight_perm[out_ind] = weight[idx];
    }

    if (idx < B * C_in * H * W) {
        // tile x to shape (9, C_in, B, H, W)
        // for a given first dimension k, the entry at an index should be
        // what the k-th kernel is multiplied with in the convolution for that pixel
        // (not a good explanation... but obvious with a picture)
        // for example, a value at (H, W) = (0, 0) will never be convolved with kernels 5, 7 and 8
        // because it is on a corner
        int x_offset = B * C_in * H * W;
        int img_size = H * W;
        int b = idx / (C_in * img_size);
        int c = (idx / img_size) % C_in;
        int h_start = (idx / W) % H;
        int w_start = idx % W;
        float x_val = x[idx];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int h_end = h_start + 1 - i;
                int w_end = w_start + 1 - j;
                if (h_end >= 0 && h_end < H && w_end >= 0 && w_end < W) {
                    x_tiled[c * B * img_size + b * img_size + h_end * W + w_end] = x_val;
                }
                // shift pointer to next kernel
                x_tiled += x_offset;
            }
        }
    }
}


__global__ void reduce_transpose_add_bias_kernel1(
    const float* out_tmp, const float* bias,
    float* out,
    int B, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // make sure we're in bound
    assert(idx < B * C_out * H * W);
    // make sure every block lies in the same channel
    assert((B * H * W) % blockDim.x == 0);

    int img_size = H * W;
    int c = idx / (B * img_size);
    int b = (idx / img_size) % B;
    int h = (idx / W) % H;
    int w = idx % W;

    __shared__ float bias_sh[1];
    if (threadIdx.x == 0) {
        bias_sh[0] = bias[c];
    }
    __syncthreads();

    float out_val = bias_sh[0];

    for (int k = 0; k < 9; k++) {
        float out_tmp_val = out_tmp[k * C_out * B * img_size + c * B * img_size + b * img_size + h * W + w];
        out_val += out_tmp_val;
    }

    out[b * C_out * img_size + c * img_size + h * W + w] = out_val;
}

void conv2d_k3_forward1(
    cublasHandle_t cublas_handle,
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int C_out, int H, int W,
    int block_size
) {
    // implements Conv2D with kernel size 1 and padding 1
    // shapes:
    // x (B, C_in, H, W)
    // weight (C_out, C_in, 3, 3)
    // bias (C_out,)

    // plan:
    // 1. transpose and tile x into x_tiled of shape (9, C_in, B, H, W)
    // 2. transpose weight into shape (9, C_out, C_in) (we will do this in the same kernel as one above)
    // 3. call cublasSgemmStridedBatched with 9 batches
    // 4. reduce over the 9 kernel dimensions, add bias, and transpose output to shape (B, C_out, H, W)

    // 1. and 2.
    // transpose weight and tile x
    float *x_tiled, *weight_perm;
    cudaCheck(cudaMalloc(&x_tiled, 9 * C_in * B * H * W * sizeof(float)));
    cudaCheck(cudaMemset(x_tiled, 0, 9 * C_in * B * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&weight_perm, 9 * C_out * C_in * sizeof(float)));
    int total_th_1 = max_int(B * C_in * H * W, C_in * C_out * 9);
    int n_blk_1 = ceil_div(total_th_1, block_size);
    permute_w_tile_x_kernel1<<<n_blk_1, block_size>>>(x, weight, x_tiled, weight_perm, B, C_in, C_out, H, W);

    // 3. matmul via cublasSgemmStridedBatched
    float* out_tmp;
    cudaCheck(cudaMalloc(&out_tmp, 9 * C_out * B * H * W * sizeof(float)));
    float alpha = 1.0f;
    float beta = 0.0f;
    int bhw = B * H * W;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, bhw, C_out, C_in, &alpha, x_tiled, bhw, C_in * bhw, weight_perm, C_in, C_out * C_in, &beta, out_tmp, bhw, C_out * bhw, 9));

    // reduce, add bias, and transpose
    int total_th_2 = B * C_out * H * W;
    int n_blk_2 = ceil_div(total_th_2, block_size);
    reduce_transpose_add_bias_kernel1<<<n_blk_2, block_size>>>(out_tmp, bias, out, B, C_out, H, W);

    // free memory
    cudaCheck(cudaFree(x_tiled));
    cudaCheck(cudaFree(weight_perm));
    cudaCheck(cudaFree(out_tmp));
}


__global__ void transpose_tile_x_kernel2(
    const float* x,
    float* x_tiled,
    int B, int C_in, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * C_in * H * W) {
        // similar to permute_w_tile_x_kernel1, except now the target shape is (C_in, 9, B, H, W)
        int img_size = H * W;
        int b = idx / (C_in * img_size);
        int c = (idx / img_size) % C_in;
        int h_start = (idx / W) % H;
        int w_start = idx % W;
        float x_val = x[idx];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int h_end = h_start + 1 - i;
                int w_end = w_start + 1 - j;
                if (h_end >= 0 && h_end < H && w_end >= 0 && w_end < W) {
                    x_tiled[c * 9 * B * img_size + (i * 3 + j) * B * img_size + b * img_size + h_end * W + w_end] = x_val;
                }
            }
        }
    }
}

__global__ void transpose_out_add_bias_kernel2(
    const float* out_tmp, const float* bias,
    float* out,
    int B, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // make sure we're in bound
    assert(idx < B * C_out * H * W);
    // make sure every block lies in the same channel
    assert((B * H * W) % blockDim.x == 0);

    int img_size = H * W;
    int c = idx / (B * img_size);
    int b = (idx / img_size) % B;
    int h = (idx / W) % H;
    int w = idx % W;

    __shared__ float bias_sh[1];
    if (threadIdx.x == 0) {
        bias_sh[0] = bias[c];
    }
    __syncthreads();
    
    out[b * C_out * img_size + c * img_size + h * W + w] = bias_sh[0] + out_tmp[idx];
}

void conv2d_k3_forward2(
    cublasHandle_t cublas_handle,
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int C_out, int H, int W,
    int block_size,
    float* t1, float* t2, float* t3
) {
    // similar to kernel 1, but transpose into a different shape so that we can just make one big matmul call
    // and no need to reduce afterwards
    // plan:
    // 1. transpose and tile x into x_tiled of shape (C_in, 9, B, H, W)
    // 2. call cublasSgemm on x_tiled and weight, no need for transposes!
    // 3. add bias

    // 1. transpose x to x_tiled
    cudaEvent_t start, end1, end2, end3;
    if (t1 != nullptr) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end1));
        cudaCheck(cudaEventCreate(&end2));
        cudaCheck(cudaEventCreate(&end3));
        cudaCheck(cudaEventRecord(start, nullptr));
    }
    float* x_tiled;
    cudaCheck(cudaMalloc(&x_tiled, C_in * 9 * B * H * W * sizeof(float)));
    cudaCheck(cudaMemset(x_tiled, 0, C_in * 9 * B * H * W * sizeof(float)));
    if (t1 != nullptr) {
        cudaCheck(cudaEventRecord(end1, nullptr));
    }
    int total_th_1 = B * C_in * H * W;
    int n_blk_1 = ceil_div(total_th_1, block_size);
    transpose_tile_x_kernel2<<<n_blk_1, block_size>>>(x, x_tiled, B, C_in, H, W);

    // 2. matmul with cublasSgemm
    float* out_tmp;
    cudaCheck(cudaMalloc(&out_tmp, C_out * B * H * W * sizeof(float)));
    float alpha = 1.0f;
    float beta = 0.0f;
    int bhw = B * H * W;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, bhw, C_out, C_in * 9, &alpha, x_tiled, bhw, weight, C_in * 9, &beta, out_tmp, bhw));


    // 3. transpose and add bias
    int total_th_2 = B * C_out * H * W;
    int n_blk_2 = ceil_div(total_th_2, block_size);
    transpose_out_add_bias_kernel2<<<n_blk_2, block_size>>>(out_tmp, bias, out, B, C_out, H, W);
    if (t2 != nullptr) {
        cudaCheck(cudaEventRecord(end2, nullptr));
    }


    // free memory
    cudaCheck(cudaFree(x_tiled));
    cudaCheck(cudaFree(out_tmp));
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

// this kernel will roughly follow kernel 5 here: https://github.com/siboehm/SGEMM_CUDA, with a few differences
// the main operation is convolving the input X with the weights
// weights are shaped (O, Cx9), and X is shaped (B, C, HxW)
// We are morally doing a matmul with dimensions O, C, HxW
template <const int BO, const int BH, const int BC, const int TO, const int TH>
__global__ void conv2d_k3_forward_kernel1(
    const float* x, const float* weight, const float* bias,
    float* out,
    const int B, const int C, const int O, const int H, const int W
) {
    // constants
    const int C9 = C * 9;
    const int weight_size = O * C * 9;
    const int x_size = B * C * H * W;
    // n_threads / block = (BO * BH) / (TO * TH) = 256
    // n_blocks = B * (O / BO) * (H * W / BH)
    // we will set gridDim to (BxHxW / BH, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = H * W / BH;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // for compute step, to determine which of the size TOxTH block the thread is responsible for
    const int threads_per_col = ceil_div(BH, TH);
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // shared mem for loading weights and X
    const int w_s_sz = BO * BC * 9;
    const int w_s_width = BC * 9;
    const int x_s_sz = BC * BH;
    __shared__ float w_s[w_s_sz];
    __shared__ float x_s[x_s_sz];

    // registers to store local results
    float thread_results[TO * TH * 9] = {0.0};
    // registers for weights and x
    float reg_w[TO] = {0.0};
    float reg_x[TH] = {0.0};

    // calculate offset in x for bounds checking later
    const int x_block_offset = batch * C * H * W + block_col * BH;
    // calculate weight offset for bounds checking
    const int w_block_offset = block_row * BO * C9;

    // outer loop over BC size blocks of channel C
    for (int c_idx = 0; c_idx < C; c_idx += BC) {
        // x and weight offsets for this block
        int x_inner_offset = c_idx * H * W;
        int w_inner_offset = c_idx * 9;
        // load weight into smem
        int w_abs_idx;
        for (int i = threadIdx.x; i < w_s_sz; i += blockDim.x) {
            int w_i = i / w_s_width;
            int w_j = i % w_s_width;
            w_abs_idx = w_block_offset + w_inner_offset + w_i * C9 + w_j;
            if (w_abs_idx < weight_size) {
                w_s[w_i * w_s_width + w_j] = weight[w_abs_idx];
            }
        }
        // load x into smem
        int x_abs_idx;
        for (int i = threadIdx.x; i < x_s_sz; i += blockDim.x) {
            int x_i = i / BH;
            int x_j = i % BH;
            x_abs_idx = x_block_offset + x_inner_offset + x_i * H * W + x_j;
            if (x_abs_idx < x_size) {
                x_s[x_i * BH + x_j] = x[x_abs_idx];
            }
        }
        __syncthreads();

        // calculate per thread results
        for (int dot_idx = 0; dot_idx < BC; dot_idx++) {
            // load x_s into registers
            for (int j = 0; j < TH; j++) {
                reg_x[j] = x_s[dot_idx * BH + thread_col * TH + j];
            }
            for (int conv_idx = 0; conv_idx < 9; conv_idx++) {
                for (int i = 0; i < TO; i++) {
                    reg_w[i] = w_s[(thread_row * TO + i) * w_s_width + dot_idx * 9 + conv_idx];
                }

                for (int i = 0; i < TO; i++) {
                    for (int j = 0; j < TH; j++) {
                        float val = reg_w[i] * reg_x[j];
                        thread_results[conv_idx * TO * TH + i * TH + j] += val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // write out results
    // need to be careful here: for each pixel at (h_abs, w_abs) and its product with kernel k (k < 9)
    // There are 2 options
    // 1. the product should be added to some pixel, then do an atomiAdd
    // 2. the product should not contribute to any pixel because it would be out of bound, in this case just skip
    int out_batch_offset = batch * O * H * W;
    for (int j = 0; j < TH; j++) {
        int hw_abs = block_col * BH + thread_col * TH + j;
        int h_abs = hw_abs / W;
        int w_abs = hw_abs % W;
        for (int i = 0; i < TO; i++) {
            int o_abs = block_row * BO + thread_row * TO + i;
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                int h_target_abs = h_abs + 1 - k_i;
                int w_target_abs = w_abs + 1 - k_j;
                if (h_target_abs >= 0 && h_target_abs < H && w_target_abs >= 0 && w_target_abs < W) {
                    int out_target_abs = out_batch_offset + o_abs * H * W + h_target_abs * W + w_target_abs;
                    int th_res_idx = k * TO * TH + i * TH + j;
                    atomicAdd(out + out_target_abs, thread_results[th_res_idx]);
                }
            }
            int out_abs = out_batch_offset + o_abs * H * W + h_abs * W + w_abs;
            atomicAdd(out + out_abs, bias[o_abs]);
        }
    }
}


// almost same as 1, except we load A as transposed
template <const int BO, const int BH, const int BC, const int TO, const int TH>
__global__ void conv2d_k3_forward_kernel2(
    const float* x, const float* weight, const float* bias,
    float* out,
    const int B, const int C, const int O, const int H, const int W
) {
    // constants
    const int C9 = C * 9;
    const int weight_size = O * C * 9;
    const int x_size = B * C * H * W;
    // n_threads / block = (BO * BH) / (TO * TH) = 256
    // n_blocks = B * (O / BO) * (H * W / BH)
    // we will set gridDim to (BxHxW / BH, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = H * W / BH;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // for compute step, to determine which of the size TOxTH block the thread is responsible for
    const int threads_per_col = ceil_div(BH, TH);
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // shared mem for loading weights and X
    const int w_s_sz = BO * BC * 9;
    const int w_s_width = BC * 9;
    const int x_s_sz = BC * BH;
    __shared__ float w_s[w_s_sz];
    __shared__ float x_s[x_s_sz];

    // registers to store local results
    float thread_results[TO * TH * 9] = {0.0};
    // registers for weights and x
    float reg_w[TO] = {0.0};
    float reg_x[TH] = {0.0};

    // calculate offset in x for bounds checking later
    const int x_block_offset = batch * C * H * W + block_col * BH;
    // calculate weight offset for bounds checking
    const int w_block_offset = block_row * BO * C9;

    // outer loop over BC size blocks of channel C
    for (int c_idx = 0; c_idx < C; c_idx += BC) {
        // x and weight offsets for this block
        int x_inner_offset = c_idx * H * W;
        int w_inner_offset = c_idx * 9;
        // load weight into smem
        int w_abs_idx;
        for (int i = threadIdx.x; i < w_s_sz; i += blockDim.x) {
            int w_i = i / w_s_width;
            int w_j = i % w_s_width;
            w_abs_idx = w_block_offset + w_inner_offset + w_i * C9 + w_j;
            if (w_abs_idx < weight_size) {
                w_s[w_j * BO + w_i] = weight[w_abs_idx];
            }
        }
        // load x into smem
        int x_abs_idx;
        for (int i = threadIdx.x; i < x_s_sz; i += blockDim.x) {
            int x_i = i / BH;
            int x_j = i % BH;
            x_abs_idx = x_block_offset + x_inner_offset + x_i * H * W + x_j;
            if (x_abs_idx < x_size) {
                x_s[x_i * BH + x_j] = x[x_abs_idx];
            }
        }
        __syncthreads();

        // calculate per thread results
        for (int dot_idx = 0; dot_idx < BC; dot_idx++) {
            // load x_s into registers
            for (int j = 0; j < TH; j++) {
                reg_x[j] = x_s[dot_idx * BH + thread_col * TH + j];
            }
            for (int conv_idx = 0; conv_idx < 9; conv_idx++) {
                for (int i = 0; i < TO; i++) {
                    reg_w[i] = w_s[(dot_idx * 9 + conv_idx) * BO + thread_row * TO + i];
                }

                for (int i = 0; i < TO; i++) {
                    for (int j = 0; j < TH; j++) {
                        float val = reg_w[i] * reg_x[j];
                        thread_results[conv_idx * TO * TH + i * TH + j] += val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // write out results
    // need to be careful here: for each pixel at (h_abs, w_abs) and its product with kernel k (k < 9)
    // There are 2 options
    // 1. the product should be added to some pixel, then do an atomiAdd
    // 2. the product should not contribute to any pixel because it would be out of bound, in this case just skip
    int out_batch_offset = batch * O * H * W;
    for (int j = 0; j < TH; j++) {
        int hw_abs = block_col * BH + thread_col * TH + j;
        int h_abs = hw_abs / W;
        int w_abs = hw_abs % W;
        for (int i = 0; i < TO; i++) {
            int o_abs = block_row * BO + thread_row * TO + i;
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                int h_target_abs = h_abs + 1 - k_i;
                int w_target_abs = w_abs + 1 - k_j;
                if (h_target_abs >= 0 && h_target_abs < H && w_target_abs >= 0 && w_target_abs < W) {
                    int out_target_abs = out_batch_offset + o_abs * H * W + h_target_abs * W + w_target_abs;
                    int th_res_idx = k * TO * TH + i * TH + j;
                    atomicAdd(out + out_target_abs, thread_results[th_res_idx]);
                }
            }
            int out_abs = out_batch_offset + o_abs * H * W + h_abs * W + w_abs;
            atomicAdd(out + out_abs, bias[o_abs]);
        }
    }
}


// almost same as 2, but we vectorize the loads and use float4
template <const int BO, const int BH, const int BC, const int TO, const int TH>
__global__ void conv2d_k3_forward_kernel3(
    float* x, float* weight, float* bias,
    float* out,
    const int B, const int C, const int O, const int H, const int W
) {
    // constants
    assert(TH == 4); // needed for reg_x loads later, so we can do one vectorized load for each thread
    const int C9 = C * 9;
    // n_threads / block = (BO * BH) / (TO * TH) = 256
    // n_blocks = B * (O / BO) * (H * W / BH)
    // we will set gridDim to (BxHxW / BH, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = H * W / BH;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // for compute step, to determine which of the size TOxTH block the thread is responsible for
    const int threads_per_col = ceil_div(BH, TH);
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // shared mem for loading weights and X
    const int w_s_sz = BO * BC * 9;
    const int w_s_width = BC * 9;
    const int x_s_sz = BC * BH;
    __shared__ float w_s[w_s_sz];
    __shared__ float x_s[x_s_sz];

    // registers to store local results
    float thread_results[TO * TH * 9] = {0.0};
    // registers for weights and x
    float reg_w[TO] = {0.0};
    float reg_x[TH] = {0.0};

    // calculate offset in x for bounds checking later
    const int x_block_offset = batch * C * H * W + block_col * BH;
    // calculate weight offset for bounds checking
    const int w_block_offset = block_row * BO * C9;

    // outer loop over BC size blocks of channel C
    for (int c_idx = 0; c_idx < C; c_idx += BC) {
        // x and weight offsets for this block
        int x_inner_offset = c_idx * H * W;
        int w_inner_offset = c_idx * 9;
        // load weight into smem
        for (int i = threadIdx.x; i < w_s_sz / 4; i += blockDim.x) {
            int inner_col = i % (w_s_width / 4);
            int inner_row = i / (w_s_width / 4);
            float4 tmp = 
                reinterpret_cast<float4 *>(&weight[w_block_offset + w_inner_offset + inner_row * C9 + inner_col * 4])[0];
            w_s[(inner_col * 4 + 0) * BO + inner_row] = tmp.x;
            w_s[(inner_col * 4 + 1) * BO + inner_row] = tmp.y;
            w_s[(inner_col * 4 + 2) * BO + inner_row] = tmp.z;
            w_s[(inner_col * 4 + 3) * BO + inner_row] = tmp.w;
        }
        // load x into smem
        for (int i = threadIdx.x; i < x_s_sz / 4; i += blockDim.x) {
            int inner_col = i % (BH / 4);
            int inner_row = i / (BH / 4);
            reinterpret_cast<float4 *>(&x_s[inner_row * BH + inner_col * 4])[0] = 
                reinterpret_cast<float4 *>(&x[x_block_offset + x_inner_offset + inner_row * H * W + inner_col * 4])[0];
        }
        __syncthreads();

        // calculate per thread results
        for (int dot_idx = 0; dot_idx < BC; dot_idx++) {
            // load x_s into registers
            // NOTE: relying on the fact that TH = 4
            reinterpret_cast<float4 *>(reg_x)[0] = reinterpret_cast<float4 *>(&x_s[dot_idx * BH + thread_col * TH])[0];
            for (int conv_idx = 0; conv_idx < 9; conv_idx++) {
                for (int i = 0; i < TO; i++) {
                    reg_w[i] = w_s[(dot_idx * 9 + conv_idx) * BO + thread_row * TO + i];
                }

                for (int i = 0; i < TO; i++) {
                    for (int j = 0; j < TH; j++) {
                        float val = reg_w[i] * reg_x[j];
                        thread_results[conv_idx * TO * TH + i * TH + j] += val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // write out results
    // need to be careful here: for each pixel at (h_abs, w_abs) and its product with kernel k (k < 9)
    // There are 2 options
    // 1. the product should be added to some pixel, then do an atomiAdd
    // 2. the product should not contribute to any pixel because it would be out of bound, in this case just skip
    int out_batch_offset = batch * O * H * W;
    for (int j = 0; j < TH; j++) {
        int hw_abs = block_col * BH + thread_col * TH + j;
        int h_abs = hw_abs / W;
        int w_abs = hw_abs % W;
        for (int i = 0; i < TO; i++) {
            int o_abs = block_row * BO + thread_row * TO + i;
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                int h_target_abs = h_abs + 1 - k_i;
                int w_target_abs = w_abs + 1 - k_j;
                if (h_target_abs >= 0 && h_target_abs < H && w_target_abs >= 0 && w_target_abs < W) {
                    int out_target_abs = out_batch_offset + o_abs * H * W + h_target_abs * W + w_target_abs;
                    int th_res_idx = k * TO * TH + i * TH + j;
                    atomicAdd(out + out_target_abs, thread_results[th_res_idx]);
                }
            }
            int out_abs = out_batch_offset + o_abs * H * W + h_abs * W + w_abs;
            atomicAdd(out + out_abs, bias[o_abs]);
        }
    }
}

// CPU version for debugging
void conv_k3_forward_cpu(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C, int OC, int H, int W
) {
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int k = 0; k < 9; k++) {
                        // check if the output will actually be used, if not skip
                        int k_i = k / 3;
                        int k_j = k % 3;
                        int h_target = h + 1 - k_i;
                        int w_target = w + 1 - k_j;
                        if (h_target < 0 || h_target >= H || w_target < 0 || w_target >= W) {
                            continue;
                        }
                        float out_bohw = 0.0f;
                        for (int c = 0; c < C; c++) {
                            float new_val = weight[oc * C * 9 + c * 9 + k] * x[b * C * H * W + c * H * W + h * W + w];
                            out_bohw += new_val;
                            //if (b == 0 && h == 0 && w == 1 && k == 5 && oc == 0) {
                            //    printf("(0, 1) OC 0 kern 5: contribution from %d: %f, out val %f\n", c, new_val, out_bohw);
                            //}
                        }
                        int out_idx = b * OC * H * W + oc * H * W + h_target * W + w_target;
                        //if (h_target == 0 && w_target == 0 && oc == 0 && b == 0) {
                        //    printf("(0, 0) getting contribution from (%d, %d) at k %d: %f, prev_val %f\n", h, w, k, out_bohw, out[out_idx]);
                        //}
                        out[out_idx] += out_bohw;
                        //if (h_target == 0 && w_target == 0 && oc == 0 && b == 0) {
                        //    printf("(0, 0) getting contribution from (%d, %d) at k %d: %f, new val %f\n", h, w, k, out_bohw, out[out_idx]);
                        //}
                    }
                    
                    //if (h == 0 && w == 0 && oc == 0 && b == 0) {
                    //    printf("(0, 0) adding bias %f\n", bias[oc]);
                    //}
                    out[b * OC * H * W + oc * H * W + h * W + w] += bias[oc];
                }
            }
            printf("(0, 0) final value: %f\n", out[0]);
            return;
        }
    }
}

void conv2d_k3_forward3(
    float* x, float* weight, float* bias,
    float* out,
    const int B, const int C_in, const int C_out, const int H, const int W
) {
    // since the kernel will add to the output, we need to zero it first
    cudaCheck(cudaMemset(out, 0, B * C_out * H * W * sizeof(float)));

    if (H * W == 64) {
        const int BH = 64;
        const int TH = 4;
        assert(C_out % 64 == 0 && C_in % 8 == 0);
        const int BC = 16;
        const int BO = 64;
        const int TO = 4;
        dim3 gridDim(B * H * W / BH, C_out / BO);
        dim3 blockDim((BO * BH) / (TO * TH));
        conv2d_k3_forward_kernel3<BO, BH, BC, TO, TH><<<gridDim, blockDim>>>(
            x, weight, bias, out, B, C_in, C_out, H, W
        );
    } else if (C_in == 3) {
        const int BH = 64;
        assert(H * W % BH == 0);
        const int TH = 4;
        const int BC = 3;
        assert(C_out % 64 == 0);
        const int BO = 64;
        const int TO = 4;
        dim3 gridDim(B * H * W / BH, C_out / BO);
        dim3 blockDim((BO * BH) / (TO * TH));
        conv2d_k3_forward_kernel2<BO, BH, BC, TO, TH><<<gridDim, blockDim>>>(
            x, weight, bias, out, B, C_in, C_out, H, W
        );
    } else if (C_out == 3) {
        const int BH = 64 * 2;
        assert(H * W % BH == 0);
        const int TH = 4;
        assert(C_in % 8 == 0);
        const int BC = 8;
        const int BO = 3;
        const int TO = 3;
        dim3 gridDim(B * H * W / BH, C_out / BO);
        dim3 blockDim((BO * BH) / (TO * TH));
        conv2d_k3_forward_kernel2<BO, BH, BC, TO, TH><<<gridDim, blockDim>>>(
            x, weight, bias, out, B, C_in, C_out, H, W
        );
    } else {
        const int BH = 64 * 4;
        assert(H * W % BH == 0);
        const int TH = 4;
        assert(C_out % 64 == 0 && C_in % 8 == 0);
        const int BC = 16;
        const int BO = 8;
        const int TO = 2;
        dim3 gridDim(B * H * W / BH, C_out / BO);
        dim3 blockDim((BO * BH) / (TO * TH));
        conv2d_k3_forward_kernel3<BO, BH, BC, TO, TH><<<gridDim, blockDim>>>(
            x, weight, bias, out, B, C_in, C_out, H, W
        );
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// backward pass

__global__ void permute_dout_tile_x_kernel(
    const float* dout, const float* x,
    float* dout_perm, float* x_tiled,
    int B, int C_in, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = H * W;
    if (idx < B * C_out * H * W) {
        // permute dout
        int b = idx / (C_out * img_size);
        int c = (idx / img_size) % C_out;
        int h = (idx / W) % H;
        int w = idx % W;
        dout_perm[c * B * img_size + b * img_size + h * W + w] = dout[idx];
    }

    if (idx < B * C_in * H * W) {
        // tile x_tiled, see permute_w_tile_x_kernel1 for explanation
        // except that we tile into a shape of (B, H, W, C_in, 9), the transpose of the tiling there
        int b = idx / (C_in * img_size);
        int c = (idx / img_size) % C_in;
        int h_start = (idx / W) % H;
        int w_start = idx % W;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int h_end = h_start + 1 - i;
                int w_end = w_start + 1 - j;
                if (h_end >= 0 && h_end < H && w_end >= 0 && w_end < W) {
                    x_tiled[b * img_size * C_in * 9 + h_end * W * C_in * 9 + w_end * C_in * 9 + c * 9 + i * 3 + j]
                        = x[idx];
                }
            }
        }
    }
}

__global__ void reduce_dx_kernel(
    const float* dx_tiled,
    float* dx,
    int B, int C_in, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * C_in * H * W) {
        int img_size = H * W;
        int b = idx / (C_in * img_size);
        int c = (idx / img_size) % C_in;
        int h_end = (idx / W) % H;
        int w_end = idx % W;
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int h_start = h_end + 1 - i;
                int w_start = w_end + 1 - j;
                if (h_start >= 0 && h_start < H && w_start >= 0 && w_start < W) {
                    sum += dx_tiled[c * 9 * B * img_size + (i * 3 + j) * B * img_size + b * img_size + h_start * W + w_start];
                }
            }
        }
        dx[idx] = sum;
    }
}


__global__ void reduce_dbias_kernel1(
    const float* dout_perm,
    float* dbias,
    int B, int C_out, int H, int W
) {
    // dout_perm has shape (C_out, B, H, W)
    // each block will reduce over one channel
    dout_perm += blockIdx.x * B * H * W;
    int bhw = B * H * W;
    assert(blockDim.x <= bhw);
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < bhw; i += blockDim.x) {
        sum += dout_perm[i];
    }
    sdata[threadIdx.x] = sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dbias[blockIdx.x] = sdata[0];
    }
}

// use cg to do warp reduce
// doesn't really seem faster
__global__ void reduce_dbias_kernel2(
    const float* dout_perm,
    float* dbias,
    int B, int C_out, int H, int W
) {
    int bhw = B * H * W;
    assert(blockDim.x <= bhw);
    // move dout_perm to the right block
    dout_perm += blockIdx.x * bhw;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32];

    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < bhw; i += blockDim.x) {
        thread_sum += dout_perm[i];
    }
    
    // warp reduce
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    shared_sum[warp_id] = warp_sum;
    __syncthreads();

    // warp reduce
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});

    if (threadIdx.x == 0) {
        dbias[blockIdx.x] = block_sum;
    }
}

void conv2d_k3_backward1(
    cublasHandle_t cublas_handle,
    const float* dout, const float* x, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C_in, int C_out, int H, int W,
    int block_size,
    float* t1, float* t2, float* t3, float* t4, float* t5, float* t6
) {
    // shapes:
    // dout (B, C_out, H, W)
    // x (B, C_in, H, W)
    // weight (C_out, C_in, 3, 3)
    // bias (C_out,)

    // we need to calculate dx, dweight, dbias separately
    // dbias is a single reduce
    // for dweight, we need to tile x into x_tile of shape (B * H * W, C_in * 9)
    // then do a matmul with dout_perm (shape (C_out, B * H * W))
    // for dx, we do dout_perm @ weight.T (store it in x_tile), and then reduce to get dx
    // If we don't want to do a reduce when computing dx, we need to make 9 copies of dout
    // which might be ok but increases memory usage. We can try it later.
    cudaEvent_t start, end1, end2, end3, end4, end5, end6;
    if (t1 != nullptr) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end1));
        cudaCheck(cudaEventCreate(&end2));
        cudaCheck(cudaEventCreate(&end3));
        cudaCheck(cudaEventCreate(&end4));
        cudaCheck(cudaEventCreate(&end5));
        cudaCheck(cudaEventCreate(&end6));
        cudaCheck(cudaEventRecord(start, nullptr));
    }

    // 1. transpose to dout_perm and tile x_tile
    float *dout_perm, *x_tiled;
    cudaCheck(cudaMalloc(&x_tiled, B * H * W * C_in * 9 * sizeof(float)));
    cudaCheck(cudaMemset(x_tiled, 0, B * H * W * C_in * 9 * sizeof(float)));
    cudaCheck(cudaMalloc(&dout_perm, B * C_out * H * W * sizeof(float)));
    if (t1 != nullptr) {
        cudaCheck(cudaEventRecord(end1, nullptr));
    }
    int total_th_1 = max_int(C_in, C_out) * B * H * W;
    int n_blk_1 = ceil_div(total_th_1, block_size);
    permute_dout_tile_x_kernel<<<n_blk_1, block_size>>>(dout, x, dout_perm, x_tiled, B, C_in, C_out, H, W);
    cudaCheck(cudaGetLastError());
    if (t2 != nullptr) {
        cudaCheck(cudaEventRecord(end2, nullptr));
    }

    // 2. get dweight by single matmul between x_tiled and dout_perm
    float alpha = 1.0f;
    float beta = 0.0f;
    int bhw = B * H * W;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C_in * 9, C_out, bhw, &alpha, x_tiled, C_in * 9, dout_perm, bhw, &beta, dweight, C_in * 9));

    if (t2 != nullptr) {
        cudaCheck(cudaEventRecord(end3, nullptr));
    }

    // 3. get dx_tile by matmul between dout_perm and weight.T, and save output in x_tiled
    // note: beta = 0 overwrites existing values in x_tiled
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, bhw, C_in * 9, C_out, &alpha, dout_perm, bhw, weight, C_in * 9, &beta, x_tiled, bhw));


    // 4. reduce x_tiled to get dx
    int total_th_2 = B * C_in * H * W;
    int n_blk_2 = ceil_div(total_th_2, block_size);
    reduce_dx_kernel<<<n_blk_2, block_size>>>(x_tiled, dx, B, C_in, H, W);

    if (t2 != nullptr) {
        cudaCheck(cudaEventRecord(end4, nullptr));
    }

    // 5. reduce dbias
    //reduce_dbias_kernel1<<<C_out, block_size, block_size * sizeof(float)>>>(dout_perm, dbias, B, C_out, H, W);
    reduce_dbias_kernel2<<<C_out, block_size>>>(dout_perm, dbias, B, C_out, H, W);

    if (t5 != nullptr) {
        cudaCheck(cudaEventRecord(end5, nullptr));
    }

    // free memory
    cudaCheck(cudaFree(dout_perm));
    cudaCheck(cudaFree(x_tiled));
    if (t5 != nullptr) {
        cudaCheck(cudaEventRecord(end6, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end1));
        cudaCheck(cudaEventSynchronize(end2));
        cudaCheck(cudaEventSynchronize(end3));
        cudaCheck(cudaEventSynchronize(end4));
        cudaCheck(cudaEventSynchronize(end5));
        cudaCheck(cudaEventSynchronize(end6));
        float elapsed1, elapsed2, elapsed3, elapsed4, elapsed5, elapsed6;
        cudaCheck(cudaEventElapsedTime(&elapsed1, start, end1));
        cudaCheck(cudaEventElapsedTime(&elapsed2, end1, end2));
        cudaCheck(cudaEventElapsedTime(&elapsed3, end2, end3));
        cudaCheck(cudaEventElapsedTime(&elapsed4, end3, end4));
        cudaCheck(cudaEventElapsedTime(&elapsed5, end4, end5));
        cudaCheck(cudaEventElapsedTime(&elapsed6, end5, end6));
        *t1 += elapsed1;
        *t2 += elapsed2;
        *t3 += elapsed3;
        *t4 += elapsed4;
        *t5 += elapsed5;
        *t6 += elapsed6;
    }
}

// almost the same as conv2d_k3_forward_kernel, except that the compute and the writes
// are different, because now we want to treat the C dimension as the inner product dimension
template <const int BO, const int BH, const int BC, const int TC, const int TH>
__global__ void dx_backward_kernel1(
    float* dout, float* weight,
    float* dx,
    const int B, const int C, const int O, const int H, const int W
) {
    const int C9 = C * 9;
    // n_threads per block = (BC * BH) / (TC * TH)
    // set gridDim to (BxHxW / BH, C / TC)
    // find the output block that this thread block is responsible for
    const int block_row = blockIdx.y;
    const int blocks_per_col = H * W / BH;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // index for the TCxTH block that the thread is responsible
    const int threads_per_col = BH / TH;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for weights and dout
    const int w_s_width = BC * 9;
    const int w_s_sz = BO * w_s_width;
    const int dout_s_sz = BO * BH;
    __shared__ float w_s[w_s_sz];
    __shared__ float dout_s[dout_s_sz];

    // registers for local results
    float thread_results[TC * TH * 9] = {0.0};
    // regs for weights and dout
    float reg_w[TC * 9] = {0.0};
    float reg_dout[TH] = {0.0};

    // calculate the block offsets
    const int dout_block_offset = batch * O * H * W + block_col * BH;
    const int w_block_offset = block_row * w_s_width;

    // outer loop over BO sizes of channel O
    for (int o_idx = 0; o_idx < O; o_idx += BO) {
        // offsets for this inner block
        int w_inner_offset = o_idx * C9;
        int dout_inner_offset = o_idx * H * W;

        // load weights to smem
        if (BC % 4 == 0) {
            // vectorize dram load
            for (int i = threadIdx.x; i < w_s_sz / 4; i += blockDim.x) {
                int inner_col = i % (w_s_width / 4);
                int inner_row = i / (w_s_width / 4);
                reinterpret_cast<float4 *>(&w_s[inner_row * w_s_width + inner_col * 4])[0] = 
                    reinterpret_cast<float4 *>(&weight[w_block_offset + w_inner_offset + inner_row * C9 + inner_col * 4])[0];
            }
        } else {
            for (int i = threadIdx.x; i < w_s_sz; i += blockDim.x) {
                int inner_col = i % w_s_width;
                int inner_row = i / w_s_width;
                w_s[inner_row * w_s_width + inner_col] = weight[w_block_offset + w_inner_offset + inner_row * C9 + inner_col];
            }
        }
        // load dout
        for (int i = threadIdx.x; i < dout_s_sz / 4; i += blockDim.x) {
            int inner_col = i % (BH / 4);
            int inner_row = i / (BH / 4);
            reinterpret_cast<float4 *>(&dout_s[inner_row * BH + inner_col * 4])[0] = 
                reinterpret_cast<float4 *>(&dout[dout_block_offset + dout_inner_offset + inner_row * H * W + inner_col * 4])[0];
        }
        __syncthreads();
        

        // calculate per thread results
        for (int dot_idx = 0; dot_idx < BO; dot_idx++) {
            // load dout and w into registers
            for (int j = 0; j < (TH / 4); j++) {
                reinterpret_cast<float4 *>(&reg_dout[j * 4])[0] = 
                    reinterpret_cast<float4 *>(&dout_s[dot_idx * BH + thread_col * TH + j * 4])[0];
            }
            // non vectorized load
            //for (int j = 0; j < TH; j++) {
            //    reg_dout[j] = dout_s[dot_idx * BH + thread_col * TH + j];
            //}
            if (TC % 4 == 0) {
                // vectorize smem load
                for (int i = 0; i < (TC * 9 / 4); i++) {
                    reinterpret_cast<float4 *>(&reg_w[i * 4])[0] = 
                        reinterpret_cast<float4 *>(&w_s[dot_idx * w_s_width + thread_row * TC * 9 + i * 4])[0];
                }
            } else {
                for (int i = 0; i < TC * 9; i++) {
                    reg_w[i] = w_s[dot_idx * w_s_width + thread_row * TC * 9 + i];
                }
            }

            for (int k = 0; k < 9; k++) {
                for (int i = 0; i < TC; i++) {
                    for (int j = 0; j < TH; j++) {
                        thread_results[k * TC * TH + i * TH + j] += 
                            reg_w[i * 9 + k] * reg_dout[j];
                    }
                }
            }
        }
        __syncthreads();
    }

    // write out results
    int out_batch_offset = batch * C * H * W;
    for (int j = 0; j < TH; j++) {
        int hw_abs = block_col * BH + thread_col * TH + j;
        int h_abs = hw_abs / W;
        int w_abs = hw_abs % W;
        for (int i = 0; i < TC; i++) {
            int c_abs = block_row * BC + thread_row * TC + i;
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                int h_target_abs = h_abs - 1 + k_i;
                int w_target_abs = w_abs - 1 + k_j;
                if (h_target_abs >= 0 && h_target_abs < H && w_target_abs >= 0 && w_target_abs < W) {
                    int out_target_abs = out_batch_offset + c_abs * H * W + h_target_abs * W + w_target_abs;
                    int th_res_idx = k * TC * TH + i * TH + j;
                    atomicAdd(dx + out_target_abs, thread_results[th_res_idx]);
                }
            }
        }
    }
}

// I wonder how fast does a transposition kernel take for the weights
__global__ void transpose_weight_kernel(
    float* weight, float* out,
    int C, int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * O * 9) {
        int o = idx / (C * 9);
        int c = (idx % C * 9) / 9;
        int k = idx % 9;
        out[c * O * 9 + o * 9 + k] = weight[idx];
    }
}

void dx_backward(
    float* dout, float* weight,
    float* dx,
    const int B, const int C_in, const int C_out, const int H, const int W
) {
    if (H < 16) {
        const int BH = 64;
        const int TH = 4;
        const int BC = 16;
        const int TC = 4;
        const int BO = 16;
        dim3 gridDim(B * H * W / BH, C_in / BC);
        dim3 blockDim((BC * BH) / (TC * TH));
        dx_backward_kernel1<BO, BH, BC, TC, TH><<<gridDim, blockDim>>>(dout, weight, dx, B, C_in, C_out, H, W);
    } else if (C_in == 3) {
        const int BH = 64 * 4;
        const int TH = 4;
        const int BC = 3;
        const int TC = 3;
        const int BO = 16;
        dim3 gridDim(B * H * W / BH, C_in / BC);
        dim3 blockDim((BC * BH) / (TC * TH));
        dx_backward_kernel1<BO, BH, BC, TC, TH><<<gridDim, blockDim>>>(dout, weight, dx, B, C_in, C_out, H, W);
    } else if (C_out == 3) {
        const int BH = 64 * 4;
        const int TH = 4;
        const int BC = 16;
        const int TC = 4;
        const int BO = 3;
        dim3 gridDim(B * H * W / BH, C_in / BC);
        dim3 blockDim((BC * BH) / (TC * TH));
        dx_backward_kernel1<BO, BH, BC, TC, TH><<<gridDim, blockDim>>>(dout, weight, dx, B, C_in, C_out, H, W);
    } else {
        const int BH = 64 * 4;
        const int TH = 4;
        const int BC = 16;
        const int TC = 4;
        const int BO = 16;
        dim3 gridDim(B * H * W / BH, C_in / BC);
        dim3 blockDim((BC * BH) / (TC * TH));
        dx_backward_kernel1<BO, BH, BC, TC, TH><<<gridDim, blockDim>>>(dout, weight, dx, B, C_in, C_out, H, W);
    }
}

void dx_backward_cpu(
    float* dout, float* weight, float* dx,
    int B, int C, int O, int H, int W
) {
    printf("dx[0] before zeroing: %f\n", dx[0]);
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int k = 0; k < 9; k++) {
                        int k_i = k / 3;
                        int k_j = k % 3;
                        int h_target = h - 1 + k_i;
                        int w_target = w - 1 + k_j;
                        if (h_target < 0 || h_target >= H || w_target < 0 || w_target >= W) {
                            continue;
                        }
                        float out_bchw = 0.0f;
                        for (int o = 0; o < O; o++) {
                            float new_val = weight[o * C * 9 + c * 9 + k] * dout[b * O * H * W + o * H * W + h * W + w];
                            out_bchw += new_val;
                        }
                        int out_idx = b * C * H * W + c * H * W + h_target * W + w_target;
                        dx[out_idx] += out_bchw;
                    }
                }
            }
            if (b == 0 && c == 0) {
                printf("dx[0]: %f\n", dx[0]);
                return;
            }
        }
    }
}

// dweight_buf will have shape (O, C, 9, B)
// and then we have to reduce over the last dimension to get the actual weight
template <const int BC, const int BO, const int BW, const int TO, const int TC>
__global__ void dweight_backward_kernel1(
    float* dout, float* x,
    float* dweight_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // ensure that we load entire image widths into smem
    // so we can do all 9 kernels at once
    assert(BW % W == 0);

    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * C / BC, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = C / BC;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    // number of image widths we will load to smem in one round
    const int NW = BW / W;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    for (int h = 0; h < H; h += (NW - 1)) {
        // load dout into smem
        int dout_inner_offset = h * W;
        int nh = (h < H - (NW - 1)) ? NW : NW - 1;
        int dout_load_sz = nh * W * BO;
        for (int i = threadIdx.x; i < dout_load_sz / 4; i++) {
            int inner_col = i % (nh * W / 4);
            int inner_row = i / (nh * W / 4);
            float4 tmp = 
                reinterpret_cast<float4 *>(&dout[dout_block_offset + dout_inner_offset + inner_row * H * W + inner_col * 4])[0];
            dout_s[(inner_col * 4 + 0) * BO + inner_row] = tmp.x;
            dout_s[(inner_col * 4 + 1) * BO + inner_row] = tmp.y;
            dout_s[(inner_col * 4 + 2) * BO + inner_row] = tmp.z;
            dout_s[(inner_col * 4 + 3) * BO + inner_row] = tmp.w;
        }
        
        // load x to smem
        int x_inner_offset = h * W;
        int x_load_sz = nh * W * BC;
        for (int i = threadIdx.x; i < x_load_sz / 4; i++) {
            int inner_col = i % (nh * W / 4);
            int inner_row = i / (nh * W / 4);
            float4 tmp = 
                reinterpret_cast<float4 *>(&x[x_block_offset + x_inner_offset + inner_row * H * W + inner_col * 4])[0];
            x_s[(inner_col * 4 + 0) * BC + inner_row] = tmp.x;
            x_s[(inner_col * 4 + 1) * BC + inner_row] = tmp.y;
            x_s[(inner_col * 4 + 2) * BC + inner_row] = tmp.z;
            x_s[(inner_col * 4 + 3) * BC + inner_row] = tmp.w;
        }
        __syncthreads();

        // if loading the last block, fill in the last block's buffer to be 0
        if (nh == NW - 1) {
            for (int i = threadIdx.x; i < W * BO; i++) {
                int inner_col = i % W;
                int inner_row = i / W;
                dout_s[(inner_col + nh * W) * BO + inner_row] = 0.0f;
            }
            for (int i = threadIdx.x; i < W * BC; i++) {
                int inner_col = i % W;
                int inner_row = i / W;
                x_s[(inner_col + nh * W) * BC + inner_row] = 0.0f;
            } 
        }
        __syncthreads();

        // calculate per thread results
        for (int x_dot_idx = 0; x_dot_idx < BW; x_dot_idx++) {
            int x_h = x_dot_idx / W;
            int x_w = x_dot_idx % W;
            // load one pixel with TC of the C channels into reg_w
            for (int j = 0; j < TC; j++) {
                reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
            }

            // for each conv filter, load a different row and calculate the outer product
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                // calculate the corresponding (h, w) that should be loaded from dout
                int dout_h = x_h + 1 - k_i;
                int dout_w = x_w + 1 - k_j;
                int dout_dot_idx = -1;
                if (dout_h >= 0 && dout_h < NW && dout_w >= 0 && dout_w < W) {
                    // this logic is a bit tricky: for the last row in smem
                    // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                    // for kernels 4, 5, 6 we need to only compute them once
                    // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                    // so it will be computed as part of the first row next time
                    if (x_h == NW - 1 && k_i == 1) {
                        dout_dot_idx = -1;
                    } else {
                        dout_dot_idx = dout_h * W + dout_w;
                    }
                }
                // if in bounds of the dout section in smem, load the corresponding pixel from smem
                // and add the products to the thread results
                // otherwise the contribution is 0
                if (dout_dot_idx != -1) {
                    for (int i = 0; i < TO; i++) {
                        reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                    }
                    for (int j = 0; j < TC; j++) {
                        for (int i = 0; i < TO; i++) {
                            thread_results[k * TO * TC + i * TC + j] += reg_dout[i] * reg_x[j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // write out results
    for (int j = 0; j < TC; j++) {
        for (int i = 0; i < TO; i++) {
            for (int k = 0; k < 9; k++) {
                int o_abs = block_row * BO + thread_row * TO + i;
                int c_abs = block_col * BC + thread_col * TC + j;
                dweight_buf[o_abs * C * 9 * B + c_abs * 9 * B + k * B + batch] = thread_results[k * TO * TC + i * TC + j];
            }
        }
    }
}


// C * O blocks
// 288 (9 * 32) threads per block, use one warp to reduce each batch dim
__global__ void dweight_reduce_kernel(
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O
) {
    assert(gridDim.x == O * C);
    int o = blockIdx.x / C;
    int c = blockIdx.x % C;
    // move input pointer
    dweight_buf += o * C * 9 * B + c * 9 * B;
    assert(blockDim.x == 32 * 9);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float thread_sum = 0.0f;
    for (int i = lane_id; i < B; i += 32) {
        thread_sum += dweight_buf[warp_id * B + i];
    }

    // warp reduce
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});

    if (lane_id == 0) {
        dweight[o * C * 9 + c * 9 + warp_id] = warp_sum;
    }
}

void dweight_backward1(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O, const int H, const int W
) {
    const int BC = 32;
    const int BO = 32;
    const int BW = 2 * 64;
    const int TO = 4;
    const int TC = 4;
    dim3 gridDim(B * C / BC, O / BO);
    dim3 blockDim(BC * BO / (TO * TC));
    dweight_backward_kernel1<BC, BO, BW, TO, TC><<<gridDim, blockDim>>>(dout, x, dweight_buf, B, C, O, H, W);
    cudaCheck(cudaGetLastError());

    int block_size = 32 * 9;
    int n_blocks = O * C;
    dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B, C, O);
}


// instead of having to load multiples of entire widths of the image into smem
// we now only load fractions of the width each time
// so the outer loop needs to loop over the height multiples (NH - 1) as well as the width multiples (WS - 1)
// results: This did seem to help with memory access: compute intensity is higher
// right now this one is much slower than kernel 1
// I think this is because the smem loads are now not aligned, which causes a lot of uncoalesced loads
template <const int BC, const int BO, const int BW, const int TO, const int TC, const int WSZ>
__global__ void dweight_backward_kernel2(
    float* dout, float* x,
    float* dweight_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // number of heights we need to load into smem in one go 
    assert(BW % WSZ == 0);
    const int NH = BW / WSZ;
    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * C / BC, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = C / BC;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    //int debug_idx = 0;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    for (int h = 0; h < H; h += (NH - 1)) {
        for (int w_offset = 0; w_offset < W; w_offset += (WSZ - 1)) {
            // load dout into smem
            int h_offset = h * W;
            int nh = (h < H - (NH - 1)) ? NH : NH - 1;
            int wsz = (w_offset < W - (WSZ - 1)) ? WSZ : WSZ - 1;
            int dout_load_sz = nh * wsz * BO;
            for (int i = threadIdx.x; i < dout_load_sz; i++) {
                int inner_w = i % wsz;
                int inner_h = (i / wsz) % nh;
                int inner_o = i / (wsz * nh);
                dout_s[(inner_h * WSZ + inner_w) * BO + inner_o] = 
                    dout[dout_block_offset + h_offset + inner_h * W + w_offset + inner_w + inner_o * H * W];
            }
            
            // load x to smem
            int x_load_sz = nh * wsz * BC;
            for (int i = threadIdx.x; i < x_load_sz; i++) {
                int inner_w = i % wsz;
                int inner_h = (i / wsz) % nh;
                int inner_c = i / (wsz * nh);
                x_s[(inner_h * WSZ + inner_w) * BC + inner_c] = 
                    x[x_block_offset + h_offset + inner_h * W + w_offset + inner_w + inner_c * H * W];
            }

            __syncthreads();

            // if loading the last block, fill in the last block's buffer to be 0
            if (nh == NH - 1) {
                for (int i = threadIdx.x; i < WSZ * BO; i++) {
                    int inner_w = i % WSZ;
                    int inner_o = i / WSZ;
                    dout_s[(inner_w + nh * WSZ) * BO + inner_o] = 0.0f;
                }
                for (int i = threadIdx.x; i < WSZ * BC; i++) {
                    int inner_w = i % WSZ;
                    int inner_c = i / WSZ;

                    x_s[(inner_w + nh * WSZ) * BC + inner_c] = 0.0f;
                } 
            }
            // if loading the last width section, fill in the final value in the buffer to 0
            if (wsz == WSZ - 1)  {
                for (int i = threadIdx.x; i < nh * BO; i++) {
                    int inner_o = i % BO;
                    int inner_h = i / BO;
                    dout_s[(inner_h * WSZ + wsz) * BO + inner_o] = 0.0f;
                }
                for (int i = threadIdx.x; i < nh * BC; i++) {
                    int inner_c = i % BC;
                    int inner_h = i / BC;
                    x_s[(inner_h * WSZ + wsz) * BC + inner_c] = 0.0f;
                }
            }
            __syncthreads();

            // calculate per thread results
            for (int x_dot_idx = 0; x_dot_idx < WSZ * nh; x_dot_idx++) {
                int x_h = x_dot_idx / WSZ;
                int x_w = x_dot_idx % WSZ;
                // load TC of the C channels of one image pixel into reg_w
                for (int j = 0; j < TC; j++) {
                    reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
                }

                // for each conv filter, load a different row and calculate the outer product
                for (int k = 0; k < 9; k++) {
                    int k_i = k / 3;
                    int k_j = k % 3;
                    // calculate the corresponding (h, w) that should be loaded from dout
                    int dout_h = x_h + 1 - k_i;
                    int dout_w = x_w + 1 - k_j;
                    int dout_h_abs = h + dout_h;
                    int dout_w_abs = w_offset + dout_w;
                    int dout_dot_idx = -1;
                    if (dout_h >= 0 && dout_h < NH && dout_w >= 0 && dout_w < WSZ && dout_w_abs < W && dout_h_abs < H) {
                        if (x_h == NH - 1 && k_i == 1) {
                            // this logic is a bit tricky: for the last row in smem
                            // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                            // for kernels 4, 5, 6 we need to only compute them once
                            // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                            // so it will be computed as part of the first row next time
                            dout_dot_idx = -1;
                        } else if (x_w == WSZ - 1 && k_j == 1) {
                            dout_dot_idx = -1;
                        } else {
                            dout_dot_idx = dout_h * WSZ + dout_w;
                        }
                    }

                    // if in bounds of the dout section in smem, load the corresponding pixel from smem
                    // and add the products to the thread results
                    // otherwise the contribution is 0
                    if (dout_dot_idx != -1) {
                        for (int i = 0; i < TO; i++) {
                            assert(dout_dot_idx * BO + thread_row * TO + i < BO * BW);
                            reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                        }
                        for (int j = 0; j < TC; j++) {
                            for (int i = 0; i < TO; i++) {
                                float new_val = reg_dout[i] * reg_x[j];
                                thread_results[k * TO * TC + i * TC + j] += new_val;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // write out results
    for (int j = 0; j < TC; j++) {
        for (int i = 0; i < TO; i++) {
            for (int k = 0; k < 9; k++) {
                int o_abs = block_row * BO + thread_row * TO + i;
                int c_abs = block_col * BC + thread_col * TC + j;
                dweight_buf[o_abs * C * 9 * B + c_abs * 9 * B + k * B + batch] = thread_results[k * TO * TC + i * TC + j];
            }
        }
    }
}

void dweight_backward2(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O, const int H, const int W
) {
    const int BC = 32;
    const int BO = 32;
    const int WSZ = 8 + 1;
    const int BW = WSZ * 9;
    const int TO = 4;
    const int TC = 4;
    dim3 gridDim(B * C / BC, O / BO);
    dim3 blockDim(BC * BO / (TC * TO));
    dweight_backward_kernel2<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, B, C, O, H, W);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    int block_size = 32 * 9;
    int n_blocks = O * C;
    dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B, C, O);
}


// kernel 2 does something silly: because are now loading WSZ sections of widths,
// and WSZ is definitely not a nice power of 2, we need to be careful
// which thread we are using when loading global memory.
// This kernel only changes the way loading is done compared to kernel 2
// effectively we do some warp tiling when loading from global mem
// kernel 3 is now quite fast (2.6ms)
// profiling tells us that we are now using very few blocks, so seems like a good idea
// to spread out the different width sections on different blocks and do a bigger reduce afterwards
template <const int BC, const int BO, const int BW, const int TO, const int TC, const int WSZ>
__global__ void dweight_backward_kernel3(
    float* dout, float* x,
    float* dweight_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // number of heights we need to load into smem in one go 
    const int NH = BW / WSZ;
    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * C / BC, O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_col = C / BC;
    assert(gridDim.x == B * blocks_per_col);
    const int batch = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;

    // warp indices for loading from global mem
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // make sure that the width sections can be loaded by a single warp
    assert(WSZ <= 32);

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    //int debug_idx = 0;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    for (int h = 0; h < H; h += (NH - 1)) {
        for (int w_offset = 0; w_offset < W; w_offset += (WSZ - 1)) {
            // load dout into smem
            int h_offset = h * W;
            int nh = (h < H - (NH - 1)) ? NH : NH - 1;
            int wsz = (w_offset < W - (WSZ - 1)) ? WSZ : WSZ - 1;
            int o_per_warp = ceil_div(BO, n_warps);
            if (lane_id < wsz) {
                for (int warp_o = 0; warp_o < o_per_warp; warp_o++) {
                    int inner_o = warp_id * o_per_warp + warp_o;
                    if (inner_o < BO) {
                        for (int inner_h = 0; inner_h < nh; inner_h++) {
                            int inner_w = lane_id;
                            dout_s[(inner_h * WSZ + inner_w) * BO + inner_o] = 
                                dout[dout_block_offset + inner_o * H * W + h_offset + inner_h * W + w_offset + inner_w];
                        }
                    }
                }
            }

            int c_per_warp = ceil_div(BC, n_warps);
            if (lane_id < wsz) {
                for (int warp_c = 0; warp_c < c_per_warp; warp_c++) {
                    int inner_c = warp_id * c_per_warp + warp_c;
                    if (inner_c < BC) {
                        for (int inner_h = 0; inner_h < nh; inner_h++) {
                            int inner_w = lane_id;
                            x_s[(inner_h * WSZ + inner_w) * BC + inner_c] = 
                                x[x_block_offset + inner_c * H * W + h_offset + inner_h * W + w_offset + inner_w];
                        }
                    }
                }
            }

            __syncthreads();
            

            // if loading the last block, fill in the last block's buffer to be 0
            if (nh == NH - 1) {
                for (int i = threadIdx.x; i < WSZ * BO; i++) {
                    int inner_w = i % WSZ;
                    int inner_o = i / WSZ;
                    dout_s[(inner_w + nh * WSZ) * BO + inner_o] = 0.0f;
                }
                for (int i = threadIdx.x; i < WSZ * BC; i++) {
                    int inner_w = i % WSZ;
                    int inner_c = i / WSZ;

                    x_s[(inner_w + nh * WSZ) * BC + inner_c] = 0.0f;
                } 
            }
            // if loading the last width section, fill in the final value in the buffer to 0
            if (wsz == WSZ - 1)  {
                for (int i = threadIdx.x; i < nh * BO; i++) {
                    int inner_o = i % BO;
                    int inner_h = i / BO;
                    dout_s[(inner_h * WSZ + wsz) * BO + inner_o] = 0.0f;
                }
                for (int i = threadIdx.x; i < nh * BC; i++) {
                    int inner_c = i % BC;
                    int inner_h = i / BC;
                    x_s[(inner_h * WSZ + wsz) * BC + inner_c] = 0.0f;
                }
            }
            __syncthreads();

            // calculate per thread results
            for (int x_dot_idx = 0; x_dot_idx < WSZ * nh; x_dot_idx++) {
                int x_h = x_dot_idx / WSZ;
                int x_w = x_dot_idx % WSZ;
                // load TC of the C channels of one image pixel into reg_w
                for (int j = 0; j < TC; j++) {
                    reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
                }

                // for each conv filter, load a different row and calculate the outer product
                for (int k = 0; k < 9; k++) {
                    int k_i = k / 3;
                    int k_j = k % 3;
                    // calculate the corresponding (h, w) that should be loaded from dout
                    int dout_h = x_h + 1 - k_i;
                    int dout_w = x_w + 1 - k_j;
                    int dout_h_abs = h + dout_h;
                    int dout_w_abs = w_offset + dout_w;
                    int dout_dot_idx = -1;
                    if (dout_h >= 0 && dout_h < NH && dout_w >= 0 && dout_w < WSZ && dout_w_abs < W && dout_h_abs < H) {
                        if (x_h == NH - 1 && k_i == 1) {
                            // this logic is a bit tricky: for the last row in smem
                            // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                            // for kernels 4, 5, 6 we need to only compute them once
                            // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                            // so it will be computed as part of the first row next time
                            dout_dot_idx = -1;
                        } else if (x_w == WSZ - 1 && k_j == 1) {
                            dout_dot_idx = -1;
                        } else {
                            dout_dot_idx = dout_h * WSZ + dout_w;
                        }
                    }

                    // if in bounds of the dout section in smem, load the corresponding pixel from smem
                    // and add the products to the thread results
                    // otherwise the contribution is 0
                    if (dout_dot_idx != -1) {
                        for (int i = 0; i < TO; i++) {
                            reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                        }
                        for (int j = 0; j < TC; j++) {
                            for (int i = 0; i < TO; i++) {
                                float new_val = reg_dout[i] * reg_x[j];
                                thread_results[k * TO * TC + i * TC + j] += new_val;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // write out results
    for (int j = 0; j < TC; j++) {
        for (int i = 0; i < TO; i++) {
            for (int k = 0; k < 9; k++) {
                int o_abs = block_row * BO + thread_row * TO + i;
                int c_abs = block_col * BC + thread_col * TC + j;
                dweight_buf[o_abs * C * 9 * B + c_abs * 9 * B + k * B + batch] = thread_results[k * TO * TC + i * TC + j];
            }
        }
    }
}

void dweight_backward3(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O, const int H, const int W
) {
    const int BC = 64;
    const int BO = 64;
    const int WSZ = 8 + 1;
    const int BW = WSZ * 9;
    const int TO = 4;
    const int TC = 4;
    assert(BW % WSZ == 0);
    dim3 gridDim(B * C / BC, O / BO);
    dim3 blockDim(BC * BO / (TC * TO));
    dweight_backward_kernel3<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, B, C, O, H, W);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    int block_size = 32 * 9;
    int n_blocks = O * C;
    dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B, C, O);
}


// trying to improve on kernel 3 by splitting the widths of each image
// to be calculated on different blocks
// Output will now be of shape (O, C, 9, B, W / (WSZ - 1))
// and we will need to reduce over the last two dimensions
template <const int BC, const int BO, const int BW, const int TO, const int TC, const int WSZ>
__global__ void dweight_backward_kernel4(
    float* dout, float* x,
    float* dweight_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // number of heights we need to load into smem in one go 
    const int NH = BW / WSZ;
    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * (W / (WSZ - 1)) * (C / BC), O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_chan = C / BC;
    const int blocks_per_width = W / (WSZ - 1);
    const int block_w = blockIdx.x % blocks_per_width;
    const int block_col = (blockIdx.x / blocks_per_width) % blocks_per_chan;
    const int batch = blockIdx.x / (blocks_per_chan * blocks_per_width);
    const int w_offset = block_w * (WSZ - 1);

    // warp indices for loading from global mem
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // make sure that the width sections can be loaded by a single warp
    assert(WSZ <= 32);

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    //int debug_idx = 0;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    for (int h = 0; h < H; h += (NH - 1)) {
        // load dout into smem
        int h_offset = h * W;
        int nh = (h < H - (NH - 1)) ? NH : NH - 1;
        int wsz = (w_offset < W - (WSZ - 1)) ? WSZ : WSZ - 1;
        int o_per_warp = ceil_div(BO, n_warps);
        if (lane_id < wsz) {
            for (int warp_o = 0; warp_o < o_per_warp; warp_o++) {
                int inner_o = warp_id * o_per_warp + warp_o;
                if (inner_o < BO) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        int inner_w = lane_id;
                        dout_s[(inner_h * WSZ + inner_w) * BO + inner_o] = 
                            dout[dout_block_offset + inner_o * H * W + h_offset + inner_h * W + w_offset + inner_w];
                    }
                }
            }
        }

        int c_per_warp = ceil_div(BC, n_warps);
        if (lane_id < wsz) {
            for (int warp_c = 0; warp_c < c_per_warp; warp_c++) {
                int inner_c = warp_id * c_per_warp + warp_c;
                if (inner_c < BC) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        int inner_w = lane_id;
                        x_s[(inner_h * WSZ + inner_w) * BC + inner_c] = 
                            x[x_block_offset + inner_c * H * W + h_offset + inner_h * W + w_offset + inner_w];
                    }
                }
            }
        }

        __syncthreads();
        

        // if loading the last block, fill in the last block's buffer to be 0
        if (nh == NH - 1) {
            for (int i = threadIdx.x; i < WSZ * BO; i++) {
                int inner_w = i % WSZ;
                int inner_o = i / WSZ;
                dout_s[(inner_w + nh * WSZ) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < WSZ * BC; i++) {
                int inner_w = i % WSZ;
                int inner_c = i / WSZ;

                x_s[(inner_w + nh * WSZ) * BC + inner_c] = 0.0f;
            } 
        }
        // if loading the last width section, fill in the final value in the buffer to 0
        if (wsz == WSZ - 1)  {
            for (int i = threadIdx.x; i < nh * BO; i++) {
                int inner_o = i % BO;
                int inner_h = i / BO;
                dout_s[(inner_h * WSZ + wsz) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < nh * BC; i++) {
                int inner_c = i % BC;
                int inner_h = i / BC;
                x_s[(inner_h * WSZ + wsz) * BC + inner_c] = 0.0f;
            }
        }
        __syncthreads();

        // calculate per thread results
        for (int x_dot_idx = 0; x_dot_idx < WSZ * nh; x_dot_idx++) {
            int x_h = x_dot_idx / WSZ;
            int x_w = x_dot_idx % WSZ;
            // load TC of the C channels of one image pixel into reg_w
            for (int j = 0; j < TC; j++) {
                reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
            }

            // for each conv filter, load a different row and calculate the outer product
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                // calculate the corresponding (h, w) that should be loaded from dout
                int dout_h = x_h + 1 - k_i;
                int dout_w = x_w + 1 - k_j;
                int dout_h_abs = h + dout_h;
                int dout_w_abs = w_offset + dout_w;
                int dout_dot_idx = -1;
                if (dout_h >= 0 && dout_h < NH && dout_w >= 0 && dout_w < WSZ && dout_w_abs < W && dout_h_abs < H) {
                    if (x_h == NH - 1 && k_i == 1) {
                        // this logic is a bit tricky: for the last row in smem
                        // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                        // for kernels 4, 5, 6 we need to only compute them once
                        // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                        // so it will be computed as part of the first row next time
                        dout_dot_idx = -1;
                    } else if (x_w == WSZ - 1 && k_j == 1) {
                        dout_dot_idx = -1;
                    } else {
                        dout_dot_idx = dout_h * WSZ + dout_w;
                    }
                }

                // if in bounds of the dout section in smem, load the corresponding pixel from smem
                // and add the products to the thread results
                // otherwise the contribution is 0
                if (dout_dot_idx != -1) {
                    for (int i = 0; i < TO; i++) {
                        reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                    }
                    for (int j = 0; j < TC; j++) {
                        for (int i = 0; i < TO; i++) {
                            float new_val = reg_dout[i] * reg_x[j];
                            thread_results[k * TO * TC + i * TC + j] += new_val;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // write out results
    for (int j = 0; j < TC; j++) {
        for (int i = 0; i < TO; i++) {
            for (int k = 0; k < 9; k++) {
                int o_abs = block_row * BO + thread_row * TO + i;
                int c_abs = block_col * BC + thread_col * TC + j;
                dweight_buf[o_abs * C * 9 * B * blocks_per_width + c_abs * 9 * B * blocks_per_width + k * B * blocks_per_width + batch * blocks_per_width + block_w] = thread_results[k * TO * TC + i * TC + j];
            }
        }
    }
}

void dweight_backward4(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O, const int H, const int W
) {
    const int BC = 64;
    const int BO = 64;
    const int WSZ = 8 + 1;
    const int BW = WSZ * 9;
    const int TO = 4;
    const int TC = 4;
    assert(BW % WSZ == 0);
    dim3 gridDim(B * (W / (WSZ - 1)) * (C / BC), O / BO);
    dim3 blockDim(BC * BO / (TC * TO));
    dweight_backward_kernel4<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, B, C, O, H, W);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    int block_size = 32 * 9;
    int n_blocks = O * C;
    dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B * W / (WSZ - 1), C, O);
}

// same as kernel 4, but vectorize loads to speed up smem load
// unfortunately as is this doesn't give any speedup
// probably because we are now using too few threads to load
// I could try to shuffle and make each warp load more heights as well, but that doesn't seem first order right now
template <const int BC, const int BO, const int BW, const int TO, const int TC, const int WSZ>
__global__ void dweight_backward_kernel5(
    float* dout, float* x,
    float* dweight_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // number of heights we need to load into smem in one go 
    const int NH = BW / WSZ;
    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * (W / (WSZ - 1)) * (C / BC), O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_chan = C / BC;
    const int blocks_per_width = W / (WSZ - 1);
    const int block_w = blockIdx.x % blocks_per_width;
    const int block_col = (blockIdx.x / blocks_per_width) % blocks_per_chan;
    const int batch = blockIdx.x / (blocks_per_chan * blocks_per_width);
    const int w_offset = block_w * (WSZ - 1);

    // warp indices for loading from global mem
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // make sure that the width sections can be loaded by a single warp
    assert(WSZ <= 32);

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    //int debug_idx = 0;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    for (int h = 0; h < H; h += (NH - 1)) {
        // load dout into smem
        int h_offset = h * W;
        int nh = (h < H - (NH - 1)) ? NH : NH - 1;
        int wsz = (w_offset < W - (WSZ - 1)) ? WSZ : WSZ - 1;
        int o_per_warp = ceil_div(BO, n_warps);
        if (lane_id < ceil_div(wsz, 4)) {
            for (int warp_o = 0; warp_o < o_per_warp; warp_o++) {
                int inner_o = warp_id * o_per_warp + warp_o;
                if (inner_o < BO) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        float4 tmp = 
                            reinterpret_cast<float4 *>(&dout[dout_block_offset + inner_o * H * W + h_offset + inner_h * W + w_offset + lane_id * 4])[0];
                        dout_s[(inner_h * WSZ + lane_id * 4 + 0) * BO + inner_o] = tmp.x;
                        if (lane_id < wsz / 4) {
                            dout_s[(inner_h * WSZ + lane_id * 4 + 1) * BO + inner_o] = tmp.y;
                            dout_s[(inner_h * WSZ + lane_id * 4 + 2) * BO + inner_o] = tmp.z;
                            dout_s[(inner_h * WSZ + lane_id * 4 + 3) * BO + inner_o] = tmp.w;
                        }
                    }
                }
            }
        }

        int c_per_warp = ceil_div(BC, n_warps);
        if (lane_id < ceil_div(wsz, 4)) {
            for (int warp_c = 0; warp_c < c_per_warp; warp_c++) {
                int inner_c = warp_id * c_per_warp + warp_c;
                if (inner_c < BC) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        float4 tmp = 
                            reinterpret_cast<float4 *>(&x[x_block_offset + inner_c * H * W + h_offset + inner_h * W + w_offset + lane_id * 4])[0];
                        x_s[(inner_h * WSZ + lane_id * 4 + 0) * BC + inner_c] = tmp.x;
                        if (lane_id < wsz / 4) {
                            x_s[(inner_h * WSZ + lane_id * 4 + 1) * BC + inner_c] = tmp.y;
                            x_s[(inner_h * WSZ + lane_id * 4 + 2) * BC + inner_c] = tmp.z;
                            x_s[(inner_h * WSZ + lane_id * 4 + 3) * BC + inner_c] = tmp.w;
                        }
                    }
                }
            }
        }

        __syncthreads();
        

        // if loading the last block, fill in the last block's buffer to be 0
        if (nh == NH - 1) {
            for (int i = threadIdx.x; i < WSZ * BO; i++) {
                int inner_w = i % WSZ;
                int inner_o = i / WSZ;
                dout_s[(inner_w + nh * WSZ) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < WSZ * BC; i++) {
                int inner_w = i % WSZ;
                int inner_c = i / WSZ;

                x_s[(inner_w + nh * WSZ) * BC + inner_c] = 0.0f;
            } 
        }
        // if loading the last width section, fill in the final value in the buffer to 0
        if (wsz == WSZ - 1)  {
            for (int i = threadIdx.x; i < nh * BO; i++) {
                int inner_o = i % BO;
                int inner_h = i / BO;
                dout_s[(inner_h * WSZ + wsz) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < nh * BC; i++) {
                int inner_c = i % BC;
                int inner_h = i / BC;
                x_s[(inner_h * WSZ + wsz) * BC + inner_c] = 0.0f;
            }
        }
        __syncthreads();

        // calculate per thread results
        for (int x_dot_idx = 0; x_dot_idx < WSZ * nh; x_dot_idx++) {
            int x_h = x_dot_idx / WSZ;
            int x_w = x_dot_idx % WSZ;
            // load TC of the C channels of one image pixel into reg_w
            for (int j = 0; j < TC; j++) {
                reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
            }

            // for each conv filter, load a different row and calculate the outer product
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                // calculate the corresponding (h, w) that should be loaded from dout
                int dout_h = x_h + 1 - k_i;
                int dout_w = x_w + 1 - k_j;
                int dout_h_abs = h + dout_h;
                int dout_w_abs = w_offset + dout_w;
                int dout_dot_idx = -1;
                if (dout_h >= 0 && dout_h < NH && dout_w >= 0 && dout_w < WSZ && dout_w_abs < W && dout_h_abs < H) {
                    if (x_h == NH - 1 && k_i == 1) {
                        // this logic is a bit tricky: for the last row in smem
                        // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                        // for kernels 4, 5, 6 we need to only compute them once
                        // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                        // so it will be computed as part of the first row next time
                        dout_dot_idx = -1;
                    } else if (x_w == WSZ - 1 && k_j == 1) {
                        dout_dot_idx = -1;
                    } else {
                        dout_dot_idx = dout_h * WSZ + dout_w;
                    }
                }

                // if in bounds of the dout section in smem, load the corresponding pixel from smem
                // and add the products to the thread results
                // otherwise the contribution is 0
                if (dout_dot_idx != -1) {
                    for (int i = 0; i < TO; i++) {
                        reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                    }
                    for (int j = 0; j < TC; j++) {
                        for (int i = 0; i < TO; i++) {
                            float new_val = reg_dout[i] * reg_x[j];
                            thread_results[k * TO * TC + i * TC + j] += new_val;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // write out results
    for (int j = 0; j < TC; j++) {
        for (int i = 0; i < TO; i++) {
            for (int k = 0; k < 9; k++) {
                int o_abs = block_row * BO + thread_row * TO + i;
                int c_abs = block_col * BC + thread_col * TC + j;
                dweight_buf[o_abs * C * 9 * B * blocks_per_width + c_abs * 9 * B * blocks_per_width + k * B * blocks_per_width + batch * blocks_per_width + block_w] = thread_results[k * TO * TC + i * TC + j];
            }
        }
    }
}

void dweight_backward5(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    const int B, const int C, const int O, const int H, const int W
) {
    const int BC = 64;
    const int BO = 64;
    const int WSZ = 8 + 1;
    const int BW = WSZ * 9;
    const int TO = 4;
    const int TC = 4;
    assert(BW % WSZ == 0);
    dim3 gridDim(B * (W / (WSZ - 1)) * (C / BC), O / BO);
    dim3 blockDim(BC * BO / (TC * TO));
    dweight_backward_kernel5<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, B, C, O, H, W);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    int block_size = 32 * 9;
    int n_blocks = O * C;
    dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B * W / (WSZ - 1), C, O);
}

// same as dweight backward 4, except we also do some of the reduce for dbias
// so it doesn't have to reduce an entire BxHxW length for each O channel
template <const int BC, const int BO, const int BW, const int TO, const int TC, const int WSZ>
__global__ void dweight_dbias_backward_kernel1(
    float* dout, float* x,
    float* dweight_buf, float* dbias_buf,
    const int B, const int C, const int O, const int H, const int W
) {
    // number of heights we need to load into smem in one go 
    const int NH = BW / WSZ;
    // n_threads per block = (BC * BO) / (TO * TC)
    // gridDim will be (B * (W / (WSZ - 1)) * (C / BC), O / BO)
    const int block_row = blockIdx.y;
    const int blocks_per_chan = C / BC;
    const int blocks_per_width = W / (WSZ - 1);
    const int block_w = blockIdx.x % blocks_per_width;
    const int block_col = (blockIdx.x / blocks_per_width) % blocks_per_chan;
    const int batch = blockIdx.x / (blocks_per_chan * blocks_per_width);
    const int w_offset = block_w * (WSZ - 1);

    // warp indices for loading from global mem
    const int n_warps = ceil_div((int)blockDim.x, 32);
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // make sure that the width sections can be loaded by a single warp
    assert(WSZ <= 32);

    // index for the TO*TC thread block
    const int threads_per_col = BC / TC;
    const int thread_col = threadIdx.x % threads_per_col;
    const int thread_row = threadIdx.x / threads_per_col;

    // smem for dout and x
    __shared__ float dout_s[BO * BW];
    __shared__ float x_s[BC * BW];

    // registers for local mem and local results
    float thread_results[TO * TC * 9] = {0.0};
    float reg_dout[TO] = {0.0};
    float reg_x[TC] = {0.0};
    float thread_bias_results[TO] = {0.0};

    // block offsets
    const int dout_block_offset = batch * O * H * W + block_row * BO * H * W;
    const int x_block_offset = batch * C * H * W + block_col * BC * H * W;

    //int debug_idx = 0;

    // we will fuse the calculation for all 9 kernel filters
    // which means we will have to reload some of the image widths
    // we can load NW blocks into smem in one go, which means we will be stepping
    // over the H dimension in (NW - 1) steps
    // this is the outer loop over H
    int h_step = (NH < H) ? NH - 1 : H;
    // if NH == H, then we are loading the entire image in one go, so step by H
    for (int h = 0; h < H; h += h_step) {
        // load dout into smem
        int h_offset = h * W;
        int nh = (h < H - (NH - 1)) ? NH : NH - 1;
        int wsz = (w_offset < W - (WSZ - 1)) ? WSZ : WSZ - 1;
        int o_per_warp = ceil_div(BO, n_warps);
        if (lane_id < wsz) {
            for (int warp_o = 0; warp_o < o_per_warp; warp_o++) {
                int inner_o = warp_id * o_per_warp + warp_o;
                if (inner_o < BO) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        int inner_w = lane_id;
                        dout_s[(inner_h * WSZ + inner_w) * BO + inner_o] = 
                            dout[dout_block_offset + inner_o * H * W + h_offset + inner_h * W + w_offset + inner_w];
                    }
                }
            }
        }

        int c_per_warp = ceil_div(BC, n_warps);
        if (lane_id < wsz) {
            for (int warp_c = 0; warp_c < c_per_warp; warp_c++) {
                int inner_c = warp_id * c_per_warp + warp_c;
                if (inner_c < BC) {
                    for (int inner_h = 0; inner_h < nh; inner_h++) {
                        int inner_w = lane_id;
                        x_s[(inner_h * WSZ + inner_w) * BC + inner_c] = 
                            x[x_block_offset + inner_c * H * W + h_offset + inner_h * W + w_offset + inner_w];
                    }
                }
            }
        }

        __syncthreads();
        

        // if loading the last block, fill in the last block's buffer to be 0
        if (nh == NH - 1) {
            for (int i = threadIdx.x; i < WSZ * BO; i++) {
                int inner_w = i % WSZ;
                int inner_o = i / WSZ;
                dout_s[(inner_w + nh * WSZ) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < WSZ * BC; i++) {
                int inner_w = i % WSZ;
                int inner_c = i / WSZ;

                x_s[(inner_w + nh * WSZ) * BC + inner_c] = 0.0f;
            } 
        }
        // if loading the last width section, fill in the final value in the buffer to 0
        if (wsz == WSZ - 1) {
            for (int i = threadIdx.x; i < nh * BO; i++) {
                int inner_o = i % BO;
                int inner_h = i / BO;
                dout_s[(inner_h * WSZ + wsz) * BO + inner_o] = 0.0f;
            }
            for (int i = threadIdx.x; i < nh * BC; i++) {
                int inner_c = i % BC;
                int inner_h = i / BC;
                x_s[(inner_h * WSZ + wsz) * BC + inner_c] = 0.0f;
            }
        }
        __syncthreads();

        // calculate per thread results
        for (int x_dot_idx = 0; x_dot_idx < WSZ * nh; x_dot_idx++) {
            int x_h = x_dot_idx / WSZ;
            int x_w = x_dot_idx % WSZ;
            // load TC of the C channels of one image pixel into reg_w
            for (int j = 0; j < TC; j++) {
                reg_x[j] = x_s[x_dot_idx * BC + thread_col * TC + j];
            }

            // for each conv filter, load a different row and calculate the outer product
            for (int k = 0; k < 9; k++) {
                int k_i = k / 3;
                int k_j = k % 3;
                // calculate the corresponding (h, w) that should be loaded from dout
                int dout_h = x_h + 1 - k_i;
                int dout_w = x_w + 1 - k_j;
                int dout_h_abs = h + dout_h;
                int dout_w_abs = w_offset + dout_w;
                int dout_dot_idx = -1;
                if (dout_h >= 0 && dout_h < NH && dout_w >= 0 && dout_w < WSZ && dout_w_abs < W && dout_h_abs < H) {
                    if (x_h == NH - 1 && k_i == 1 && H > NH) {
                        // this logic is a bit tricky: for the last row in smem
                        // because it will be loaded as part of the next block as well, and it will be the first row in the next block
                        // for kernels 4, 5, 6 we need to only compute them once
                        // this statement ensures that we don't compute the 4, 5, 6 kernels for the last row in smem
                        // so it will be computed as part of the first row next time
                        dout_dot_idx = -1;
                    } else if (x_w == WSZ - 1 && k_j == 1 && W > WSZ) {
                        // if x_w is at the last pixel, and we have more than one width sections, and we are not are the last section
                        // then don't compute 1, 4, 7 kernels and wait for the next section to compute them
                        dout_dot_idx = -1;
                    } else {
                        dout_dot_idx = dout_h * WSZ + dout_w;
                    }
                }

                // if in bounds of the dout section in smem, load the corresponding pixel from smem
                // and add the products to the thread results
                // otherwise the contribution is 0
                if (dout_dot_idx != -1) {
                    for (int i = 0; i < TO; i++) {
                        reg_dout[i] = dout_s[dout_dot_idx * BO + thread_row * TO + i];
                    }
                    for (int j = 0; j < TC; j++) {
                        for (int i = 0; i < TO; i++) {
                            float new_val = reg_dout[i] * reg_x[j];
                            thread_results[k * TO * TC + i * TC + j] += new_val;
                        }
                    }
                    // update dbias when k = 4, where dout_h and dout_w match x_h and x_w
                    if (k == 4 && block_col == 0) {
                        for (int i = 0; i < TO; i++) {
                            thread_bias_results[i] += reg_dout[i];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // write out dweight results
    for (int i = 0; i < TO; i++) {
        int o_abs = block_row * BO + thread_row * TO + i;
        for (int j = 0; j < TC; j++) {
            int c_abs = block_col * BC + thread_col * TC + j;
            for (int k = 0; k < 9; k++) {
                dweight_buf[o_abs * C * 9 * B * blocks_per_width + c_abs * 9 * B * blocks_per_width + k * B * blocks_per_width + batch * blocks_per_width + block_w] = thread_results[k * TO * TC + i * TC + j];
            }
        }
        // write out dbias results
        if (block_col == 0) {
            dbias_buf[o_abs * B * blocks_per_width + batch * blocks_per_width + block_w] = thread_bias_results[i];
        }
    }
}


__global__ void dbias_reduce_kernel1(
    float* dbias_buf, float* dbias,
    const int B
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    dbias_buf += blockIdx.x * B;

    int lane_id = threadIdx.x % 32;

    float db_thread_sum = 0.0f;
    for (int i = lane_id; i < B; i += 32) {
        db_thread_sum += dbias_buf[i];
    }

    float db_warp_sum = cg::reduce(warp, db_thread_sum, cg::plus<float>{});

    if (lane_id == 0) {
        dbias[blockIdx.x] = db_warp_sum;
    }
}

void dweight_dbias_backward1(
    float* dout, float* x,
    float* dweight_buf, float* dweight,
    float* dbias_buf, float* dbias,
    const int B, const int C, const int O, const int H, const int W
) {
    if (H * W == 64) {
        const int BC = 64;
        const int BO = 64;
        const int WSZ = 8;
        const int BW = WSZ * 8;
        const int TO = 4;
        const int TC = 4;
        assert(BW % WSZ == 0);
        dim3 gridDim(B * (C / BC), O / BO);
        dim3 blockDim(BC * BO / (TC * TO));
        dweight_dbias_backward_kernel1<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, dbias_buf, B, C, O, H, W);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());

        int block_size = 32 * 9;
        int n_blocks = O * C;
        dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B, C, O);
        dbias_reduce_kernel1<<<O, 32>>>(dbias_buf, dbias, B);
    } else if (C == 3) {
        const int BC = 3;
        const int BO = 64;
        const int WSZ = 8 + 1;
        const int BW = WSZ * 9;
        const int TO = 4;
        const int TC = 3;
        assert(BW % WSZ == 0);
        dim3 gridDim(B * (W / (WSZ - 1)) * (C / BC), O / BO);
        dim3 blockDim(BC * BO / (TC * TO));
        dweight_dbias_backward_kernel1<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, dbias_buf, B, C, O, H, W);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());

        int block_size = 32 * 9;
        int n_blocks = O * C;
        dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B * W / (WSZ - 1), C, O);
        dbias_reduce_kernel1<<<O, 32>>>(dbias_buf, dbias, B * W / (WSZ - 1));
    } else if (O == 3) {
        const int BC = 64;
        const int BO = 3;
        const int WSZ = 8 + 1;
        const int BW = WSZ * 9;
        const int TO = 3;
        const int TC = 4;
        assert(BW % WSZ == 0);
        dim3 gridDim(B * (W / (WSZ - 1)) * (C / BC), O / BO);
        dim3 blockDim(BC * BO / (TC * TO));
        dweight_dbias_backward_kernel1<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, dbias_buf, B, C, O, H, W);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());

        int block_size = 32 * 9;
        int n_blocks = O * C;
        dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B * W / (WSZ - 1), C, O);
        dbias_reduce_kernel1<<<O, 32>>>(dbias_buf, dbias, B * W / (WSZ - 1));
    } else {
        const int BC = 64;
        const int BO = 64;
        const int WSZ = 8 + 1;
        const int BW = WSZ * 9;
        const int TO = 4;
        const int TC = 4;
        assert(BW % WSZ == 0);
        dim3 gridDim(B * (W / (WSZ - 1)) * (C / BC), O / BO);
        dim3 blockDim(BC * BO / (TC * TO));
        dweight_dbias_backward_kernel1<BC, BO, BW, TO, TC, WSZ><<<gridDim, blockDim>>>(dout, x, dweight_buf, dbias_buf, B, C, O, H, W);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());

        int block_size = 32 * 9;
        int n_blocks = O * C;
        dweight_reduce_kernel<<<n_blocks, block_size>>>(dweight_buf, dweight, B * W / (WSZ - 1), C, O);
        dbias_reduce_kernel1<<<O, 32>>>(dbias_buf, dbias, B * W / (WSZ - 1));
    }
}

void conv2d_k3_backward2(
    float* dout, float* x, float* weight,
    float* dweight_buf, float* dbias_buf,
    float* dx, float* dweight, float* dbias,
    const int B, const int C_in, const int C_out, const int H, const int W
) {
    // dx currently requires the output buffer to be zeroed
    cudaCheck(cudaMemset(dx, 0, B * C_in * H * W * sizeof(float)));
    dx_backward(dout, weight, dx, B, C_in, C_out, H, W);
    dweight_dbias_backward1(dout, x, dweight_buf, dweight, dbias_buf, dbias, B, C_in, C_out, H, W);
}
// ----------------------------------------------------------------------------
// kernel launcher
void conv2d_k3_forward(
    cublasHandle_t cublas_handle,
    int kernel_num,
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int C_out, int H, int W,
    int block_size
) {
    switch (kernel_num) {
        case 1:
            conv2d_k3_forward1(cublas_handle, x, weight, bias, out, B, C_in, C_out, H, W, block_size);
            break;
        case 2:
            conv2d_k3_forward2(cublas_handle, x, weight, bias, out, B, C_in, C_out, H, W, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            break;
    }
}


#ifndef LINKING
int main( int argc, char **argv) {
    const int B = 32;
    const int C_in = 192;
    const int C_out = 64;
    const int H = 64;
    const int W = 64;

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);
    printf("device registers per block: %d\n", deviceProp.regsPerBlock);
    printf("shared mem per block: %ld\n", deviceProp.sharedMemPerBlock);

    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    // create host memory to load data
    float* x = (float*)malloc(B * C_in * H * W * sizeof(float));
    float* weight = (float*)malloc(C_out * C_in * 3 * 3 * sizeof(float));
    float* bias = (float*)malloc(C_out * sizeof(float));
    float* out = (float*)malloc(B * C_out * H * W * sizeof(float));
    float* dout = (float*)malloc(B * C_out * H * W * sizeof(float));
    float* dx = (float*)malloc(B * C_in * H * W * sizeof(float));
    float* dweight = (float*)malloc(C_out * C_in * 3 * 3 * sizeof(float));
    float* dbias = (float*)malloc(C_out * sizeof(float));
    // debug
    float* dout_perm = (float*)malloc(B * C_out * H * W * sizeof(float));
    float* x_tiled = (float*)malloc(B * H * W * C_in * 9 * sizeof(float));

    // read saved output
    FILE *file = fopen("conv2d_k3.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), B * C_in * H * W, file);
    freadCheck(weight, sizeof(float), C_out * C_in * 3 * 3, file);
    freadCheck(bias, sizeof(float), C_out, file);
    freadCheck(out, sizeof(float), B * C_out * H * W, file);
    freadCheck(dout, sizeof(float), B * C_out * H * W, file);
    freadCheck(dx, sizeof(float), B * C_in * H * W, file);
    freadCheck(dweight, sizeof(float), C_out * C_in * 3 * 3, file);
    freadCheck(dbias, sizeof(float), C_out, file);
    freadCheck(dout_perm, sizeof(float), B * C_out * H * W, file);
    freadCheck(x_tiled, sizeof(float), B * H * W * C_in * 9, file);
    fclose(file);

    // allocate device memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_dout, *d_dx, *d_dweight, *d_dbias;
    cudaCheck(cudaMalloc(&d_x, B * C_in * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C_out * C_in * 3 * 3 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C_out * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * C_out * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * C_out * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, B * C_in * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, C_out * C_in * 3 * 3 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, C_out * sizeof(float)));

    cudaCheck(cudaMemset(d_out, 0, B * C_out * H * W * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C_out * C_in * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C_out * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * C_out * H * W * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel number from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Running kernel %d\n", kernel_num);

    printf("Checking forward pass\n");
    float forward_acc = 1e-1;

    //for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    //    int block_size = block_sizes[j];
    //    printf("\nBlock size: %d\n", block_size);
    //    conv2d_k3_forward(cublas_handle, kernel_num, d_x, d_weight, d_bias, d_out, B, C_in, C_out, H, W, block_size);
    //    validate_result(d_out, out, "out", B * C_out * H * W, forward_acc);
    //}
    //printf("\nChecking cpu forward\n");
    //float* out_cpu_buf = (float*)malloc(B * C_out * H * W * sizeof(float));
    //conv_k3_forward_cpu(x, weight, bias, out_cpu_buf, B, C_in, C_out, H, W);
    //float* out_gpu_buf;
    //cudaCheck(cudaMalloc(&out_gpu_buf, B * C_out * H * W * sizeof(float)));
    //cudaCheck(cudaMemcpy(out_gpu_buf, out, B * C_out * H * W * sizeof(float), cudaMemcpyHostToDevice));
    //validate_result(out_gpu_buf, out, "out cpu", B * C_out * H * W);
    //cudaCheck(cudaFree(out_gpu_buf));
    //free(out_cpu_buf);

    printf("\nChecking new kernel\n");
    conv2d_k3_forward3(d_x, d_weight, d_bias, d_out, B, C_in, C_out, H, W);
    cudaCheck(cudaGetLastError());
    printf("\nChecking new kernel results\n");
    validate_result(d_out, out, "out", B * C_out * H * W, forward_acc);
    printf("Forward pass successful\n\n");

    printf("\nmanually benchmark new forward kernel\n");
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
        conv2d_k3_forward3(d_x, d_weight, d_bias, d_out, B, C_in, C_out, H, W);
        cudaCheck(cudaEventRecord(end, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        new_time += elapsed_time;
    }
    new_time /= repeat_times;
    printf("new kernel time: %.4f ms\n", new_time);

    printf("Checking backward pass\n");

    float dw_acc = 1e-2;
    float* d_dweight_buf;
    cudaCheck(cudaMalloc(&d_dweight_buf, C_out * C_in * 3 * 3 * B * 32 * sizeof(float)));
    float* d_dbias_buf;
    cudaCheck(cudaMalloc(&d_dbias_buf, C_out * 256 * sizeof(float)));
    conv2d_k3_backward2(d_dout, d_x, d_weight, d_dweight_buf, d_dbias_buf, d_dx, d_dweight, d_dbias, B, C_in, C_out, H, W);

    printf("\nChecking dweight\n");
    validate_result(d_dweight, dweight, "dweight", C_out * C_in * 3 * 3, dw_acc);

    printf("\nChecking dbias\n");
    validate_result(d_dbias, dbias, "dbias", C_out, dw_acc);

    printf("\nChecking dx\n");
    validate_result(d_dx, dx, "dx", B * C_in * H * W, dw_acc);

    new_time = 0.0f;
    printf("\nManually benchmarking new dweight kernel\n");
    for (int i = 0; i < repeat_times; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        cudaEvent_t start, end;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
        cudaCheck(cudaEventRecord(start, nullptr));
        conv2d_k3_backward2(d_dout, d_x, d_weight, d_dweight_buf, d_dbias_buf, d_dx, d_dweight, d_dbias, B, C_in, C_out, H, W);
        cudaCheck(cudaEventRecord(end, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        new_time += elapsed_time;
    }
    new_time /= repeat_times;
    printf("new kernel time: %.4f ms\n", new_time);
    cudaCheck(cudaFree(d_dweight_buf));
    cudaCheck(cudaFree(d_dbias_buf));



    printf("Backward pass successful\n");

    // old benchmarking code
    //printf("All results match. Starting benchmarks.\n\n");
    //printf("Forward pass benchmarks:\n");
    //for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    //    int block_size = block_sizes[j];

    //    int repeat_times = 100;
    //    float elapsed_time = benchmark_kernel(repeat_times, conv2d_k3_forward,
    //                                          cublas_handle, kernel_num, d_x, d_weight, d_bias, d_out,
    //                                          B, C_in, C_out, H, W, block_size);

    //    float tflops = (float)B * H * W * C_in * C_out * 9 * 2 / elapsed_time * 1e3f / 1e12f;
    //    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    //}
    printf("\n\n Old benchmarks\n");
    printf("Running our own forward benchmarks\n");
    float t1 = 0.0f;
    float t2 = 0.0f;
    float t3 = 0.0f;
    float t_total = 0.0f;
    for (int i = 0; i < repeat_times; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        conv2d_k3_forward2(cublas_handle, d_x, d_weight, d_bias, d_out, B, C_in, C_out, H, W, 512, &t1, &t2, &t3);
    }
    t1 /= repeat_times;
    t2 /= repeat_times;
    t3 /= repeat_times;
    t_total = t1 + t2 + t3;

    printf("t1: %.4f ms\n", t1);
    printf("t2: %.4f ms\n", t2);
    printf("t3: %.4f ms\n", t3);
    printf("t_total: %.4f ms\n", t_total);

    //printf("\nBackward pass benchmarks:\n");
    //for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    //    int block_size = block_sizes[j];

    //    int repeat_times = 100;
    //    float elapsed_time = benchmark_kernel(repeat_times, conv2d_k3_backward1,
    //                                          cublas_handle, d_dout, d_x, d_weight, d_dx, d_dweight, d_dbias,
    //                                          B, C_in, C_out, H, W, block_size);

    //    float tflops = (float)B * H * W * C_in * C_out * (9 + 1) * 2 / elapsed_time * 1e3f / 1e12f;
    //    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    //}
    printf("\nRunning our own backward benchmarks\n");
    t1 = 0.0f;
    t2 = 0.0f;
    t3 = 0.0f;
    float t4 = 0.0f;
    float t5 = 0.0f;
    float t6 = 0.0f;
    t_total = 0.0f;
    for (int i = 0; i < repeat_times; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        conv2d_k3_backward1(cublas_handle, d_dout, d_x, d_weight, d_dx, d_dweight, d_dbias, B, C_in, C_out, H, W, 512, &t1, &t2, &t3, &t4, &t5, &t6);
    }
    cudaCheck(cudaFree(flush_buffer));

    t1 /= repeat_times;
    t2 /= repeat_times;
    t3 /= repeat_times;
    t4 /= repeat_times;
    t5 /= repeat_times;
    t6 /= repeat_times;
    t_total = t1 + t2 + t3 + t4 + t5 + t6;

    printf("t1: %.4f ms\n", t1);
    printf("t2: %.4f ms\n", t2);
    printf("t3: %.4f ms\n", t3);
    printf("t4: %.4f ms\n", t4);
    printf("t5: %.4f ms\n", t5);
    printf("t6: %.4f ms\n", t6);
    printf("t_total: %.4f ms\n", t_total);

    // free memory
    free(x);
    free(weight);
    free(bias);
    free(out);
    free(dout);
    free(dx);
    free(dweight);
    free(dbias);
    free(dout_perm);
    free(x_tiled);
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