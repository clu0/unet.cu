#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"
#include <unistd.h>
#include "groupnorm.cuh"


// -----------------------------------------------------------------------------------------------
// GPU kernels

// Essentially taken from llm.c's kernel 5
// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// we will have one block per group, which means there are B * C / group_size blocks
__global__ void groupnorm_forward_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32]; // block_size max is 1024 = 32 * 32 warps
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int block_pixels = img_size * group_size;
    // group index
    int g = blockIdx.x % n_groups;

    // move pointers
    x += blockIdx.x * img_size * group_size;
    out += blockIdx.x * img_size * group_size;
    // each block will only every acces group_size channels
    weight += g * group_size;
    bias += g * group_size;

    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;
    for (int i = threadIdx.x; i < block_pixels; i += blockDim.x) {
        float val = x[i];
        thread_sum += val;
        thread_sum2 += val * val;
    }

    // warp reduce
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    float warp_sum2 = cg:: reduce(warp, thread_sum2, cg::plus<float>{});
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    
    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{});
    block_sum /= block_pixels;
    block_sum2 /= block_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    if (threadIdx.x == 0 && mean != nullptr) {
        mean[blockIdx.x] = m;
    }
    if (threadIdx.x == 0 && rstd != nullptr) {
        rstd[blockIdx.x] = s;
    }

    for (int i = threadIdx.x; i < block_pixels; i += blockDim.x) {
        int c_mod_group = (i / img_size) % group_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c_mod_group] + bias[c_mod_group];
    }
}

// most similar to kernel 2 in llm.c
// not doing any float16 optimizations yet
// main change is to have each block allocated to a single group
__global__ void groupnorm_backward_kernel(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int img_size, int group_size, int n_groups
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32]; // block_size max is 1024 = 32 * 32 warps
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int block_pixels = img_size * group_size;
    // group index
    int g = blockIdx.x % n_groups;

    // move pointers
    dout += blockIdx.x * img_size * group_size;
    x += blockIdx.x * img_size * group_size;
    dx += blockIdx.x * img_size * group_size;
    weight += g * group_size;
    dweight += g * group_size;
    dbias += g * group_size;

    float m_val = mean[blockIdx.x];
    float rstd_val = rstd[blockIdx.x];


    // calculate the two mean terms in the group dimension
    // first is dout * weight, and second is dout * weight * norm
    // where norm = (x - mean) * rstd
    float w_dout_thread = 0.0f;
    float w_dout_norm_thread = 0.0f;
    for (int i = threadIdx.x; i < block_pixels; i += blockDim.x) {
        int c_mod_group = (i / img_size) % group_size;
        float cur_w_dout = weight[c_mod_group] * dout[i];
        w_dout_thread += cur_w_dout;
        float norm = (x[i] - m_val) * rstd_val;
        w_dout_norm_thread += cur_w_dout * norm;
    }
    // warp reduce
    float w_dout_warp = cg::reduce(warp, w_dout_thread, cg::plus<float>{});
    float w_dout_norm_warp = cg::reduce(warp, w_dout_norm_thread, cg::plus<float>{});
    // store warp sum in shared mem
    shared_sum[warp_id] = w_dout_warp;
    shared_sum2[warp_id] = w_dout_norm_warp;
    __syncthreads();

    // load warp sums from shared memory
    w_dout_warp = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    w_dout_norm_warp = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float w_dout_block = cg::reduce(warp, w_dout_warp, cg::plus<float>{});
    float w_dout_norm_block = cg::reduce(warp, w_dout_norm_warp, cg::plus<float>{});
    w_dout_block /= block_pixels;
    w_dout_norm_block /= block_pixels;

    // update dx
    for (int i = threadIdx.x; i < block_pixels; i += blockDim.x) {
        // in bounds of image
        // accumulate dw and db
        float dout_val = dout[i];
        float norm = (x[i] - m_val) * rstd_val;

        // update dx
        int c_mod_group = (i / img_size) % group_size;
        float w_dout = weight[c_mod_group] * dout_val;
        dx[i] = (w_dout - w_dout_block - norm * w_dout_norm_block) * rstd_val;
    }
    // update dw and db
    // use different methods when the image size is large or small

    // if the image size is larger than the block size
    // loop over the channels and use the whole block on each channel
    // otherwise, assign each warp to a channel
    // in either case image size must be larger than the warp size
    //assert(img_size % warp.size() == 0);
    assert(blockDim.x % warp.size() == 0);
    if (img_size % blockDim.x == 0) {
        for (int c = 0; c < group_size; c++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;
            for (int i = threadIdx.x; i < img_size; i += blockDim.x) {
                float dout_val = dout[i];
                db_thread += dout_val;
                float norm = (x[i] - m_val) * rstd_val;
                dw_thread += dout_val * norm;
            }

            // move pointers
            dout += img_size;
            x += img_size;

            // warp reduce
            float dw_warp = cg::reduce(warp, dw_thread, cg::plus<float>{});
            float db_warp = cg::reduce(warp, db_thread, cg::plus<float>{});
            ////// store warp sum in shared mem
            if (lane_id == 0) {
                shared_sum[warp_id] = dw_warp;
                shared_sum2[warp_id] = db_warp;
            }
            __syncthreads();
            // use the first thread to reduce the shared memory sums and save to global memory
            if (threadIdx.x == 0) {
                float dw_block = 0.0f;
                float db_block = 0.0f;
                for (int i = 0; i < num_warps; i++) {
                    dw_block += shared_sum[i];
                    db_block += shared_sum2[i];
                }
                atomicAdd(dweight + c, dw_block);
                atomicAdd(dbias + c, db_block);
            }
        }
    } else {
        // if group size is large, need to loop over the group channels with the whole block
        int block_reps = ceil_div(group_size, num_warps);
        for (int br = 0; br < block_reps; br++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;

            int ch = br * num_warps + warp_id;
            if (ch < group_size) {
                const float* dout_ch = dout + ch * img_size;
                const float* x_ch = x + ch * img_size;
                for (int i = lane_id; i < img_size; i += warp.size()) {
                    float dout_val = dout_ch[i];
                    db_thread += dout_val;
                    float norm = (x_ch[i] - m_val) * rstd_val;
                    dw_thread += dout_val * norm;
                }
                
                // warp reduce
                float dw_warp = cg::reduce(warp, dw_thread, cg::plus<float>{});
                float db_warp = cg::reduce(warp, db_thread, cg::plus<float>{});
                // since each warp takes care of an entire image
                // directly store result
                if (lane_id == 0) {
                    atomicAdd(dweight + ch, dw_warp);
                    atomicAdd(dbias + ch, db_warp);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------
// CUDA kernel launcher

void groupnorm_forward(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(512, img_size * group_size), 32);
    groupnorm_forward_kernel<<<n_blocks, block_size>>>(
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups
    );
    cudaCheck(cudaGetLastError());
}

void groupnorm_backward(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(512, img_size * group_size), 32 * group_size);
    groupnorm_backward_kernel<<<n_blocks, block_size>>>(
        dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, group_size, n_groups
    );
    cudaCheck(cudaGetLastError());
}


// -----------------------------------------------------------------------------------------------

#ifndef LINKING
int main(int argc, char **argv) {
    //int B = 32;
    //int C = 64;
    //int H = 32;
    //int W = 32;
    srand(0);
    int B = 16;
    int C = 128;
    int H = 4;
    int W = 8;

    int img_size = H * W;
    int n_groups = 32;


    // create host memory to load data
    float *x = (float*)malloc(B * C * img_size * sizeof(float));
    float *weight = (float*)malloc(C * sizeof(float));
    float *bias = (float*)malloc(C * sizeof(float));
    float *out = (float*)malloc(B * C * img_size * sizeof(float));
    float *dout = (float*)malloc(B * C * img_size * sizeof(float));
    float *dx = (float*)malloc(B * C * img_size * sizeof(float));
    float *dweight = (float*)malloc(C * sizeof(float));
    float *dbias = (float*)malloc(C * sizeof(float));


    // read saved output
    FILE *file = fopenCheck("groupnorm.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), B * C * img_size, file);
    freadCheck(weight, sizeof(float), C, file);
    freadCheck(bias, sizeof(float), C, file);
    freadCheck(out, sizeof(float), B * C * img_size, file);
    freadCheck(dout, sizeof(float), B * C * img_size, file);
    freadCheck(dx, sizeof(float), B * C * img_size, file);
    freadCheck(dweight, sizeof(float), C, file);
    freadCheck(dbias, sizeof(float), C, file);
    fcloseCheck(file);


    // device memory
    float *d_x, *d_weight, *d_bias, *d_out, *d_mean, *d_rstd, *d_dout, *d_dx, *d_dweight, *d_dbias;
    cudaCheck(cudaMalloc(&d_x, B * C * img_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * C * img_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * n_groups * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * n_groups * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * C * img_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, B * C * img_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, B * C * img_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * C * img_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_dx, 0, B * C * img_size * sizeof(float)));
    cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(float)));
    cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(float)));

    printf("Checking forward pass\n");
    float fwd_acc = 1e-1;
    groupnorm_forward(d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(d_out, out, "out", B * C * img_size, fwd_acc);
    printf("Forward pass successful\n");

    // check the values of rstd
    float *rstd = (float*)malloc(B * n_groups * sizeof(float));

    printf("Checking backward pass\n");
    groupnorm_backward(d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias, B, C, H, W, n_groups);
    float acc = 1e-3;
    printf("Checking dbias\n");
    validate_result(d_dbias, dbias, "dbias", C, acc);
    printf("Checking dweight\n");
    validate_result(d_dweight, dweight, "dweight", C, acc);
    printf("Checking dx\n");
    validate_result(d_dx, dx, "dx", B * C * img_size, 1.0f);
    printf("Backward pass successful\n");

    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks\n");
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, groupnorm_forward,
                                            d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                            B, C, H, W, n_groups);
    printf("time %.4f ms\n", elapsed_time);

    printf("Backward pass benchmarks\n");
    elapsed_time = benchmark_kernel(repeat_times, groupnorm_backward,
                                    d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                    B, C, H, W, n_groups);
    printf("time %.4f ms\n", elapsed_time);
}
#endif