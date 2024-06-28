#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"


// this will only be called with one block
__global__ void mse_forward_kernel(
    const float* inp, const float* y, float* loss, int N
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float shared_sum[32]; // max 32 warps
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float diff = inp[i] - y[i];
        thread_sum += diff * diff;
    }

    // warp reduce
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    shared_sum[warp_id] = warp_sum;
    __syncthreads();
    
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});

    if (threadIdx.x == 0) {
        loss[0] = block_sum / N;
    }
}

void mse_forward(const float* inp, const float* y, float* loss, int N, int block_size) {
    mse_forward_kernel<<<1, block_size>>>(inp, y, loss, N);
}



// not calculating dy in backward
// note that backward for MSE doesn't require the loss value
__global__ void mse_backward_kernel(
    const float* inp, const float* y, float* dinp, int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float x = inp[idx];
        dinp[idx] = 2.0f * (x - y[idx]) / N;
    }
}

void mse_backward(const float* inp, const float* y, float* dinp, int N, int block_size) {
    int n_blocks = ceil_div(N, block_size);
    mse_backward_kernel<<<n_blocks, block_size>>>(inp, y, dinp, N);
}

#ifndef LINKING
int main(int argc, char **argv) {
    int N = 32 * 3 * 64 * 64;
    int block_size = 512;

    // host memory
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* dx = (float*)malloc(N * sizeof(float));
    float* loss = (float*)malloc(sizeof(float));

    FILE *file = fopenCheck("mse.bin", "rb");
    if (!file) {
        return -1;
    }
    freadCheck(x, sizeof(float), N, file);
    freadCheck(y, sizeof(float), N, file);
    freadCheck(dx, sizeof(float), N, file);
    freadCheck(loss, sizeof(float), 1, file);
    fcloseCheck(file);

    // device memory
    float *d_x, *d_y, *d_dx, *d_loss;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_loss, sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("Checking forward pass\n");
    mse_forward(d_x, d_y, d_loss, N, block_size);
    validate_result(d_loss, loss, "loss", 1);
    printf("Forward pass successful\n");

    printf("Checking backward pass\n");
    mse_backward(d_x, d_y, d_dx, N, block_size);
    validate_result(d_dx, dx, "dinp", N);
    printf("Backward pass successful\n");
}
#endif