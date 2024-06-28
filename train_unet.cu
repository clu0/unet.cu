#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.cuh"
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <string.h>
#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mode_t

namespace cg = cooperative_groups;

// ------------------------------------------------------------------------------------------------
// Linear layer
typedef struct {
    float* w; // OC, C
    float* b; // OC
} LinearParams;

typedef struct {
    float* inp; // N, C
    float* out; // N, OC
} LinearActs;


void linear_set_param_ptrs(
    LinearParams* params,
    float* params_memory,
    int C, int OC
) {
    params->w = params_memory;
    params_memory += OC * C;
    params->b = params_memory;
}

inline size_t linear_count_params(int C, int OC) {
    return OC * C + OC;
}
// ---------------------------------------------
// kernels

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

// ------------------------------------------------------------------------------------------------
// Broadcast tensor layer

// kernel to broadcast a tensor from shape (B, C) to (B, C, H, W)
// Assuming one thread per pixel
__global__ void broadcast_last_dims_forward_kernel(
    const float* x,
    float* out,
    int N, int H, int W, int n_bk_per_img
) {
    int n = blockIdx.x / n_bk_per_img;
    __shared__ float x_sh[1];
    if (threadIdx.x == 0) {
        x_sh[0] = x[n];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * H * W) {
        out[idx] = x_sh[0];
    }
}

void broadcast_last_dims_forward(
    const float* x,
    float* out,
    int N, int H, int W, int block_size
) {
    block_size = min_int(block_size, H * W);
    int n_bk_per_img = ceil_div(H * W, block_size);
    broadcast_last_dims_forward_kernel<<<N * n_bk_per_img, block_size>>>(x, out, N, H, W, n_bk_per_img);
};

// backwards for broadcast
// use one warp per image
__global__ void broadcast_last_dims_backward_kernel(
    const float* dout,
    float* dx,
    int N, int H, int W
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) { return; }

    int img_size = H * W;
    dout += idx * img_size;
    dx += idx;

    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < img_size; i += warp.size()) {
        sum += dout[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    if (warp.thread_rank() == 0) {
        dx[0] = sum;
    }
}

void broadcast_last_dims_backward(
    const float* dout,
    float* dx,
    int N, int H, int W, int block_size
) {
    block_size = min_int(block_size, H * W);
    assert(block_size % 32 == 0);
    int warps_per_block = block_size / 32;
    int gridDim = ceil_div(N, warps_per_block);
    broadcast_last_dims_backward_kernel<<<gridDim, block_size>>>(dout, dx, N, H, W);
};

// ------------------------------------------------------------------------------------------------
// Add layer
__global__ void add_forward_kernel(
    const float* a, const float* b,
    float* out,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

void add_forward(
    const float* a, const float* b,
    float* out,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    add_forward_kernel<<<n_blk, block_size>>>(a, b, out, N);
}

// add A and B in place, and store result in B
__global__ void add_inplace_forward_kernel(
    const float* a, float* b,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        b[idx] += a[idx];
    }
}

void add_inplace_forward(
    const float* a, float* b,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    add_inplace_forward_kernel<<<n_blk, block_size>>>(a, b, N);
}

// ------------------------------------------------------------------------------------------------
// SiLU layer

typedef struct {
    float* x;
    float* out;
} SiluActs;

// ----------------------------------------------
// kernels

__global__ void silu_forward_kernel(
    const float* x,
    float* out,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    float x_val = x[idx];
    out[idx] = x_val / (1.0f + expf(-x_val));
}

void silu_forward(
    const float* x,
    float* out,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    silu_forward_kernel<<<n_blk, block_size>>>(x, out, N);
}

__global__ void silu_backward_kernel(
    const float* dout, const float* x,
    float* dx,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    float out_val = dout[idx];
    float x_val = x[idx];
    float expx = expf(-x_val);
    float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
    dx[idx] = out_val * grad_silu;
}

void silu_backward(
    const float* dout, const float* x,
    float* dx,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    silu_backward_kernel<<<n_blk, block_size>>>(dout, x, dx, N);
}

// ------------------------------------------------------------------------------------------------
// Upsample
typedef struct {
    float* x;
    float* out;
} UpsampleActs;

// one thread per pixel
__global__ void upsample_forward_kernel(
    const float* x,
    float* out,
    int B, int C, int H, int W
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_size = H * W;
    if (flat_idx >= B * C * img_size) {
        return;
    }

    int H_out = H*2;
    int W_out = W*2;
    int img_out_size = H_out * W_out;

    int b = flat_idx / (C * img_size);
    int c = (flat_idx / img_size) % C;
    int i = (flat_idx / W) % H;
    int j = flat_idx % W;

    // move pointers
    x += b * C * img_size + c * img_size + i * W + j;
    out += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float x_val = x[0];

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            out[ii * W_out + jj] = x_val;
        }
    }
}

void upsample_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size
) {
    int gridDim = ceil_div(B * C * H * W, block_size);
    upsample_forward_kernel<<<gridDim, block_size>>>(x, out, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

// number of threads equals the number of pixels
// in downsampled image
__global__ void upsample_backward_kernel(
    const float* dout,
    float* dx,
    int B, int C, int H, int W
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_in_size = H * W;
    if (flat_idx >= B * C * img_in_size) {
        return;
    }

    int H_out = H*2;
    int W_out = W*2;
    int img_out_size = H_out * W_out;

    int b = flat_idx / (C * img_in_size);
    int c = (flat_idx / img_in_size) % C;
    int i = (flat_idx / W) % H;
    int j = flat_idx % W;

    // move pointers
    dx += b * C * img_in_size + c * img_in_size + i * W + j;
    dout += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float dout_sum = 0.0f;
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            dout_sum += dout[ii * W_out + jj];
        }
    }
    dx[0] = dout_sum;
}

void upsample_backward1(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size
) {
    int gridDim = ceil_div(B * C * H * W, block_size);
    upsample_backward_kernel<<<gridDim, block_size>>>(dout, dx, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// Avgpool layer

typedef struct {
    float* x;
    float* out;
} AvgpoolActs;

__global__ void avgpool_2d_forward_kernel(
    const float* x,  // (B, C, H, W)
    float* out, // (B, C, H/2, W/2)
    int B, int C, int H, int W
) {
    //printf("started kernel\n");
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int H_2 = H / 2;
    int W_2 = W / 2;
    int img_out_size = H_2 * W_2;
    if (flat_idx >= B * C * img_out_size) {
        return;
    }
    int img_size = H * W;
    int b = flat_idx / (C * img_out_size);
    int c = (flat_idx / img_out_size) % C;
    int i = (flat_idx / W_2) % H_2;
    int j = flat_idx % W_2;

    // move pointers
    x += b * C * img_size + c * img_size + 2 * i * W + 2 * j;
    out += b * C * img_out_size + c * img_out_size + i * W_2 + j;

    float sum = 0.0f;
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            sum += x[ii * W + jj];
        }
    }
    out[0] = sum / 4.0f;
}

void avgpool_2d_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    const int block_size
) {
    int gridDim = ceil_div(B * C * (H / 2) * (W / 2), block_size);
    avgpool_2d_forward_kernel<<<gridDim, block_size>>>(x, out, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

__global__ void avgpool_2d_backward_kernel(
    const float* dout,
    float* dx,
    int B, int C, int H_out, int W_out // H_out and W_out are the shape of the output from forward, i.e. half the original sizes
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_out_size = H_out * W_out;
    if (flat_idx >= B * C * img_out_size) {
        return;
    }
    int H_in = H_out * 2;
    int W_in = W_out * 2;
    int img_in_size = H_in * W_in;
    int b = flat_idx / (C * img_out_size);
    int c = (flat_idx / img_out_size) % C;
    int i = (flat_idx / W_out) % H_out;
    int j = flat_idx % W_out;

    // move pointers
    dout += b * C * img_out_size + c * img_out_size + i * W_out + j;
    dx += b * C * img_in_size + c * img_in_size + 2 * i * W_in + 2 * j;

    float d = dout[0] / 4.0f;

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            dx[ii * W_in + jj] = d;
        }
    }
}

void avgpool_2d_backward1(
    const float* dout,
    float* dx,
    int B, int C, int H, int W,
    const int block_size
) {
    int H_out = H / 2;
    int W_out = W / 2;
    int gridDim = ceil_div(B * C * H_out * W_out, block_size);
    avgpool_2d_backward_kernel<<<gridDim, block_size>>>(dout, dx, B, C, H_out, W_out);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// Concat channel layer
// used to concat unet skip connections in the channel dimension
typedef struct {
    float* x1;
    float* x2;
    float* out;
} ConcatChannelActs;

__global__ void concat_channel_forward_kernel(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        // copy input from x1
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;
        out[out_idx] = x1[idx];
    }
    if (idx < B * C2 * H * W) {
        // copy input from x2
        // move over from x1
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        out[out_idx] = x2[idx];
    }
}

void concat_channel_forward(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W, int block_size
) {
    int N = B * max_int(C1, C2) * H * W;
    int n_blk = ceil_div(N, block_size);
    concat_channel_forward_kernel<<<n_blk, block_size>>>(x1, x2, out, B, C1, C2, H, W);
}

__global__ void concat_channel_backward_kernel(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;

        dx1[idx] = dout[out_idx];
    }
    if (idx < B * C2 * H * W) {
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        dx2[idx] = dout[out_idx];
    }
}

void concat_channel_backward(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W, int block_size
) {
    int N = B * max_int(C1, C2) * H * W;
    int n_blk = ceil_div(N, block_size);
    concat_channel_backward_kernel<<<n_blk, block_size>>>(dout, dx1, dx2, B, C1, C2, H, W);
}

// ------------------------------------------------------------------------------------------------
// Conv2d 1x1 layer

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

// ------------------------------------------------------------------------------------------------
// Conv2d 3x3 layer
typedef struct {
    float* w;
    float* b;
} ConvK3Params;

inline void convk3_set_param_ptrs(
    ConvK3Params* params,
    float* params_memory,
    int C_in, int C_out
) {
    params->w = params_memory;
    params->b = params->w + C_in * C_out * 9;
}

inline size_t convk3_count_params(int C_in, int C_out) {
    return C_in * C_out * 9 + C_out;
}

typedef struct {
    float* inp;
    float* out;
} ConvK3Acts;

// ---------------------------------------------
// kernels

// this kernel will roughly follow kernel 5 here: https://github.com/siboehm/SGEMM_CUDA, with a few differences
// the main operation is convolving the input X with the weights
// weights are shaped (O, Cx9), and X is shaped (B, C, HxW)
// We are morally doing a matmul with dimensions O, C, HxW
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

__global__ void dbias_reduce_kernel1(
    float* dbias_buf, float* dbias,
    const int B
) {
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

// ------------------------------------------------------------------------------------------------
// GroupNorm

typedef struct {
    float* w;
    float* b;
} GroupNormParams;

inline void gn_set_param_ptrs(
    GroupNormParams* params,
    float* params_memory,
    int C
) {
    params->w = params_memory;
    params->b = params->w + C;
}

typedef struct {
    float* x;
    float* out;
    float* mean;
    float* rstd;
} GroupNormActs;

inline void gn_set_act_ptrs(
    GroupNormActs* acts,
    float* acts_memory,
    int B, int C, int H, int W, int n_groups
) {
    acts->out = acts_memory;
    acts->mean = acts->out + B * C * H * W;
    acts->rstd = acts->mean + B * n_groups;
}

typedef struct {
    float* dx;
    float* dout;
} GroupNormBackActs;

// ---------------------------------------------
// kernels

// Essentially taken from llm.c's kernel 5
// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// we will have one block per group, which means there are B * C / group_size blocks
__global__ void groupnorm_forward_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
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

// most similar to kernel 2 in llm.c
// not doing any float16 optimizations yet
// main change is to have each block allocated to a single group
__global__ void groupnorm_backward_kernel(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int img_size, int group_size, int n_groups
) {
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
// ------------------------------------------------------------------------------------------------
// ResBlock

#define NUM_RES_PARAM_TENSORS 12
typedef struct {
    float* gn1_w; // C
    float* gn1_b; // C
    float* cv3_1_w; // C_out, C, 3, 3
    float* cv3_1_b; // C_out
    float* l_emb_w; // C_out, C_emb
    float* l_emb_b; // C_out
    float* gn2_w; // C_out
    float* gn2_b; // C_out
    float* cv3_2_w; // C_out, C_out, 3, 3
    float* cv3_2_b; // C_out
    float* res_cv1_w; // C_out, C
    float* res_cv1_b; // C_out
    size_t param_sizes[NUM_RES_PARAM_TENSORS];
    size_t n_params;
} ResBlockParameters;

#define NUM_RES_ACT_TENSORS 18 // not counting input and emb, which will come from outs of previous layers
typedef struct {
    float* gn1; // B, C, H, W
    float* gn1_mean; // B, n_groups
    float* gn1_rstd; // B, n_groups
    float* silu1; // B, C, H, W
    float* ud_h; // B, C, H_out, W_out
    float* ud_x; // B, C, H_out, W_out
    float* cv3_1; // B, C_out, H_out, W_out
    float* silu_emb; // B, C_emb
    float* l_emb; // B, C_out
    float* broad_emb; // B, C_out, H_out, W_out
    float* add1; // B, C_out, H_out, W_out
    float* gn2; // B, C_out, H_out, W_out
    float* gn2_mean; // B, n_groups
    float* gn2_rstd; // B, n_groups
    float* silu2; // B, C_out, H_out, W_out
    float* cv3_2; // B, C_out, H_out, W_out
    float* res_cv1; // B, C_out, H_out, W_out
    float* add2; // B, C_out, H_out, W_out
    size_t act_sizes[NUM_RES_ACT_TENSORS];
    size_t n_acts;
    float* input; // B, C, H, W
    float* emb; // B, C_emb
} ResBlockActivations;


// the last two backward tensors, dx and demb
// we don't need to allocate extra space for these
// instead we will just overwrite forward pass activations acts->input and acts->emb
#define NUM_RES_BACKWARD_TENSORS 7
typedef struct {
    float* buf_BCemb;
    float* buf_BCHoWo;
    float* buf1_BCHW;
    float* buf2_BCHW;
    float* dout; // B, C_out, H_out, W_out
    float* dweight_buf;
    float* dbias_buf;
    float* dx; // B, C, H, W
    float* demb; // B, C_emb
    size_t back_sizes[NUM_RES_BACKWARD_TENSORS];
    size_t n_backs;
} ResBlockBackwardActivations;

void resblock_count_params(
    ResBlockParameters* params,
    int C, int C_emb, int C_out, int B, int H, int W, int up, int down, int gn_n_groups
) {
    int H_out = H;
    int W_out = W;
    if (up) {
        H_out *= 2;
        W_out *= 2;
    } else if (down) {
        H_out /= 2;
        W_out /= 2;
    }
    // fill and count param sizes
    size_t* param_sizes = params->param_sizes;
    param_sizes[0] = C; // gn1_w
    param_sizes[1] = C; // gn1_b
    param_sizes[2] = C_out * C * 9; // cv3_1_w
    param_sizes[3] = C_out; // cv3_1_b
    param_sizes[4] = C_emb * C_out; // l_emb_w
    param_sizes[5] = C_out; // l_emb_b
    param_sizes[6] = C_out; // gn2_w
    param_sizes[7] = C_out; // gn2_b
    param_sizes[8] = C_out * C_out * 9; // cv3_2_w
    param_sizes[9] = C_out; // cv3_2_b
    param_sizes[10] = C_out * C; // res_cv1_w
    param_sizes[11] = C_out; // res_cv1_b
    if (C_out != C) {
        param_sizes[10] = C_out * C; // res_cv1_w
        param_sizes[11] = C_out; // res_cv1_b
    } else {
        param_sizes[10] = 0;
        param_sizes[11] = 0;
    }

    // count number of params
    size_t num_parameters = 0;
    for  (size_t i = 0; i < NUM_RES_PARAM_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    params->n_params = num_parameters;
}

void resblock_count_acts(
    ResBlockActivations* acts,
    int C, int C_emb, int C_out, int B, int H, int W, int up, int down, int gn_n_groups
) {
    int H_out = H;
    int W_out = W;
    if (up) {
        H_out *= 2;
        W_out *= 2;
    } else if (down) {
        H_out /= 2;
        W_out /= 2;
    }
    // fill and count activation sizes
    size_t* acts_sizes = acts->act_sizes;
    acts_sizes[0] = B * C * H * W; // gn1
    acts_sizes[1] = B * gn_n_groups; // gn1_mean
    acts_sizes[2] = B * gn_n_groups; // gn1_rstd
    acts_sizes[3] = B * C * H * W; // silu1
    acts_sizes[4] = B * C * H_out * W_out; // ud_h
    acts_sizes[5] = B * C * H_out * W_out; // ud_x
    acts_sizes[6] = B * C_out * H_out * W_out; // cv3_1
    acts_sizes[7] = B * C_emb; // silu_emb
    acts_sizes[8] = B * C_out; // l_emb
    acts_sizes[9] = B * C_out * H_out * W_out; // broad_emb
    acts_sizes[10] = B * C_out * H_out * W_out; // add1
    acts_sizes[11] = B * C_out * H_out * W_out; // gn2
    acts_sizes[12] = B * gn_n_groups; // gn2_mean
    acts_sizes[13] = B * gn_n_groups; // gn2_rstd
    acts_sizes[14] = B * C_out * H_out * W_out; // silu2
    acts_sizes[15] = B * C_out * H_out * W_out; // cv3_2
    acts_sizes[16] = (C_out == C) ? 0 : B * C_out * H_out * W_out; // res_cv1
    acts_sizes[17] = B * C_out * H_out * W_out; // add2

    size_t num_acts_params = 0;
    for (size_t i = 0; i < NUM_RES_ACT_TENSORS; i++) {
        num_acts_params += acts_sizes[i];
    }
    acts->n_acts = num_acts_params;
}

void set_resblock_params_ptrs(
    int C,
    int C_out,
    ResBlockParameters *params,
    float* params_memory_gpu
) {
    float** ptrs[] = {
        &params->gn1_w, &params->gn1_b, &params->cv3_1_w, &params->cv3_1_b,
        &params->l_emb_w, &params->l_emb_b, &params->gn2_w, &params->gn2_b,
        &params->cv3_2_w, &params->cv3_2_b, &params->res_cv1_w, &params->res_cv1_b
    };
    int num_param_tensors = NUM_RES_PARAM_TENSORS;
    if (C_out == C) {
        num_param_tensors -= 2;
    }
    float* params_ptr = params_memory_gpu;
    for (int i = 0; i < num_param_tensors; i++) {
        *(ptrs[i]) = params_ptr;
        params_ptr += params->param_sizes[i];
    }
    if (C_out == C) {
        params->res_cv1_w = nullptr;
        params->res_cv1_b = nullptr;
    }
}

void set_resblock_acts_ptrs(
    int C,
    int C_out,
    ResBlockActivations *acts,
    float* acts_memory
) {
    float** acts_ptrs[] = {
        &acts->gn1, &acts->gn1_mean, &acts->gn1_rstd, &acts->silu1,
        &acts->ud_h, &acts->ud_x, &acts->cv3_1, &acts->silu_emb, &acts->l_emb, &acts->broad_emb,
        &acts->add1, &acts->gn2, &acts->gn2_mean, &acts->gn2_rstd, &acts->silu2, &acts->cv3_2,
        &acts->res_cv1, &acts->add2
    };
    float* acts_ptr = acts_memory;
    for (int i = 0; i < NUM_RES_ACT_TENSORS - 2; i++) {
        *(acts_ptrs[i]) = acts_ptr;
        acts_ptr += acts->act_sizes[i];
    }

    if (C_out == C) {
        acts->res_cv1 = nullptr;
        acts->add2 = acts_ptr;
    } else {
        acts->res_cv1 = acts_ptr;
        acts_ptr += acts->act_sizes[NUM_RES_ACT_TENSORS - 2];
        acts->add2 = acts_ptr;
    }
}

// assume all memory is allocated, run the forward pass
void resblock_forward(
    cublasHandle_t cublas_handle,
    int C,
    int C_emb,
    int C_out,
    int B,
    int H,
    int W,
    int block_size,
    int up,
    int down,
    int gn_n_groups,
    ResBlockParameters *params,
    ResBlockActivations *acts
) {
    int H_out = H;
    int W_out = W;
    if (up) {
        H_out *= 2;
        W_out *= 2;
    } else if (down) {
        H_out /= 2;
        W_out /= 2;
    }
    // up to first 3x3 conv in BigGAN res block
    // note: residual stream needs a convolution at the end if C != C_out
    groupnorm_forward(acts->input, params->gn1_w, params->gn1_b, acts->gn1, acts->gn1_mean, acts->gn1_rstd,
        B, C, H, W, gn_n_groups);
    cudaCheck(cudaGetLastError());
    silu_forward(acts->gn1, acts->silu1, B * C * H * W, block_size);
    cudaCheck(cudaGetLastError());
    if (up) {
        upsample_forward1(acts->ud_h, acts->silu1, B, C, H, W, block_size);
        upsample_forward1(acts->ud_x, acts->input, B, C, H, W, block_size);
    } else if (down) {
        avgpool_2d_forward1(acts->ud_h, acts->silu1, B, C, H, W, block_size);
        avgpool_2d_forward1(acts->ud_x, acts->input, B, C, H, W, block_size);
    } else {
        cudaCheck(cudaMemcpy(acts->ud_h, acts->silu1, B * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(acts->ud_x, acts->input, B * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    cudaCheck(cudaGetLastError());
    conv2d_k3_forward3(acts->ud_h, params->cv3_1_w, params->cv3_1_b, acts->cv3_1, B, C, C_out, H_out, W_out);
    cudaCheck(cudaGetLastError());

    // project and add embedding layer
    silu_forward(acts->emb, acts->silu_emb, B * C_emb, block_size);
    cudaCheck(cudaGetLastError());
    matmul_forward2(cublas_handle, acts->l_emb, acts->silu_emb, params->l_emb_w, params->l_emb_b, B, C_emb, C_out, block_size);
    cudaCheck(cudaGetLastError());
    broadcast_last_dims_forward(acts->l_emb, acts->broad_emb, B * C_out, H_out, W_out, block_size);
    cudaCheck(cudaGetLastError());
    add_forward(acts->cv3_1, acts->broad_emb, acts->add1, B * C_out * H_out * W_out, block_size);
    cudaCheck(cudaGetLastError());

    // second group of norm -> silu -> 3x3 conv
    groupnorm_forward(acts->add1, params->gn2_w, params->gn2_b, acts->gn2, acts->gn2_mean, acts->gn2_rstd,
        B, C_out, H_out, W_out, gn_n_groups);
    cudaCheck(cudaGetLastError());
    silu_forward(acts->gn2, acts->silu2, B * C_out * H_out * W_out, block_size);
    cudaCheck(cudaGetLastError());
    conv2d_k3_forward3(acts->silu2, params->cv3_2_w, params->cv3_2_b, acts->cv3_2, B, C_out, C_out, H_out, W_out);
    cudaCheck(cudaGetLastError());

    // add residual (change channels with 1x1 conv if neccessary)
    if (C_out == C) {
        add_forward(acts->cv3_2, acts->ud_x, acts->add2, B * C_out * H_out * W_out, block_size);
        cudaCheck(cudaGetLastError());
    } else {
        conv2d_k1_forward2(acts->ud_x, params->res_cv1_w, params->res_cv1_b, acts->res_cv1, B, C, H_out, W_out, C_out);
        cudaCheck(cudaGetLastError());
        add_forward(acts->cv3_2, acts->res_cv1, acts->add2, B * C_out * H_out * W_out, block_size);
        cudaCheck(cudaGetLastError());
    }
}

void resblock_backward(
    cublasHandle_t cublas_handle,
    int C,
    int C_emb,
    int C_out,
    int B,
    int H,
    int W,
    int block_size,
    int up,
    int down,
    int gn_n_groups,
    ResBlockParameters *params,
    ResBlockParameters *grads,
    ResBlockActivations *acts,
    ResBlockBackwardActivations *back_acts
) {
    int H_out = H;
    int W_out = W;
    if (up) {
        H_out *= 2;
        W_out *= 2;
    } else if (down) {
        H_out /= 2;
        W_out /= 2;
    }
    // Note: we're using the activation of the final conv layer as a buffer to store
    // backward pass gradients, because we don't need it in the backward pass anymore.
    float *demb = back_acts->demb;
    float *dx = back_acts->dx;
    float* dout = back_acts->dout;
    float *res_dx = back_acts->buf_BCHoWo;
    float* buf1_BCoHoWo = acts->cv3_2;
    float* buf2_BCoHoWo = acts->add2;
    cudaCheck(cudaMemset(buf1_BCoHoWo, 0, B * C_out * H_out * W_out * sizeof(float)));
    cudaCheck(cudaMemset(buf2_BCoHoWo, 0, B * C_out * H_out * W_out * sizeof(float)));
    // Note: we don't need to run a separate kernel to backprop through an add operation
    // we can just share the dout buffer between the two inputs
    if (C_out != C) {
        conv2d_k1_backward1(cublas_handle, dout, acts->ud_x, params->res_cv1_w, res_dx, grads->res_cv1_w, grads->res_cv1_b, B, C, C_out, H_out, W_out, block_size);
        cudaCheck(cudaGetLastError());
    } else {
        // point res_dx to dout
        res_dx = dout;
    }
    // NOTE: after backprop through cv1 above, we can use ud_x as a (B, C, H_out, W_out) buffer
    float* buf1_BCHoWo = acts->ud_x;

    // second group of norm -> silu -> 3x3 conv
    conv2d_k3_backward2(dout, acts->silu2, params->cv3_2_w, back_acts->dweight_buf, back_acts->dbias_buf, buf1_BCoHoWo, grads->cv3_2_w, grads->cv3_2_b, B, C_out, C_out, H_out, W_out);
    cudaCheck(cudaGetLastError());
    silu_backward(buf1_BCoHoWo, acts->gn2, buf2_BCoHoWo, B * C_out * H_out * W_out, block_size);
    cudaCheck(cudaGetLastError());
    groupnorm_backward(buf2_BCoHoWo, acts->add1, acts->gn2_mean, acts->gn2_rstd, params->gn2_w, buf1_BCoHoWo, grads->gn2_w, grads->gn2_b, B, C_out, H_out, W_out, gn_n_groups);
    cudaCheck(cudaGetLastError());

    // emb layer
    // note at this point we can also use the last layer activations as buffers
    broadcast_last_dims_backward(buf1_BCoHoWo, acts->l_emb, B * C_out, H_out, W_out, block_size);
    cudaCheck(cudaGetLastError());
    matmul_backward1(cublas_handle, back_acts->buf_BCemb, grads->l_emb_w, grads->l_emb_b, acts->l_emb, acts->silu_emb, params->l_emb_w, B, C_emb, C_out);
    cudaCheck(cudaGetLastError());
    silu_backward(back_acts->buf_BCemb, acts->emb, demb, B * C_emb, block_size);
    cudaCheck(cudaGetLastError());

    // first layers
    conv2d_k3_backward2(buf1_BCoHoWo, acts->ud_h, params->cv3_1_w, back_acts->dweight_buf, back_acts->dbias_buf, buf1_BCHoWo, grads->cv3_1_w, grads->cv3_1_b, B, C, C_out, H_out, W_out);
    cudaCheck(cudaGetLastError());
    // use acts->ud_x as a buffer for dx
    float* buf1_BCHW = back_acts->buf1_BCHW;
    float* buf2_BCHW = back_acts->buf2_BCHW;
    if (up) {
        upsample_backward1(buf1_BCHW, buf1_BCHoWo, B, C, H, W, block_size);
        cudaCheck(cudaGetLastError());
        upsample_backward1(buf2_BCHW, res_dx, B, C, H, W, block_size);
        cudaCheck(cudaGetLastError());
    } else if (down) {
        avgpool_2d_backward1(buf1_BCHoWo, buf1_BCHW, B, C, H, W, block_size);
        cudaCheck(cudaGetLastError());
        avgpool_2d_backward1(res_dx, buf2_BCHW, B, C, H, W, block_size);
        cudaCheck(cudaGetLastError());
    } else {
        cudaCheck(cudaMemcpy(buf1_BCHW, buf1_BCHoWo, B * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(buf2_BCHW, res_dx, B * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaGetLastError());
    }
    // use silu as the dx buffer
    silu_backward(buf1_BCHW, acts->gn1, acts->silu1, B * C * H * W, block_size);
    cudaCheck(cudaGetLastError());
    groupnorm_backward(acts->silu1, acts->input, acts->gn1_mean, acts->gn1_rstd, params->gn1_w, buf1_BCHW, grads->gn1_w, grads->gn1_b, B, C, H, W, gn_n_groups);
    cudaCheck(cudaGetLastError());

    add_forward(buf1_BCHW, buf2_BCHW, dx, B * C * H * W, block_size);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// self attention block
// assumes you have Q, K, V
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // we don't want causal attention, so we don't set any values to infinity
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        inp[idx] *= scale;
    }
}

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        // subtract max for numerical stability
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

// a simple attention forward function taken directly from llm.c kernel 3
void attention_forward1
(
    cublasHandle_t cublas_handle,
    float* out, float* qkvr, float* preatt, float* att,
    float* inp,
    int B, int T, int C, int NH,
    const int block_size
) {
    // note: resure inp as a scratch buffer because it is not used in backward
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    float* vaccum = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

// slight edit from the softmax_autoregressive_backward_kernel from train_gpt2_fp32
// our attention is no longer autoregressive
__global__ void softmax_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0.0f;
        for (int t2 = block.thread_rank(); t2 < T; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward
(
    cublasHandle_t cublas_handle,
    float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
    const float* dout,
    const float* qkvr, const float* att,
    int B, int T, int C, int NH
) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// Attention layer, including groupnorm, qkv projection, and residual stream

#define NUM_ATT_PARAM_TENSORS 6
typedef struct {
    float* gn_w; // C
    float* gn_b; // C
    float* qkv_w; // 3xC, C
    float* qkv_b; // 3xC
    float* proj_w; // C, C
    float* proj_b; // C
    size_t param_sizes[NUM_ATT_PARAM_TENSORS];
    size_t n_params;
} AttentionParams;

#define NUM_ATT_ACT_TENSORS 12 // not counting input, which will reuse output memory from previous layer
typedef struct {
    float* gn; // B, C, H, W
    float* gn_mean; // B, n_groups
    float* gn_rstd; // B, n_groups
    float* perm1; // B, HxW, C
    float* qkv1; // B, HxW, 3xC
    float* qkv2; // B, HxW, 3xC
    float* preatt; // B, NH, T, T
    float* att; // B, NH, T, T
    float* att_out; // B, HxW, C
    float* proj; // B, HxW, C
    float* perm2; // B, C, H, W
    float* add; // B, C, H, W
    size_t act_sizes[NUM_ATT_ACT_TENSORS];
    size_t n_acts;
    float* input; // B, C, H, W
} AttentionActs;

#define NUM_ATT_BACKWARD_ACTS_TENSORS 7  // dinp will use input memory
typedef struct {
    float* buf1_BCHW; // B, C, H, W
    float* buf2_BCHW; // B, C, H, W
    float* buf_B3CHW; // B, 3xC, H, W
    float* dqkvr; // B, HxW, 3xC
    float* dpreatt; // B, NH, T, T
    float* datt; // B, NH, T, T
    float* dout; // B, C, H, W
    float* dinp; // B, C, H, W
    size_t back_sizes[NUM_ATT_BACKWARD_ACTS_TENSORS];
    size_t n_backs;
} AttentionBackwardActs;

void attention_block_count_params(AttentionParams* params, int C) {
    size_t *param_sizes = params->param_sizes;
    param_sizes[0] = C; // gn_w
    param_sizes[1] = C; // gn_b
    param_sizes[2] = 3 * C * C; // qkv_w
    param_sizes[3] = 3 * C; // qkv_b
    param_sizes[4] = C * C; // proj_w
    param_sizes[5] = C; // proj_b

    size_t num_parameters = 0;
    for (int i = 0; i < NUM_ATT_PARAM_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    params->n_params = num_parameters;
}
void attention_block_count_acts(
    AttentionActs* acts,
    int B, int C, int H, int W, int gn_n_groups, int HS
) {
    int C3 = 3 * C;
    int T = H * W;
    int NH = C / HS;
    size_t *act_sizes = acts->act_sizes;
    act_sizes[0] = B * C * H * W; // gn
    act_sizes[1] = B * gn_n_groups; // gn_mean
    act_sizes[2] = B * gn_n_groups; // gn_rstd
    act_sizes[3] = B * T * C; // perm1
    act_sizes[4] = B * T * C3; // qkv1
    act_sizes[5] = B * T * C3; // qkv2
    act_sizes[6] = B * NH * T * T; // preatt
    act_sizes[7] = B * NH * T * T; // att
    act_sizes[8] = B * T * C; // att_out
    act_sizes[9] = B * T * C; // proj
    act_sizes[10] = B * C * H * W; // perm2
    act_sizes[11] = B * C * H * W; // add

    size_t num_acts_params = 0;
    for (int i = 0; i < NUM_ATT_ACT_TENSORS; i++) {
        num_acts_params += act_sizes[i];
    }
    acts->n_acts = num_acts_params;
}

void set_attention_params_pointers(
    AttentionParams* params,
    float* params_memory
) {
    float** ptrs[] = {
        &params->gn_w,
        &params->gn_b,
        &params->qkv_w,
        &params->qkv_b,
        &params->proj_w,
        &params->proj_b
    };
    for (int i = 0; i < NUM_ATT_PARAM_TENSORS; i++) {
        *(ptrs[i]) = params_memory;
        params_memory += params->param_sizes[i];
    }
}

void set_attention_acts_pointers(
    AttentionActs* acts,
    float* acts_memory
) {
    float** ptrs[] = {
        &acts->gn,
        &acts->gn_mean,
        &acts->gn_rstd,
        &acts->perm1,
        &acts->qkv1,
        &acts->qkv2,
        &acts->preatt,
        &acts->att,
        &acts->att_out,
        &acts->proj,
        &acts->perm2,
        &acts->add
    };
    for (int i = 0; i < NUM_ATT_ACT_TENSORS; i++) {
        *(ptrs[i]) = acts_memory;
        acts_memory += acts->act_sizes[i];
    }
}

// ----------------------------------------------------
// kernels

// this is an opportunity for improvement:
// rewrite the attention layer so we don't need to permute the input
__global__ void attention_permute_kernel(
    const float* inp, float* out,
    int B, int C, int H, int W
) {
    // from B, C, H, W to B, HxW, C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = H * W;
    if (idx >= B * C * img_size) { return; }

    int b = idx / (img_size * C);
    int c = idx / img_size % C;
    int h = idx / W % H;
    int w = idx % W;

    out[b * C * img_size + h * C * W + w * C + c] = inp[idx];
}

void attention_permute(
    const float* inp, float* out,
    int B, int C, int H, int W, int block_size
) {
    int num_threads = B * C * H * W;
    attention_permute_kernel<<<ceil_div(num_threads, block_size), block_size>>>(inp, out, B, C, H, W);
}

__global__ void attention_unpermute_kernel(
    const float* inp, float* out,
    int B, int C, int H, int W
) {
    // from B, HxW, C to B, C, H, W
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_size = H * W;
    if (idx >= B * C * img_size) { return; }

    int b = idx / (img_size * C);
    int c = idx / img_size % C;
    int h = idx / W % H;
    int w = idx % W;

    out[idx] = inp[b * C * img_size + h * C * W + w * C + c];
}
void attention_unpermute(
    const float* inp, float* out,
    int B, int C, int H, int W, int block_size
) {
    int num_threads = B * C * H * W;
    attention_unpermute_kernel<<<ceil_div(num_threads, block_size), block_size>>>(inp, out, B, C, H, W);
}

void attention_block_forward
(
    cublasHandle_t cublas_handle,
    int B, int C, int H, int W, int HS, int gn_n_groups, int block_size,
    AttentionParams* params, AttentionActs* acts
) {
    // apply groupnorm
    groupnorm_forward(acts->input, params->gn_w, params->gn_b, acts->gn, acts->gn_mean, acts->gn_rstd, B, C, H, W, gn_n_groups);
    // permute from B, C, H, W to B, HxW, C
    attention_permute(acts->gn, acts->perm1, B, C, H, W, block_size);
    // get qkv
    matmul_forward2(cublas_handle, acts->qkv1, acts->perm1, params->qkv_w, params->qkv_b, B * H * W, C, 3 * C, block_size);
    // apply attention
    attention_forward1(cublas_handle, acts->att_out, acts->qkv2, acts->preatt, acts->att, acts->qkv1, B, H * W, C, C / HS, block_size);
    // attention projection
    matmul_forward2(cublas_handle, acts->proj, acts->att_out, params->proj_w, params->proj_b, B * H * W, C, C, block_size);
    // permute from B, HxW, C to B, C, H, W
    attention_unpermute(acts->proj, acts->perm2, B, C, H, W, block_size);
    // add back residual stream
    add_forward(acts->input, acts->perm2, acts->add, B * C * H * W, block_size);
}

void attention_block_backward
(
    cublasHandle_t cublas_handle,
    int B, int C, int H, int W, int HS, int gn_n_groups, int block_size,
    AttentionParams* params, AttentionActs* acts,
    AttentionBackwardActs* back_acts, AttentionParams* grads
) {
    // permute from B, C, H, W to B, HxW, C
    attention_permute(back_acts->dout, back_acts->buf1_BCHW, B, C, H, W, block_size);
    // backward through projection
    matmul_backward1(cublas_handle, back_acts->buf2_BCHW, grads->proj_w, grads->proj_b, back_acts->buf1_BCHW, acts->att_out, params->proj_w, B * H * W, C, C);
    // backward through attention
    attention_backward(cublas_handle, back_acts->buf_B3CHW, back_acts->dqkvr, back_acts->dpreatt, back_acts->datt, back_acts->buf1_BCHW, back_acts->buf2_BCHW, acts->qkv2, acts->att, B, H * W, C, C / HS);
    // backward through qkv
    matmul_backward1(cublas_handle, back_acts->buf1_BCHW, grads->qkv_w, grads->qkv_b, back_acts->buf_B3CHW, acts->perm1, params->qkv_w, B * H * W, C, 3 * C);
    // permute from B, HxW, C to B, C, H, W
    attention_unpermute(back_acts->buf1_BCHW, back_acts->buf2_BCHW, B, C, H, W, block_size);
    // backward through groupnorm
    groupnorm_backward(back_acts->buf2_BCHW, acts->input, acts->gn_mean, acts->gn_rstd, params->gn_w, back_acts->buf1_BCHW, grads->gn_w, grads->gn_b, B, C, H, W, gn_n_groups);
    // add residual gradient
    add_forward(back_acts->buf1_BCHW, back_acts->dout, back_acts->dinp, B * C * H * W, block_size);
}

// ------------------------------------------------------------------------------------------------
// MSE loss
// this will only be called with one block
__global__ void mse_forward_kernel(
    const float* inp, const float* y, float* loss, int N
) {
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

// ------------------------------------------------------------------------------------------------
// DataLoader

typedef struct {
    int B;
    int C_in;
    int H;
    int W;
    int img_size;
    int n_imgs;
    FILE* img_file;
    long cur_position;
    long file_size;
    long header_offset;
    long num_batches;
    float* input;
} DataLoader;

void dataloader_init(DataLoader* loader, const char* filename, int B) {
    loader->B = B;

    loader->img_file = fopenCheck(filename, "rb");
    int data_header[256];
    loader->header_offset = 256 * sizeof(int);
    freadCheck(data_header, sizeof(int), 256, loader->img_file);
    if (data_header[0] != 20240620) {
        fprintf(stderr, "Invalid magic data file\n");
        exit(1);
    }
    loader->n_imgs = data_header[1];
    loader->C_in = data_header[2];
    loader->H = data_header[3];
    loader->W = data_header[4];
    loader->img_size = loader->C_in * loader->H * loader->W;

    // determine file size
    fseekCheck(loader->img_file, 0, SEEK_SET);
    fseekCheck(loader->img_file, 0, SEEK_END);
    loader->file_size = ftell(loader->img_file);
    fseekCheck(loader->img_file, 0, SEEK_SET);
    assert((loader->file_size - loader->header_offset) / (loader->img_size * sizeof(float)) == loader->n_imgs);
    
    loader->cur_position = loader->header_offset;
    loader->num_batches = loader->n_imgs / B;

    // allocate cuda CPU pinned memory for inputs
    cudaMallocHost((void**)&loader->input, B * loader->img_size * sizeof(float));
}

void dataloader_reset(DataLoader* loader) {
    loader->cur_position = loader->header_offset;
}

void dataloader_next_batch(DataLoader* loader) {
    int B = loader->B;
    int img_size = loader->img_size;
    if (loader->cur_position + (B * img_size * sizeof(float)) > loader->file_size) {
        dataloader_reset(loader);
    }
    fseekCheck(loader->img_file, loader->cur_position, SEEK_SET);
    freadCheck(loader->input, sizeof(float), B * img_size, loader->img_file);
    loader->cur_position += B * img_size * sizeof(float);
}

void dataloader_free(DataLoader* loader) {
    fcloseCheck(loader->img_file);
    cudaFreeHost(loader->input);
}


// ------------------------------------------------------------------------------------------------
// Diffusion
typedef struct {
    int B;
    int img_size;
    int max_period;
    float* betas; // (max_period,)
    float* sqrt_alphas_cumprod; // (max_period,)
    float* sqrt_one_minus_alphas_cumprod; // (max_period,)
    int block_size;
    curandState* curand_states;
} GaussianDiffusion;

__global__ void init_curand_states(curandState *state, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

void diffusion_init(
    GaussianDiffusion* diffusion, int B, int C_in, int H, int W, int max_period, int block_size = 512,
    unsigned long seed = 0, float beta_start = 1e-4, float beta_end = 0.02f
) {
    diffusion->B = B;
    diffusion->max_period = max_period;
    diffusion->block_size = block_size;
    diffusion->img_size = C_in * H * W;

    float scale = 1000.0f / max_period;
    beta_start *= scale;
    beta_end *= scale;

    // calculate betas and alphas on host then move to device
    float* betas = (float*)mallocCheck(max_period * sizeof(float));
    float* sqrt_alphas_cumprod = (float*)mallocCheck(max_period * sizeof(float));
    float* sqrt_one_minus_alphas_cumprod = (float*)mallocCheck(max_period * sizeof(float));

    float alpha_cumprod = 1.0f;
    for (int t = 1; t <= max_period; t++) {
        // linear interpolate for beta
        betas[t - 1] = (beta_start * (max_period - t) + beta_end * (t - 1)) / (max_period - 1);
        alpha_cumprod *= 1.0f - betas[t - 1];
        sqrt_alphas_cumprod[t - 1] = sqrtf(alpha_cumprod);
        sqrt_one_minus_alphas_cumprod[t - 1] = sqrtf(1.0f - alpha_cumprod);
    }

    float *d_betas, *d_sqrt_alphas_cumprod, *d_sqrt_one_minus_alphas_cumprod;
    cudaCheck(cudaMalloc(&d_betas, max_period * sizeof(float)));
    cudaCheck(cudaMalloc(&d_sqrt_alphas_cumprod, max_period * sizeof(float)));
    cudaCheck(cudaMalloc(&d_sqrt_one_minus_alphas_cumprod, max_period * sizeof(float)));

    cudaCheck(cudaMemcpy(d_betas, betas, max_period * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sqrt_alphas_cumprod, sqrt_alphas_cumprod, max_period * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sqrt_one_minus_alphas_cumprod, sqrt_one_minus_alphas_cumprod, max_period * sizeof(float), cudaMemcpyHostToDevice));

    free(betas);
    free(sqrt_alphas_cumprod);
    free(sqrt_one_minus_alphas_cumprod);

    diffusion->betas = d_betas;
    diffusion->sqrt_alphas_cumprod = d_sqrt_alphas_cumprod;
    diffusion->sqrt_one_minus_alphas_cumprod = d_sqrt_one_minus_alphas_cumprod;

    // init curand states for sampling noise of the size of one batch of inputs
    if (diffusion->curand_states == nullptr) {
        printf("initializing curand states\n");
        int total_states = B * C_in * H * W;
        int n_blocks = ceil_div(total_states, block_size);
        cudaCheck(cudaMalloc(&diffusion->curand_states, total_states * sizeof(curandState)));
        init_curand_states<<<n_blocks, block_size>>>(diffusion->curand_states, seed, total_states);
    }
}

// fill BxN noise array with standard gaussians
__global__ void diffusion_draw_normal_kernel(
    float* noise, int B, int N, curandState* curand_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        noise[idx] = curand_normal(&curand_states[idx]);
    }
}

void diffusion_draw_normal(
    GaussianDiffusion *diffusion,
    float* noise
) {
    int B = diffusion->B;
    int img_size = diffusion->img_size;
    int total_states = B * img_size;
    int n_blocks = ceil_div(total_states, diffusion->block_size);
    diffusion_draw_normal_kernel<<<n_blocks, diffusion->block_size>>>(noise, B, img_size, diffusion->curand_states);
}

// given input x of shape (B, N) and timesteps of shape (B,)
// replace each x entry by the result of flowing forward by diffusion for time t
// with randomness coming from noise
__global__ void diffusion_forward_by_t_kernel(
    float* x, float* timesteps, float* noise,
    float* sqrt_alphas_cumprod, float* sqrt_one_minus_alphas_cumprod,
    int B, int N, curandState* curand_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        int b = idx / N;
        float x_val = x[idx];
        float noise_val = noise[idx];
        int t = (int)timesteps[b];
        float sqrt_alpha_t = sqrt_alphas_cumprod[t];
        float sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t];
        x[idx] = x_val * sqrt_alpha_t + noise_val * sqrt_one_minus_alpha_t;
    }
}

void diffusion_forward_by_t(
    GaussianDiffusion* diffusion,
    float* x, float* timesteps, float* noise
) {
    int B = diffusion->B;
    int img_size = diffusion->img_size;
    int block_size = diffusion->block_size;
    int total_states = B * img_size;
    int n_blocks = ceil_div(total_states, block_size);
    diffusion_forward_by_t_kernel<<<n_blocks, block_size>>>(
        x, timesteps, noise, diffusion->sqrt_alphas_cumprod, diffusion->sqrt_one_minus_alphas_cumprod, B, img_size, diffusion->curand_states
    );
}

void diffusion_free(GaussianDiffusion* diffusion) {
    cudaCheck(cudaFree(diffusion->betas));
    cudaCheck(cudaFree(diffusion->sqrt_alphas_cumprod));
    cudaCheck(cudaFree(diffusion->sqrt_one_minus_alphas_cumprod));
    cudaCheck(cudaFree(diffusion->curand_states));
}

// one block with B threads
__global__ void sample_timestep_kernel(
    float* timesteps, int B,
    curandState* curand_states, int max_period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        int t = curand(&curand_states[idx]) % max_period;
        timesteps[idx] = (float)t;
    }
}

// samples B timesteps in range [0, max_period), convert them to floats and
// store in timesteps
void sample_timesteps(GaussianDiffusion *diffusion, float* timesteps) {
    sample_timestep_kernel<<<1, diffusion->B>>>(timesteps, diffusion->B, diffusion->curand_states, diffusion->max_period);
}

// ------------------------------------------------------------------------------------------------
// timestep embedding states
typedef struct {
    float* freqs;
    int half_dim;
    int B;
    int max_period;
} TimestepEmbedding;

__global__ void fill_freqs_kernel(float* freqs, int half_dim, int max_period) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < half_dim) {
        float log_period = logf(max_period);
        float logit = -log_period * idx / half_dim;
        freqs[idx] = expf(logit);
    }
}

void init_timestep_embedding(TimestepEmbedding* emb, int dim, int B, int max_period) {
    assert(dim % 2 == 0);
    int half = dim / 2;
    emb->half_dim = half;
    emb->max_period = max_period;
    emb->B = B;
    cudaCheck(cudaMalloc(&emb->freqs, half * sizeof(float)));
    // fill freq values
    int block_size = 512;
    int n_blocks = ceil_div(half, block_size);
    fill_freqs_kernel<<<n_blocks, block_size>>>(emb->freqs, half, max_period);
}

void free_timestep_embedding(TimestepEmbedding* emb) {
    cudaCheck(cudaFree(emb->freqs));
}

__global__ void fill_embeddings_kernel(
    const float* timesteps, const float* freqs, float* embeddings, int half, int B
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < B * half) {
        int i = idx / half;
        int j = idx % half;
        float arg = timesteps[i] * freqs[j];
        embeddings[i * 2 * half + j] = cosf(arg);
        embeddings[i * 2 * half + j + half] = sinf(arg);
    }
}

// timesteps is a (B,) array of timesteps that are integers but saved as floats
// out is a (B, dim) array
void get_timestep_embeddings(
    TimestepEmbedding* emb, const float* timesteps, float* out
) {
    int block_size = 512;
    int n_blocks = ceil_div(emb->half_dim * emb->B, block_size);
    fill_embeddings_kernel<<<n_blocks, block_size>>>(timesteps, emb->freqs, out, emb->half_dim, emb->B);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// Unet init

typedef struct {
    int C_in;
    int C_model;
    int C_time;
    int C_out;
    int B;
    int H;
    int W;
    int gn_n_groups;
    int HS;
    int block_size;
    int n_layers;
    int n_levels;
    int n_res;
    int att_start_level;
    int* channel_mult;
    int max_period;
} UnetConfig;

// we just need a single set of buffers that are shared for all the backward pass activations
#define NUM_SKIP_CONNECTIONS 12 // (n_res + 1) * n_levels
#define NUM_BACK_BUFS 10
#define UNET_NUM_BACK_ACTS 22
typedef struct {
    float* dinp_buf1; // used to store the input gradient, which will be reused as dout for the previous layer
    float* dinp_buf2;
    float* buf1_BCHW; // NOTE: first 4 buffers will have  max B * C * H * W size
    float* buf2_BCHW;
    float* buf3_BCHW;
    float* buf4_BCHW;
    float* buf1_B3CHW; // NOTE: last 2 buffers will have max B * 3 * C * H * W size for attention layers
    float* buf2_B3CHW; // so max of H * W is only over dimensions that have attention
    float* demb_buf1; // used to store cumulative gradients w.r.t. timestep embeddings
    float* demb_buf2;
    float* skip_dinps[NUM_SKIP_CONNECTIONS];
} UnetBackActs;

// This is used as a block to store all the params / grads / acts for all the layers
typedef struct {
    float* mem_ptr;
    size_t total_size;
    int n_layers;
    void** layer_ptrs;
    size_t* layer_sizes;
} UnetBlock;

typedef struct {
    float* back_mem;
    UnetBackActs back_acts;
    size_t back_sizes[UNET_NUM_BACK_ACTS];
    size_t n_backs;
} UnetBackBlock;

typedef struct {
    UnetConfig config;
    float* input;
    float* timestep_emb; // this is the already computed embeddings of shape (B, C_model)
    UnetBlock params;
    UnetBlock grads;
    UnetBlock acts;
    UnetBackBlock back;
    float* skip_acts[NUM_SKIP_CONNECTIONS]; // pointers to skip connetion activations
    // the next two are for debug purposes only, just to keep track of the buffer that contains the grad w.r.t. inputs
    float* dinp;
    // model output
    float* output;
    // targets and losses
    float* target;
    float mean_loss;
    float* gpu_loss;
    // buffers for AdamW
    float* m_memory;
    float* v_memory;
    TimestepEmbedding time_emb_state;
} Unet;

void unet_init(Unet* unet) {
    unet->input = nullptr;
    unet->timestep_emb = nullptr;
    unet->params.mem_ptr = nullptr;
    unet->grads.mem_ptr = nullptr;
    unet->acts.mem_ptr = nullptr;
    unet->back.back_mem = nullptr;
    unet->dinp = nullptr;
    unet->output = nullptr;
    unet->target = nullptr;
    unet->mean_loss = -1.0f;
    unet->gpu_loss = nullptr;
    unet->m_memory = nullptr;
    unet->v_memory = nullptr;

    init_timestep_embedding(&unet->time_emb_state, unet->config.C_model, unet->config.B, unet->config.max_period);
}


template <typename T>
void free_not_null(T** ptr) {
    if (*ptr != nullptr) {
        free(*ptr);
        *ptr = nullptr;
    }
}

void free_unet_block(UnetBlock* params) {
    free_not_null(&params->layer_sizes);
    for (int l = 0; l < params->n_layers; l++) {
        free_not_null(&(params->layer_ptrs[l]));
    }
    free_not_null(&params->layer_ptrs);
    cudaCheck(cudaFree(params->mem_ptr));
}

void free_unet(Unet* unet) {
    //free(unet->config.channel_mult);

    cudaCheck(cudaFree(unet->input));
    cudaCheck(cudaFree(unet->timestep_emb));
    cudaCheck(cudaFree(unet->gpu_loss));
    cudaCheck(cudaFree(unet->m_memory));
    cudaCheck(cudaFree(unet->v_memory));

    free_unet_block(&(unet->params));
    free_unet_block(&(unet->grads));
    free_unet_block(&(unet->acts));

    cudaCheck(cudaFree(unet->back.back_mem));
    free_timestep_embedding(&unet->time_emb_state);
}

void unet_count_layers(UnetConfig* config) {
    int n_layers = 0;
    int n_res = config->n_res;
    int n_levels = config->n_levels;
    int att_start_level = config->att_start_level;

    // timestep emb
    n_layers += 3;

    // initial conv
    n_layers += 1;

    // add downsample resblock layers
    n_layers += n_res * n_levels;

    // add downsample attention
    n_layers += n_res * (n_levels - att_start_level);

    // add avgpool layers
    n_layers += n_levels - 1;

    // add middle layers
    n_layers += 3;

    // add upsample concat layers
    n_layers += (n_res + 1) * n_levels;

    // add upsample resblock layers
    n_layers += (n_res + 1) * n_levels;

    // add upsample attention layers
    n_layers += (n_res + 1) * (n_levels - att_start_level);

    // add upsample layers
    n_layers += n_levels - 1;

    // output layers
    n_layers += 3;

    config->n_layers = n_layers;
}

// ------------------------------------------------------------------------------------------------
// allocate memory for structs with info for each layer and count params

int make_linear_params_ptrs(
    UnetBlock* params,
    int layer,
    int C_in, int C_out
) {
    params->layer_ptrs[layer] = (void*)mallocCheck(sizeof(LinearParams));
    params->layer_sizes[layer] = linear_count_params(C_in, C_out);
    return layer + 1;
}

int make_silu_params_ptrs(UnetBlock* params, int layer) {
    params->layer_ptrs[layer] = nullptr;
    params->layer_sizes[layer] = 0;
    return layer + 1;
}

int make_convk3_params_ptrs(
    UnetBlock* params, int layer,
    int C_in, int OC, int B, int H, int W
) {
    params->layer_ptrs[layer] = (void*)mallocCheck(sizeof(ConvK3Params));
    params->layer_sizes[layer] = convk3_count_params(C_in, OC);
    return layer + 1;
}

int make_resblock_params_ptrs(
    UnetBlock* params, int layer,
    int C, int C_emb, int C_out, int B, int H, int W, int gn_n_groups
) {
    params->layer_ptrs[layer] = (void*)mallocCheck(sizeof(ResBlockParameters));
    ResBlockParameters* resblock_params = (ResBlockParameters*)params->layer_ptrs[layer];
    resblock_count_params(resblock_params, C, C_emb, C_out, B, H, W, 0, 0, gn_n_groups);
    params->layer_sizes[layer] = resblock_params->n_params;
    return layer + 1;
}

int make_avgpool_params_ptrs(UnetBlock* params, int layer) {
    params->layer_ptrs[layer] = nullptr;
    params->layer_sizes[layer] = 0;
    return layer + 1;
}

int make_attention_block_params_ptrs(UnetBlock* params, int layer, int C) {
    params->layer_ptrs[layer] = (void*)mallocCheck(sizeof(AttentionParams));
    AttentionParams* attention_params = (AttentionParams*)params->layer_ptrs[layer];
    attention_block_count_params(attention_params, C);
    params->layer_sizes[layer] = attention_params->n_params;
    return layer + 1;
}

int make_concat_params_ptrs(UnetBlock* params, int layer) {
    params->layer_ptrs[layer] = nullptr;
    params->layer_sizes[layer] = 0;
    return layer + 1;
}

int make_upsample_params_ptrs(UnetBlock* params, int layer) {
    params->layer_ptrs[layer] = nullptr;
    params->layer_sizes[layer] = 0;
    return layer + 1;
}

int make_groupnorm_params_ptrs(UnetBlock* params, int layer, int C) {
    params->layer_ptrs[layer] = (void*)mallocCheck(sizeof(GroupNormParams));
    params->layer_sizes[layer] = C * 2;
    return layer + 1;
}

void unet_setup_and_count_params(UnetConfig* config, UnetBlock* params) {
    assert (config->n_layers > 0);

    params->n_layers = config->n_layers;
    params->layer_ptrs = (void**)mallocCheck(config->n_layers * sizeof(void*));
    params->layer_sizes = (size_t*)mallocCheck(config->n_layers * sizeof(size_t));

    int B = config->B;
    int C_in = config->C_in;
    int C_model = config->C_model;
    int C_time = config->C_time;
    int C_out = config->C_out;
    int H = config->H;
    int W = config->W;
    int gn_n_groups = config->gn_n_groups;
    int n_levels = config->n_levels;
    int n_res = config->n_res;
    int att_start_level = config->att_start_level;

    int layer = 0;
    layer = make_linear_params_ptrs(params, layer, C_model, C_time);
    layer = make_silu_params_ptrs(params, layer);
    layer = make_linear_params_ptrs(params, layer, C_time, C_time);
    int OC = C_model;
    layer = make_convk3_params_ptrs(params, layer, C_in, OC, B, H, W);

    int C = OC;
    for (int l = 0; l < n_levels; l++) {
        int mult = config->channel_mult[l];
        OC = C_model * mult;
        for (int r = 0; r < n_res; r++) {
            layer = make_resblock_params_ptrs(params, layer, C, C_time, OC, B, H, W, gn_n_groups);
            C = OC;
            if (l >= att_start_level) {
                layer = make_attention_block_params_ptrs(params, layer, C);
            }
        }
        if (l < n_levels - 1) {
            layer = make_avgpool_params_ptrs(params, layer);
            H /= 2;
            W /= 2;
        }
    }

    // middle layers
    layer = make_resblock_params_ptrs(params, layer, C, C_time, C, B, H, W, gn_n_groups);
    layer = make_attention_block_params_ptrs(params, layer, C);
    layer = make_resblock_params_ptrs(params, layer, C, C_time, C, B, H, W, gn_n_groups);

    // upwards: concat then resblock
    for (int l = n_levels - 1; l >= 0; l--) {
        int mult = config->channel_mult[l];
        int C_skip = C_model * mult;
        for (int r = 0; r < n_res + 1; r++) {
            if (r == n_res && l > 0) {
                int prev_mult = config->channel_mult[l - 1];
                C_skip = C_model * prev_mult;
            }
            OC = C_model * mult;
            layer = make_concat_params_ptrs(params, layer);
            layer = make_resblock_params_ptrs(params, layer, C + C_skip, C_time, OC, B, H, W, gn_n_groups);
            C = OC;
            if (l >= att_start_level) {
                layer = make_attention_block_params_ptrs(params, layer, C);
            }
        }
        if (l > 0) {
            layer = make_upsample_params_ptrs(params, layer);
            H *= 2;
            W *= 2;
        }
    }

    layer = make_groupnorm_params_ptrs(params, layer, C);
    layer = make_silu_params_ptrs(params, layer);
    layer = make_convk3_params_ptrs(params, layer, C, C_out, B, H, W);

    // count total size
    size_t total_size = 0;
    for (int i = 0; i < config->n_layers; i++) {
        total_size += params->layer_sizes[i];
    }
    params->total_size = total_size;
}

// ------------------------------------------------------------------------------------------------
// setup Unet activations pointers
int make_linear_acts_ptrs(
    UnetBlock* acts,
    int layer,
    int B, int C_out
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(LinearActs));
    // acts buffers only hold outputs
    acts->layer_sizes[layer] = B * C_out;
    return layer + 1;
}

int make_silu_acts_ptrs(UnetBlock* acts, int layer, int N) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(SiluActs));
    acts->layer_sizes[layer] = N;
    return layer + 1;
}

int make_convk3_acts_ptrs(
    UnetBlock* acts, int layer,
    int B, int OC, int H, int W
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(ConvK3Acts));
    acts->layer_sizes[layer] = B * OC * H * W;
    return layer + 1;
}

int make_resblock_acts_ptrs(
    UnetBlock* acts, int layer,
    int C, int C_emb, int C_out, int B, int H, int W, int gn_n_groups
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(ResBlockActivations));
    ResBlockActivations* resblock_acts = (ResBlockActivations*)acts->layer_ptrs[layer];
    resblock_count_acts(resblock_acts, C, C_emb, C_out, B, H, W, 0, 0, gn_n_groups);
    acts->layer_sizes[layer] = resblock_acts->n_acts;
    return layer + 1;
}

int make_avgpool_acts_ptrs(UnetBlock* acts, int layer, int B, int C, int H, int W) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(AvgpoolActs));
    acts->layer_sizes[layer] = B * C * H/2 * W/2;
    return layer + 1;
}

int make_attention_block_acts_ptrs(
    UnetBlock* acts, int layer,
    int B, int C, int H, int W, int gn_n_groups, int HS
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(AttentionActs));
    AttentionActs* att_acts = (AttentionActs*)acts->layer_ptrs[layer];
    attention_block_count_acts(att_acts, B, C, H, W, gn_n_groups, HS);
    acts->layer_sizes[layer] = att_acts->n_acts;
    return layer + 1;
}

int make_concat_acts_ptrs(
    UnetBlock* acts, int layer,
    int B, int C1, int C2, int H, int W
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(ConcatChannelActs));
    acts->layer_sizes[layer] = B * (C1 + C2) * H * W;
    return layer + 1;
}

int make_upsample_acts_ptrs(
    UnetBlock* acts, int layer,
    int B, int C, int H, int W
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(UpsampleActs));
    acts->layer_sizes[layer] = B * C * H * 2 * W * 2;
    return layer + 1;
}

int make_groupnorm_acts_ptrs(
    UnetBlock* acts, int layer,
    int B, int C, int H, int W, int gn_n_groups
) {
    acts->layer_ptrs[layer] = (void*)mallocCheck(sizeof(GroupNormActs));
    // out (B, C, H, W), mean and rstd (B, gn_n_groups)
    acts->layer_sizes[layer] = B * C * H * W + B * gn_n_groups * 2;
    return layer + 1;
}

void unet_setup_and_count_acts(UnetConfig* config, UnetBlock* acts) {
    assert (config->n_layers > 0);

    acts->n_layers = config->n_layers;
    acts->layer_ptrs = (void**)mallocCheck(config->n_layers * sizeof(void*));
    acts->layer_sizes = (size_t*)mallocCheck(config->n_layers * sizeof(size_t));

    int B = config->B;
    int C_model = config->C_model;
    int C_time = config->C_time;
    int C_out = config->C_out;
    int H = config->H;
    int W = config->W;
    int gn_n_groups = config->gn_n_groups;
    int HS = config->HS;
    int n_levels = config->n_levels;
    int n_res = config->n_res;
    int att_start_level = config->att_start_level;

    int layer = 0;
    layer = make_linear_acts_ptrs(acts, layer, B, C_time);
    layer = make_silu_acts_ptrs(acts, layer, B * C_time);
    layer = make_linear_acts_ptrs(acts, layer, B, C_time);
    int OC = C_model;
    layer = make_convk3_acts_ptrs(acts, layer, B, OC, H, W);

    int C = OC;
    for (int l = 0; l < n_levels; l++) {
        int mult = config->channel_mult[l];
        OC = C_model * mult;
        for (int r = 0; r < n_res; r++) {
            layer = make_resblock_acts_ptrs(acts, layer, C, C_time, OC, B, H, W, gn_n_groups);
            C = OC;
            if (l >= att_start_level) {
                layer = make_attention_block_acts_ptrs(acts, layer, B, C, H, W, gn_n_groups, HS);
            }
        }
        if (l < n_levels - 1) {
            layer = make_avgpool_acts_ptrs(acts, layer, B, C, H, W);
            H /= 2;
            W /= 2;
        }
    }

    layer = make_resblock_acts_ptrs(acts, layer, C, C_time, C, B, H, W, gn_n_groups);
    layer = make_attention_block_acts_ptrs(acts, layer, B, C, H, W, gn_n_groups, HS);
    layer = make_resblock_acts_ptrs(acts, layer, C, C_time, C, B, H, W, gn_n_groups);

    // upwards
    for (int l = n_levels - 1; l >= 0; l--) {
        int mult = config->channel_mult[l];
        int C_skip = mult * C_model;
        for (int r = 0; r <= n_res; r++) {
            if (r == n_res && l > 0) {
                int prev_mult = config->channel_mult[l - 1];
                C_skip = prev_mult * C_model;
            }
            layer = make_concat_acts_ptrs(acts, layer, B, C, C_skip, H, W);
            OC = mult * C_model;
            layer = make_resblock_acts_ptrs(acts, layer, C + C_skip, C_time, OC, B, H, W, gn_n_groups);
            C = OC;
            if (l >= att_start_level) {
                layer = make_attention_block_acts_ptrs(acts, layer, B, C, H, W, gn_n_groups, HS);
            }
        }
        if (l > 0) {
            layer = make_upsample_acts_ptrs(acts, layer, B, C, H, W);
            H *= 2;
            W *= 2;
        }
    }
    layer = make_groupnorm_acts_ptrs(acts, layer, B, C, H, W, gn_n_groups);
    layer = make_silu_acts_ptrs(acts, layer, B * C * H * W);
    layer = make_convk3_acts_ptrs(acts, layer, B, C_out, H, W);

    // count total size
    size_t total_size = 0;
    for (int i = 0; i < config->n_layers; i++) {
        total_size += acts->layer_sizes[i];
    }
    acts->total_size = total_size;
}

void unet_setup_and_count_backs(UnetConfig* config, UnetBackBlock* back) {
    int B = config->B;
    int C_model = config->C_model;
    int C_time = config->C_time;
    int H = config->H;
    int W = config->W;

    size_t max_BCHW = 3 * B * C_model * H * W;
    size_t max_B3CHW = 2 * B * 3 * C_model * H * W;
    back->back_sizes[0] = max_BCHW;
    back->back_sizes[1] = max_BCHW;
    back->back_sizes[2] = max_BCHW;
    back->back_sizes[3] = max_BCHW;
    back->back_sizes[4] = max_BCHW;
    back->back_sizes[5] = max_BCHW;
    back->back_sizes[6] = max_B3CHW;
    back->back_sizes[7] = max_B3CHW;
    back->back_sizes[8] = B * C_time;
    back->back_sizes[9] = B * C_time;

    // fill the sizes of the skip connection backward acts
    int C = C_model;
    for (int l = 0; l < config->n_levels; l++) {
        for (int r = 0; r < (config->n_res + 1); r++) {
            int s = l * 3 + r + NUM_BACK_BUFS;
            assert(s < UNET_NUM_BACK_ACTS);
            back->back_sizes[s] = B * C * H * W;
            int mult = config->channel_mult[l];
            C = C_model * mult;
            if (l > 0 && r == 0) {
                H /= 2;
                W /= 2;
            }
        }
    }

    size_t n_backs = 0;
    for (int i = 0; i < UNET_NUM_BACK_ACTS; i++) {
        n_backs += back->back_sizes[i];
    }
    back->n_backs = n_backs;
}


void unet_make_ptrs_and_count_memory(Unet* unet)
{
    unet_setup_and_count_params(&(unet->config), &(unet->params));
    unet_setup_and_count_params(&(unet->config), &(unet->grads));
    unet_setup_and_count_acts(&(unet->config), &(unet->acts));
    unet_setup_and_count_backs(&(unet->config), &(unet->back));
}

// ------------------------------------------------------------------------------------------------
// malloc unet memory and set pointers
typedef struct {
    float* input;
    float* prev_out;
    int layer;
    float* emb;
    float* emb_cum; // cumulative demb
    float* mem_ptr;
    int skip_idx;
} LayerInfo;


void set_linear_params_ptrs(
    UnetBlock* params,
    LayerInfo* l_info,
    int C_in, int C_out
) {
    int layer = l_info->layer;
    LinearParams* linear_params = (LinearParams*)params->layer_ptrs[layer];
    linear_set_param_ptrs(linear_params, l_info->mem_ptr, C_in, C_out);
    l_info->mem_ptr += params->layer_sizes[layer];
    l_info->layer++;
}

void set_convk3_params_ptrs(UnetBlock* params, LayerInfo* l_info, int C_in, int OC) {
    int layer = l_info->layer;
    ConvK3Params* convk3_params = (ConvK3Params*)params->layer_ptrs[layer];
    convk3_set_param_ptrs(convk3_params, l_info->mem_ptr, C_in, OC);
    l_info->mem_ptr += params->layer_sizes[layer];
    l_info->layer++;
}

void set_resblock_layer_params_ptrs(UnetBlock* params, LayerInfo* l_info, int C, int OC) {
    int layer = l_info->layer;
    ResBlockParameters* resblock_params = (ResBlockParameters*)params->layer_ptrs[layer];
    set_resblock_params_ptrs(C, OC, resblock_params, l_info->mem_ptr);
    l_info->mem_ptr += params->layer_sizes[layer];
    l_info->layer++;
}

void set_attention_layer_params_ptrs(UnetBlock* params, LayerInfo* l_info) {
    int layer = l_info->layer;
    AttentionParams* att_params = (AttentionParams*)params->layer_ptrs[layer];
    set_attention_params_pointers(att_params, l_info->mem_ptr);
    l_info->mem_ptr += params->layer_sizes[layer];
    l_info->layer++;
}

void set_groupnorm_layer_params_ptrs(UnetBlock* params, LayerInfo* l_info, int C) {
    int layer = l_info->layer;
    GroupNormParams* gn_params = (GroupNormParams*)params->layer_ptrs[layer];
    gn_set_param_ptrs(gn_params, l_info->mem_ptr, C);
    l_info->mem_ptr += params->layer_sizes[layer];
    l_info->layer++;
}


void point_unet_params(UnetConfig* config, UnetBlock* params) {
    // common constants
    int C_in = config->C_in;
    int C_model = config->C_model;
    int C_time = config->C_time;
    int C_out = config->C_out;
    int n_levels = config->n_levels;
    int n_res = config->n_res;
    int att_start_level = config->att_start_level;

    LayerInfo l_info;
    l_info.layer = 0;
    l_info.mem_ptr = params->mem_ptr;
    set_linear_params_ptrs(params, &l_info, C_model, C_time);
    l_info.layer++; // skip silu layer
    set_linear_params_ptrs(params, &l_info, C_time, C_time);
    int OC = C_model;
    set_convk3_params_ptrs(params, &l_info, C_in, OC);

    int C = OC;
    for (int l = 0; l < n_levels; l++) {
        int mult = config->channel_mult[l];
        OC = C_model * mult;
        for (int r = 0; r < n_res; r++) {
            set_resblock_layer_params_ptrs(params, &l_info, C, OC);
            C = OC;
            if (l >= att_start_level) {
                set_attention_layer_params_ptrs(params, &l_info);
            }
        }
        if (l < n_levels - 1) {
            l_info.layer++; // skip avgpool layer
        }
    }

    set_resblock_layer_params_ptrs(params, &l_info, C, C);
    set_attention_layer_params_ptrs(params, &l_info);
    set_resblock_layer_params_ptrs(params, &l_info, C, C);

    // upward: skip one layer for concat
    for (int l = n_levels - 1; l >= 0; l--) {
        int mult = config->channel_mult[l];
        int C_skip = mult * C_model;
        for (int r = 0; r <= n_res; r++) {
            if (r == n_res && l > 0) {
                C_skip = C_model * config->channel_mult[l - 1];
            }
            l_info.layer++;
            OC = mult * C_model;
            set_resblock_layer_params_ptrs(params, &l_info, C + C_skip, OC);
            C = OC;
            if (l >= att_start_level) {
                set_attention_layer_params_ptrs(params, &l_info);
            }
        }
        if (l > 0) {
            l_info.layer++; // upsample
        }
    }

    set_groupnorm_layer_params_ptrs(params, &l_info, C);
    l_info.layer++; // silu
    set_convk3_params_ptrs(params, &l_info, C, C_out);
}

void set_linear_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    LinearActs* linear_acts = (LinearActs*)acts->layer_ptrs[layer];
    linear_acts->out = l_info->mem_ptr;
    l_info->mem_ptr += acts->layer_sizes[layer];
    linear_acts->inp = l_info->input;
    l_info->input = linear_acts->out;
    l_info->layer++;
}

void set_silu_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    SiluActs* silu_acts = (SiluActs*)acts->layer_ptrs[layer];
    silu_acts->out = l_info->mem_ptr;
    l_info->mem_ptr += acts->layer_sizes[layer];
    silu_acts->x = l_info->input;
    l_info->input = silu_acts->out;
    l_info->layer++;
}

void set_convk3_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    ConvK3Acts* convk3_acts = (ConvK3Acts*)acts->layer_ptrs[layer];
    convk3_acts->out = l_info->mem_ptr;
    l_info->mem_ptr += acts->layer_sizes[layer];
    convk3_acts->inp = l_info->input;
    l_info->input = convk3_acts->out;
    l_info->layer++;
}

void set_resblock_layer_acts_ptrs(UnetBlock* acts, LayerInfo* l_info, int C, int OC) {
    int layer = l_info->layer;
    ResBlockActivations* resblock_acts = (ResBlockActivations*)acts->layer_ptrs[layer];
    set_resblock_acts_ptrs(C, OC, resblock_acts, l_info->mem_ptr);
    resblock_acts->input = l_info->input;
    resblock_acts->emb = l_info->emb;
    l_info->mem_ptr += acts->layer_sizes[layer];
    l_info->input = resblock_acts->add2;
    l_info->layer++;
}

void set_avgpool_layer_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    AvgpoolActs* avgpool_acts = (AvgpoolActs*)acts->layer_ptrs[layer];
    avgpool_acts->out = l_info->mem_ptr;
    avgpool_acts->x = l_info->input;
    l_info->mem_ptr += acts->layer_sizes[layer];
    l_info->input = avgpool_acts->out;
    l_info->layer++;
}

void set_attention_layer_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    AttentionActs* att_acts = (AttentionActs*)acts->layer_ptrs[layer];
    set_attention_acts_pointers(att_acts, l_info->mem_ptr);
    att_acts->input = l_info->input;
    l_info->mem_ptr += acts->layer_sizes[layer];
    l_info->input = att_acts->add;
    l_info->layer++;
}

void set_concat_layer_acts_ptrs(UnetBlock* acts, LayerInfo* l_info, float** skip_acts) {
    int layer = l_info->layer;
    ConcatChannelActs* cat_acts = (ConcatChannelActs*)acts->layer_ptrs[layer];
    cat_acts->out = l_info->mem_ptr;
    cat_acts->x1 = l_info->input;
    cat_acts->x2 = skip_acts[l_info->skip_idx];

    l_info->mem_ptr += acts->layer_sizes[layer];
    l_info->input = cat_acts->out;
    l_info->layer++;
    l_info->skip_idx--;
}

void set_upsample_layer_acts_ptrs(UnetBlock* acts, LayerInfo* l_info) {
    int layer = l_info->layer;
    UpsampleActs* up_acts = (UpsampleActs*)acts->layer_ptrs[layer];
    up_acts->x = l_info->input;
    up_acts->out = l_info->mem_ptr;

    l_info->mem_ptr += acts-> layer_sizes[layer];
    l_info->input = up_acts->out;
    l_info->layer++;
}

void set_groupnorm_layer_acts_ptrs(
    UnetBlock* acts, LayerInfo* l_info,
    int B, int C, int H, int W, int gn_n_groups
) {
    int layer = l_info->layer;
    GroupNormActs* gn_a = (GroupNormActs*)acts->layer_ptrs[layer];
    gn_set_act_ptrs(gn_a, l_info->mem_ptr, B, C, H, W, gn_n_groups);
    gn_a->x = l_info->input;
    l_info->mem_ptr += acts->layer_sizes[layer];
    l_info->input = gn_a->out;
    l_info->layer++;
}

void point_unet_acts(UnetConfig* config, UnetBlock* acts, float** skip_acts, float* input, float* timestep_emb) {
    // common constants
    int B = config->B;
    int C_model = config->C_model;
    int H = config->H;
    int W = config->W;
    int gn_n_groups = config->gn_n_groups;
    int n_levels = config->n_levels;
    int n_res = config->n_res;
    int att_start_level = config->att_start_level;

    LayerInfo l_info;
    l_info.layer = 0;
    l_info.mem_ptr = acts->mem_ptr;

    // timestep emb
    l_info.input = timestep_emb;
    set_linear_acts_ptrs(acts, &l_info);
    set_silu_acts_ptrs(acts, &l_info);
    set_linear_acts_ptrs(acts, &l_info);
    // point input to model input, and set outputu of the last linear layer as the embedding input
    l_info.emb = l_info.input;
    l_info.input = input;

    int OC = C_model;
    set_convk3_acts_ptrs(acts, &l_info);
    int skip_idx = 0;
    skip_acts[skip_idx] = l_info.input; // set the output of the previouse layer as input to the skip connection
    skip_idx++;


    int C = OC;
    for (int l = 0; l < n_levels; l++) {
        int mult = config->channel_mult[l];
        OC = C_model * mult;
        for (int r = 0; r < n_res; r++) {
            set_resblock_layer_acts_ptrs(acts, &l_info, C, OC);
            if (l < att_start_level) {
                // for the first 2 layers, the skip connection comes from the resblocks
                skip_acts[skip_idx] = l_info.input;
                skip_idx++;
            }

            C = OC;
            if (l >= att_start_level) {
                set_attention_layer_acts_ptrs(acts, &l_info);
                skip_acts[skip_idx] = l_info.input;
                skip_idx++;
            }
        }
        if (l < n_levels - 1) {
            set_avgpool_layer_acts_ptrs(acts, &l_info);
            skip_acts[skip_idx] = l_info.input;
            skip_idx++;
        }
    }

    set_resblock_layer_acts_ptrs(acts, &l_info, C, C);
    set_attention_layer_acts_ptrs(acts, &l_info);
    set_resblock_layer_acts_ptrs(acts, &l_info, C, C);

    // upward: set concat
    l_info.skip_idx = NUM_SKIP_CONNECTIONS - 1;
    for (int l = n_levels - 1; l >= 0; l--) {
        int mult = config->channel_mult[l];
        int C_skip = mult * C_model;
        for (int r = 0; r <= n_res; r++) {
            if (r == n_res && l > 0) {
                C_skip = C_model * config->channel_mult[l - 1];
            }
            set_concat_layer_acts_ptrs(acts, &l_info, skip_acts);
            OC = mult * C_model;
            set_resblock_layer_acts_ptrs(acts, &l_info, C + C_skip, OC);
            C = OC;
            if (l >= att_start_level) {
                set_attention_layer_acts_ptrs(acts, &l_info);
            }
        }
        if (l > 0) {
            set_upsample_layer_acts_ptrs(acts, &l_info);
        }
    }
    set_groupnorm_layer_acts_ptrs(acts, &l_info, B, C, H, W, gn_n_groups);
    set_silu_acts_ptrs(acts, &l_info);
    set_convk3_acts_ptrs(acts, &l_info);
}

void set_unet_back_pointers(UnetBackBlock* back) {
    UnetBackActs* back_acts = &(back->back_acts);
    float* back_mem_ptr = back->back_mem;

    float** ptrs[] = {
        &(back_acts->dinp_buf1),
        &(back_acts->dinp_buf2),
        &(back_acts->buf1_BCHW),
        &(back_acts->buf2_BCHW),
        &(back_acts->buf3_BCHW),
        &(back_acts->buf4_BCHW),
        &(back_acts->buf1_B3CHW),
        &(back_acts->buf2_B3CHW),
        &(back_acts->demb_buf1),
        &(back_acts->demb_buf2)
    };

    for (int i = 0; i < NUM_BACK_BUFS; i++) {
        *(ptrs[i]) = back_mem_ptr;
        back_mem_ptr += back->back_sizes[i];
    }

    // point the back acts for skip connections
    for (int s = 0; s < NUM_SKIP_CONNECTIONS; s++) {
        back_acts->skip_dinps[s] = back_mem_ptr;
        back_mem_ptr += back->back_sizes[NUM_BACK_BUFS + s];
    }
}

void malloc_and_point(Unet* unet) {
    float *param_mem, *grad_mem, *act_mem, *back_mem;
    cudaCheck(cudaMalloc(&param_mem, unet->params.total_size * sizeof(float)));
    cudaCheck(cudaMalloc(&grad_mem, unet->params.total_size * sizeof(float)));
    cudaCheck(cudaMalloc(&act_mem, unet->acts.total_size * sizeof(float)));
    cudaCheck(cudaMalloc(&back_mem, unet->back.n_backs * sizeof(float)));

    cudaCheck(cudaMemset(param_mem, 0, unet->params.total_size * sizeof(float)));
    cudaCheck(cudaMemset(grad_mem, 0, unet->params.total_size * sizeof(float)));
    cudaCheck(cudaMemset(act_mem, 0, unet->acts.total_size * sizeof(float)));
    cudaCheck(cudaMemset(back_mem, 0, unet->back.n_backs * sizeof(float)));

    // constants
    int B = unet->config.B;
    int C_in = unet->config.C_in;
    int C_model = unet->config.C_model;
    int H = unet->config.H;
    int W = unet->config.W;

    // malloc and point input and timestep emb
    cudaCheck(cudaMalloc(&(unet->input), B * C_in * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&(unet->timestep_emb), B * C_model * sizeof(float)));
    // malloc a nicely aligned buffer for the loss
    cudaCheck(cudaMalloc(&(unet->gpu_loss), 64 * sizeof(float)));

    unet->params.mem_ptr = param_mem;
    unet->grads.mem_ptr = grad_mem;
    unet->acts.mem_ptr = act_mem;
    unet->back.back_mem = back_mem;

    point_unet_params(&(unet->config), &(unet->params));
    point_unet_params(&(unet->config), &(unet->grads));
    point_unet_acts(&(unet->config), &(unet->acts), unet->skip_acts, unet->input, unet->timestep_emb);
    set_unet_back_pointers(&(unet->back));
}

// ------------------------------------------------------------------------------------------------
// forward pass

void linear_layer_forward(
    Unet* unet,
    LayerInfo* l_info,
    cublasHandle_t cublas_handle, int N, int C, int OC, int block_size
) {
    int layer = l_info->layer;
    LinearParams* linear_params = (LinearParams*)unet->params.layer_ptrs[layer];
    LinearActs* linear_acts = (LinearActs*)unet->acts.layer_ptrs[layer];
    matmul_forward2(cublas_handle, linear_acts->out, linear_acts->inp, linear_params->w, linear_params->b, N, C, OC, block_size);
    l_info->layer++;
}

void silu_layer_forward(
    Unet* unet, LayerInfo* l_info,
    int N, int block_size
) {
    SiluActs* acts = (SiluActs*)unet->acts.layer_ptrs[l_info->layer];
    silu_forward(acts->x, acts->out, N, block_size);
    l_info->layer++;
}

void convk3_layer_forward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int B, int C, int OC, int H, int W, int block_size
) {
    int layer = l_info->layer;
    ConvK3Params* conv_params = (ConvK3Params*)unet->params.layer_ptrs[layer];
    ConvK3Acts* conv_acts = (ConvK3Acts*)unet->acts.layer_ptrs[layer];
    conv2d_k3_forward3(conv_acts->inp, conv_params->w, conv_params->b, conv_acts->out, B, C, OC, H, W);
    l_info->layer++;
}
void resblock_layer_forward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int C_in, int C_emb, int C_out, int B, int H, int W, int gn_n_groups, int block_size
) {
    int layer = l_info->layer;
    ResBlockParameters* resblock_params = (ResBlockParameters*)unet->params.layer_ptrs[layer];
    ResBlockActivations* resblock_acts = (ResBlockActivations*)unet->acts.layer_ptrs[layer];
    resblock_forward(cublas_handle, C_in, C_emb, C_out, B, H, W, block_size, 0, 0, gn_n_groups, resblock_params, resblock_acts);
    l_info->layer++;
}

void avgpool_layer_forward(
    Unet* unet, LayerInfo* l_info,
    int B, int C, int H, int W, int block_size
) {
    int layer = l_info->layer;
    AvgpoolActs* avgpool_acts = (AvgpoolActs*)unet->acts.layer_ptrs[layer];
    avgpool_2d_forward1(avgpool_acts->out, avgpool_acts->x, B, C, H, W, block_size);
    l_info->layer++;
}

void attention_layer_forward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int B, int C, int H, int W, int HS, int gn_n_groups, int block_size
) {
    int layer = l_info->layer;
    AttentionParams* att_params = (AttentionParams*)unet->params.layer_ptrs[layer];
    AttentionActs* att_acts = (AttentionActs*)unet->acts.layer_ptrs[layer];
    attention_block_forward(cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, att_params, att_acts);
    l_info->layer++;
}

void concat_layer_forward(
    Unet* unet,
    LayerInfo* l_info,
    int B, int C1, int C2, int H, int W, int block_size
) {
    int layer = l_info->layer;
    ConcatChannelActs* cat_acts = (ConcatChannelActs*)unet->acts.layer_ptrs[layer];
    concat_channel_forward(cat_acts->x1, cat_acts->x2, cat_acts->out, B, C1, C2, H, W, block_size);
    l_info->layer++;
}

void upsample_layer_forward(
    Unet* unet, LayerInfo* l_info,
    int B, int C, int H, int W, int block_size
) {
    int layer = l_info->layer;
    UpsampleActs* up_acts = (UpsampleActs*)unet->acts.layer_ptrs[layer];
    upsample_forward1(up_acts->out, up_acts->x, B, C, H, W, block_size);
    l_info->layer++;
}

void groupnorm_layer_forward(
    Unet* unet, LayerInfo* l_info,
    int B, int C, int H, int W, int gn_n_groups
) {
    int layer = l_info->layer;
    GroupNormParams* gn_p = (GroupNormParams*)unet->params.layer_ptrs[layer];
    GroupNormActs* gn_a = (GroupNormActs*)unet->acts.layer_ptrs[layer];
    groupnorm_forward(gn_a->x, gn_p->w, gn_p->b, gn_a->out, gn_a->mean, gn_a->rstd, B, C, H, W, gn_n_groups);
    l_info->layer++;
}

void unet_forward(
    cublasHandle_t cublas_handle,
    Unet* unet
) {
    // common constants
    UnetConfig config = unet->config;
    int B = config.B;
    int C_in = config.C_in;
    int C_model = config.C_model;
    int C_time = config.C_time;
    int C_out = config.C_out;
    int H = config.H;
    int W = config.W;
    int block_size = config.block_size;
    int gn_n_groups = config.gn_n_groups;
    int HS = config.HS;
    int n_levels = config.n_levels;
    int n_res = config.n_res;
    int att_start_level = config.att_start_level;


    LayerInfo l_info;
    l_info.layer = 0;
    linear_layer_forward(unet, &l_info, cublas_handle, B, C_model, C_time, block_size);
    silu_layer_forward(unet, &l_info, B * C_time, block_size);
    linear_layer_forward(unet, &l_info, cublas_handle, B, C_time, C_time, block_size);
    int OC = C_model;
    convk3_layer_forward(unet, &l_info, cublas_handle, B, C_in, OC, H, W, block_size);

    int C = OC;
    for (int l = 0; l < n_levels; l++) {
        int mult = config.channel_mult[l];
        OC = C_model * mult;
        for (int r = 0; r < n_res; r++) {
            resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, OC, B, H, W, gn_n_groups, block_size);
            C = OC;
            if (l >= att_start_level) {
                attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
            }
        }
        if (l < n_levels - 1) {
            avgpool_layer_forward(unet, &l_info, B, C, H, W, block_size);
            H /= 2;
            W /= 2;
        }
    }

    resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
    resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);

    // upward
    for (int l = n_levels - 1; l >= 0; l--) {
        int C_skip = C_model * config.channel_mult[l];
        for (int r = 0; r <= n_res; r++) {
            if (r == n_res && l > 0) {
                C_skip = C_model * config.channel_mult[l - 1];
            }
            concat_layer_forward(unet, &l_info, B, C, C_skip, H, W, block_size);
            OC = C_model * config.channel_mult[l];
            resblock_layer_forward(unet, &l_info, cublas_handle, C + C_skip, C_time, OC, B, H, W, gn_n_groups, block_size);
            C = OC;
            if (l >= att_start_level) {
                attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
            }
        }
        if (l > 0) {
            upsample_layer_forward(unet, &l_info, B, C, H, W, block_size);
            H *= 2;
            W *= 2;
        }
    }

    groupnorm_layer_forward(unet, &l_info, B, C, H, W, gn_n_groups);
    silu_layer_forward(unet, &l_info, B * C * H * W, block_size);
    convk3_layer_forward(unet, &l_info, cublas_handle, B, C, C_out, H, W, block_size);

    ConvK3Acts* last_acts = (ConvK3Acts*)unet->acts.layer_ptrs[l_info.layer - 1];
    unet->output = last_acts->out;
    if (unet->target == nullptr) {
        // don't have targets, not computing loss
        unet->mean_loss = -1.0f;
    } else {
        mse_forward(unet->output, unet->target, unet->gpu_loss, B * C_out * H * W, block_size);
        cudaCheck(cudaMemcpy(&(unet->mean_loss), unet->gpu_loss, sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ------------------------------------------------------------------------------------------------
// Unet backward

void swap_layer_in_out_ptrs(LayerInfo* l_info) {
    float* temp = l_info->input;
    l_info->input = l_info->prev_out;
    l_info->prev_out = temp;
}

void linear_layer_backward(
    Unet* unet,
    LayerInfo* l_info,
    cublasHandle_t cublas_handle, int N, int C, int OC
) {
    int layer = l_info->layer;
    LinearParams* linear_params = (LinearParams*)unet->params.layer_ptrs[layer];
    LinearParams* linear_grads = (LinearParams*)unet->grads.layer_ptrs[layer];
    LinearActs* linear_acts = (LinearActs*)unet->acts.layer_ptrs[layer];
    matmul_backward1(cublas_handle, l_info->prev_out, linear_grads->w, linear_grads->b, l_info->input, linear_acts->inp, linear_params->w, N, C, OC);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void silu_layer_backward(
    Unet* unet, LayerInfo* l_info,
    int N, int block_size
) {
    SiluActs* acts = (SiluActs*)unet->acts.layer_ptrs[l_info->layer];
    silu_backward(l_info->input, acts->x, l_info->prev_out, N, block_size);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void convk3_layer_backward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int B, int C, int OC, int H, int W, int block_size,
    int add_skip = 0
) {
    if (add_skip) {
        // need to add a skip connection component to dout, only used in the input downward layers
        float* skip_dinp = unet->back.back_acts.skip_dinps[l_info->skip_idx];
        add_inplace_forward(skip_dinp, l_info->input, B * OC * H * W, block_size);
        l_info->skip_idx--;
    }
    int layer = l_info->layer;
    ConvK3Params* conv_params = (ConvK3Params*)unet->params.layer_ptrs[layer];
    ConvK3Params* conv_grads = (ConvK3Params*)unet->grads.layer_ptrs[layer];
    ConvK3Acts* conv_acts = (ConvK3Acts*)unet->acts.layer_ptrs[layer];
    UnetBackActs* unet_back_acts = &(unet->back.back_acts);
    conv2d_k3_backward2(l_info->input, conv_acts->inp, conv_params->w, unet_back_acts->buf1_BCHW, unet_back_acts->buf2_BCHW, l_info->prev_out, conv_grads->w, conv_grads->b, B, C, OC, H, W);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void resblock_layer_backward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int C_in, int C_emb, int C_out, int B, int H, int W, int gn_n_groups, int block_size,
    int add_skip = 0
) {
    if (add_skip) {
        // need to add a skip connection component to dout, only used in the input downward layers
        float* skip_dinp = unet->back.back_acts.skip_dinps[l_info->skip_idx];
        add_inplace_forward(skip_dinp, l_info->input, B * C_out * H * W, block_size);
        l_info->skip_idx--;
    }
    int layer = l_info->layer;
    ResBlockParameters* resblock_params = (ResBlockParameters*)unet->params.layer_ptrs[layer];
    ResBlockParameters* resblock_grads = (ResBlockParameters*)unet->grads.layer_ptrs[layer];
    ResBlockActivations* resblock_acts = (ResBlockActivations*)unet->acts.layer_ptrs[layer];
    ResBlockBackwardActivations back_acts;
    UnetBackActs* unet_back_acts = &(unet->back.back_acts);

    // point back_acts to buffers
    back_acts.buf_BCemb = unet_back_acts->buf1_BCHW;
    back_acts.buf_BCHoWo = unet_back_acts->buf2_BCHW;
    back_acts.buf1_BCHW = unet_back_acts->buf3_BCHW;
    back_acts.buf2_BCHW = unet_back_acts->buf4_BCHW;
    back_acts.dweight_buf = unet_back_acts->buf1_B3CHW;
    back_acts.dbias_buf = unet_back_acts->buf2_B3CHW;
    back_acts.dout = l_info->input;
    back_acts.dx = l_info->prev_out;
    back_acts.demb = l_info->emb;

    resblock_backward(cublas_handle, C_in, C_emb, C_out, B, H, W, block_size, 0, 0, gn_n_groups, resblock_params, resblock_grads, resblock_acts, &back_acts);
    swap_layer_in_out_ptrs(l_info);

    // add demb to cumulative demb
    add_inplace_forward(l_info->emb, l_info->emb_cum, B * C_emb, block_size);
    l_info->layer--;
}

void attention_layer_backward(
    Unet* unet, LayerInfo* l_info,
    cublasHandle_t cublas_handle, int B, int C, int H, int W, int HS, int gn_n_groups, int block_size,
    int add_skip = 0
) {
    if (add_skip) {
        // need to add a skip connection component to dout, only used in the input downward layers
        float* skip_dinp = unet->back.back_acts.skip_dinps[l_info->skip_idx];
        //printf("att adding skip %d\n", l_info->skip_idx);
        add_inplace_forward(skip_dinp, l_info->input, B * C * H * W, block_size);
        l_info->skip_idx--;
    }
    int layer = l_info->layer;
    AttentionParams* att_params = (AttentionParams*)unet->params.layer_ptrs[layer];
    AttentionParams* att_grads = (AttentionParams*)unet->grads.layer_ptrs[layer];
    AttentionActs* att_acts = (AttentionActs*)unet->acts.layer_ptrs[layer];
    AttentionBackwardActs att_backs;
    UnetBackActs* unet_backs = &(unet->back.back_acts);
    att_backs.buf1_BCHW = unet_backs->buf1_BCHW;
    att_backs.buf2_BCHW = unet_backs->buf2_BCHW;
    att_backs.buf_B3CHW = unet_backs->buf1_B3CHW;
    att_backs.dqkvr = unet_backs->buf2_B3CHW;
    att_backs.dpreatt = unet_backs->buf3_BCHW;
    att_backs.datt = unet_backs->buf4_BCHW;
    att_backs.dout = l_info->input;
    att_backs.dinp = l_info->prev_out;

    attention_block_backward(cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, att_params, att_acts, &att_backs, att_grads);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void concat_layer_backward(
    Unet* unet, LayerInfo* l_info,
    int B, int C1, int C2, int H, int W, int block_size
) {
    float* skip_dinp = unet->back.back_acts.skip_dinps[l_info->skip_idx];
    //printf("concat skip %d\n", l_info->skip_idx);
    concat_channel_backward(l_info->input, l_info->prev_out, skip_dinp, B, C1, C2, H, W, block_size);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
    l_info->skip_idx++;
}


void avgpool_layer_backward(
    Unet* unet, LayerInfo* l_info,
    int B, int C, int H, int W, int block_size,
    int add_skip = 0
) {
    if (add_skip) {
        // need to add a skip connection component to dout, only used in the input downward layers
        float* skip_dinp = unet->back.back_acts.skip_dinps[l_info->skip_idx];
        //printf("avgpool adding skip %d\n", l_info->skip_idx);
        add_inplace_forward(skip_dinp, l_info->input, B * C * H/2 * W/2, block_size);
        l_info->skip_idx--;
    }
    avgpool_2d_backward1(l_info->input, l_info->prev_out, B, C, H, W, block_size);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void upsample_layer_backward(
    Unet* unet, LayerInfo *l_info,
    int B, int C,  int H, int W, int block_size
) {
    upsample_backward1(l_info->prev_out, l_info->input, B, C, H, W, block_size);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void groupnorm_layer_backward(
    Unet* unet, LayerInfo* l_info,
    int B, int C, int H, int W, int gn_n_groups
) {
    int layer = l_info->layer;
    GroupNormParams* gn_p = (GroupNormParams*)unet->params.layer_ptrs[layer];
    GroupNormParams* gn_g = (GroupNormParams*)unet->grads.layer_ptrs[layer];
    GroupNormActs* gn_a = (GroupNormActs*)unet->acts.layer_ptrs[layer];
    groupnorm_backward(l_info->input, gn_a->x, gn_a->mean, gn_a->rstd, gn_p->w, l_info->prev_out, gn_g->w, gn_g->b, B, C, H, W, gn_n_groups);
    swap_layer_in_out_ptrs(l_info);
    l_info->layer--;
}

void unet_backward(
    cublasHandle_t cublas_handle,
    Unet* unet
   
) {
    if (unet->mean_loss == -1.0f or unet->target == nullptr) {
        printf("Error: must run forward with targets before calling backward\n");
        exit(EXIT_FAILURE);
    }
    // common constants
    UnetConfig config = unet->config;
    int B = config.B;
    int C_in = config.C_in;
    int C_model = config.C_model;
    int C_time = config.C_time;
    int C_out = config.C_out;
    int H = config.H;
    int W = config.W;
    int block_size = config.block_size;
    int gn_n_groups = config.gn_n_groups;
    int HS = config.HS;
    int n_levels = config.n_levels;
    int n_res = config.n_res;
    int att_start_level = config.att_start_level;

    UnetBackActs* back_acts = &(unet->back.back_acts);
    mse_backward(unet->output, unet->target, back_acts->dinp_buf1, B * C_out * H * W, block_size);

    LayerInfo l_info;
    l_info.layer = config.n_layers - 1;
    l_info.input = back_acts->dinp_buf1;
    l_info.prev_out = back_acts->dinp_buf2;
    l_info.emb = back_acts->demb_buf1;
    l_info.emb_cum = back_acts->demb_buf2;
    l_info.skip_idx = 0;

    int C;
    int OC;
    int C_skip;
    C = C_model;
    OC = C_model;
    convk3_layer_backward(unet, &l_info, cublas_handle, B, C, C_out, H, W, block_size);
    silu_layer_backward(unet, &l_info, B * C * H * W, block_size);
    groupnorm_layer_backward(unet, &l_info, B, C, H, W, gn_n_groups);

    for (int l = 0; l < n_levels; l++) {
        for (int r = n_res; r >= 0; r--) {
            if (r == n_res && l > 0) {
                C_skip = C_model * config.channel_mult[l - 1];
            } else {
                C_skip = C_model * config.channel_mult[l];
            }
            if (l >= att_start_level) {
                attention_layer_backward(unet, &l_info, cublas_handle, B, OC, H, W, HS, gn_n_groups, block_size);
            }
            if (r == 0 && l < n_levels - 1) {
                C = C_model * config.channel_mult[l + 1];
            }
            resblock_layer_backward(unet, &l_info, cublas_handle, C + C_skip, C_time, OC, B, H, W, gn_n_groups, block_size);
            concat_layer_backward(unet, &l_info, B, C, C_skip, H, W, block_size);
        }
        if (l < n_levels - 1) {
            H /= 2;
            W /= 2;
            upsample_layer_backward(unet, &l_info, B, C, H, W, block_size);
            OC = C_model * config.channel_mult[l + 1];
        }
    }

    resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    attention_layer_backward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
    resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);

    // we will need to change the backward for all the other blocks below to include the dout contribution from the skip connection.
    l_info.skip_idx = NUM_SKIP_CONNECTIONS - 1;
    for (int l = n_levels - 1; l >= 0; l--) {
        if (l < n_levels - 1) {
            H *= 2;
            W *= 2;
            avgpool_layer_backward(unet, &l_info, B, C, H, W, block_size, 1);
        }
        for (int r = 0; r < n_res; r++) {
            if (l >= att_start_level) {
                attention_layer_backward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, 1);
            }
            int res_add_skip = (l < att_start_level) ? 1 : 0;
            if (r == n_res - 1 && l > 0) {
                int prev_mult = config.channel_mult[l - 1];
                C = prev_mult * C_model;
            }
            resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, OC, B, H, W, gn_n_groups, block_size, res_add_skip);
            OC = C;
        }
    }

    convk3_layer_backward(unet, &l_info, cublas_handle, B, C_in, OC, H, W, block_size, 1);
    unet->dinp = l_info.input;

    // reset dout to point to emb_cum
    l_info.input = l_info.emb_cum;
    linear_layer_backward(unet, &l_info, cublas_handle, B, C_time, C_time);
    silu_layer_backward(unet, &l_info, B * C_time, block_size);
    linear_layer_backward(unet, &l_info, cublas_handle, B, C_model, C_time);
}

// ------------------------------------------------------------------------------------------------
// zero grad and ADAMW

void unet_zero_grad(Unet* unet) {
    cudaCheck(cudaMemset(unet->grads.mem_ptr, 0, unet->grads.total_size * sizeof(float)));
    cudaCheck(cudaMemset(unet->back.back_mem, 0, unet->back.n_backs * sizeof(float)));
    //cudaCheck(cudaMemset(unet->acts.mem_ptr, 0, unet->acts.total_size * sizeof(float)));
}

// below copied from llm.c

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

void unet_update(Unet* unet, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // lazy allocation
    if (unet->m_memory == nullptr) {
        size_t param_size = unet->params.total_size * sizeof(float);
        cudaCheck(cudaMalloc(&(unet->m_memory), param_size));
        cudaCheck(cudaMalloc(&(unet->v_memory), param_size));
        cudaCheck(cudaMemset(unet->m_memory, 0, param_size));
        cudaCheck(cudaMemset(unet->v_memory, 0, param_size));
        printf("allocated %zu MiB for AdamW state m\n", param_size / 1024 / 1024);
        printf("allocated %zu MiB for AdamW state v\n", param_size / 1024 / 1024);
    }

    int block_size = unet->config.block_size;
    int n_blocks = ceil_div(unet->params.total_size, (size_t)block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<n_blocks, block_size>>>(unet->params.mem_ptr, unet->grads.mem_ptr, unet->m_memory, unet->v_memory, unet->params.total_size,
                    learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// save model states

void save_unet_states(Unet* unet, GaussianDiffusion* diffusion, const char* filename) {
    // create header
    int model_header[256];
    model_header[0] = 12345678; // magic number
    model_header[1] = unet->config.B;
    model_header[2] = unet->config.C_in;
    model_header[3] = unet->config.C_model;
    model_header[4] = unet->config.C_out;
    model_header[5] = unet->config.H;
    model_header[6] = unet->config.W;
    model_header[7] = unet->config.max_period;
    model_header[8] = 1; // whether to save the AdamW states, this will be 0 if the model is saved from pytorch
    model_header[9] = 1; // whether to save rng states

    FILE* model_file = fopenCheck(filename, "wb");
    printf("\nsaving model details");
    int result = fwrite(model_header, sizeof(int), 256, model_file);
    assert(result == 256);

    // save params
    float* params_cpu_buf = (float*)mallocCheck(unet->params.total_size * sizeof(float));
    cudaCheck(cudaMemcpy(params_cpu_buf, unet->params.mem_ptr, unet->params.total_size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("saving model params\n");
    result = fwrite(params_cpu_buf, sizeof(float), unet->params.total_size, model_file);
    assert(result == unet->params.total_size);

    if (model_header[8]) {
        // save AdamW states
        cudaCheck(cudaMemcpy(params_cpu_buf, unet->m_memory, unet->params.total_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("saving AdamW state m\n");
        result = fwrite(params_cpu_buf, sizeof(float), unet->params.total_size, model_file);
        assert(result == unet->params.total_size);
        cudaCheck(cudaMemcpy(params_cpu_buf, unet->v_memory, unet->params.total_size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("saving AdamW state v\n");
        result = fwrite(params_cpu_buf, sizeof(float), unet->params.total_size, model_file);
        assert(result == unet->params.total_size);
    }

    if (model_header[9]) {
        // save rngs
        int n_states = diffusion->B * diffusion->img_size;
        void *curand_state_buf = mallocCheck(n_states * sizeof(curandState_t));
        cudaCheck(cudaMemcpy(curand_state_buf, diffusion->curand_states, diffusion->B * diffusion->img_size * sizeof(curandState_t), cudaMemcpyDeviceToHost));
        printf("saving rng states\n");
        result = fwrite(curand_state_buf, sizeof(curandState_t), n_states, model_file);
        assert(result == n_states);
        free(curand_state_buf);
    }

    fcloseCheck(model_file);
    free(params_cpu_buf);
    printf("saved model and diffusion states\n\n");
}

// ------------------------------------------------------------------------------------------------
// load unet and diffusion states

void load_unet_and_diffusion_states(
    Unet* unet, GaussianDiffusion* diffusion, const char* filename
) {
    FILE *model_file = fopenCheck(filename, "rb");
    // read header
    printf("\nloading model header\n");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 12345678) {
        fprintf(stderr, "Invalid magic model file\n");
        exit(1);
    }
    int B = model_header[1];
    int C_in = model_header[2];
    int C_model = model_header[3];
    int C_out = model_header[4];
    int H = model_header[5];
    int W = model_header[6];
    int max_period = model_header[7];
    int save_adamw = model_header[8];
    int save_rng = model_header[9];

    // TODO: cleanup this code
    int* channel_mult = (int*)mallocCheck(4 * sizeof(int));
    channel_mult[0] = 1;
    channel_mult[1] = 2;
    channel_mult[2] = 3;
    channel_mult[3] = 4;

    // setup unet config
    UnetConfig* config = &(unet->config);
    config->B = B;
    config->C_in = C_in;
    config->C_model = C_model;
    config->C_time = C_model * 4;
    config->C_out = C_out;
    config->H = H;
    config->W = W;
    config->block_size = 512;
    config->gn_n_groups = 32;
    config->n_levels = 4;
    config->channel_mult = channel_mult;
    config->HS = 32;
    config->n_res = 2;
    config->max_period = max_period;
    config->att_start_level = 2;

    unet_init(unet);
    unet_count_layers(config);
    unet_make_ptrs_and_count_memory(unet);
    malloc_and_point(unet);

    // read params
    float* params_mem_cpu = (float*)mallocCheck(unet->params.total_size * sizeof(float));
    printf("loading model params\n");
    freadCheck(params_mem_cpu, sizeof(float), unet->params.total_size, model_file);
    cudaCheck(cudaMemcpy(unet->params.mem_ptr, params_mem_cpu, unet->params.total_size * sizeof(float), cudaMemcpyHostToDevice));

    if (save_adamw) {
        // read AdamW states
        printf("loading AdamW states\n");
        freadCheck(params_mem_cpu, sizeof(float), unet->params.total_size, model_file);
        cudaCheck(cudaMalloc(&(unet->m_memory), unet->params.total_size * sizeof(float)));
        cudaCheck(cudaMemcpy(unet->m_memory, params_mem_cpu, unet->params.total_size * sizeof(float), cudaMemcpyHostToDevice));
        freadCheck(params_mem_cpu, sizeof(float), unet->params.total_size, model_file);
        cudaCheck(cudaMalloc(&(unet->v_memory), unet->params.total_size * sizeof(float)));
        cudaCheck(cudaMemcpy(unet->v_memory, params_mem_cpu, unet->params.total_size * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        printf("not loading AdamW states\n");
        unet->m_memory = nullptr;
        unet->v_memory = nullptr;
    }

    if (save_rng) {
        // read curand rng states
        printf("loading rng states\n");
        int n_states = B * C_in * H * W;
        void *curand_state_buf = mallocCheck(n_states * sizeof(curandState_t));
        freadCheck(curand_state_buf, sizeof(curandState_t), n_states, model_file);
        // initialize diffusion curant buffer
        cudaCheck(cudaMalloc(&(diffusion->curand_states), n_states * sizeof(curandState_t)));
        cudaCheck(cudaMemcpy(diffusion->curand_states, curand_state_buf, n_states * sizeof(curandState_t), cudaMemcpyHostToDevice));
        free(curand_state_buf);
    } else {
        printf("not loading rng states\n");
        diffusion->curand_states = nullptr;
    }
    diffusion_init(diffusion, B, C_in, H, W, 1000);

    free(params_mem_cpu);
    fcloseCheck(model_file);
    printf("loaded model and diffusion states \n\n");
}



// ------------------------------------------------------------------------------------------------

typedef struct {
    int count;
    float mean_loss;
} LossCounter;

void update_loss_counter(LossCounter* counter, float loss) {
    float old_mean = counter->mean_loss;
    float old_count = (float)counter->count;
    counter->mean_loss = old_mean * old_count / (old_count + 1) + loss / (old_count + 1);
    counter->count++;
}

void clear_loss_counter(LossCounter* counter) {
    counter->count = 0;
    counter->mean_loss = 0.0f;
}

void create_directory(const char *dir) {
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        mkdir(dir, 0700); // Create directory with permissions 0700
    }
}

int main(int argc, char **argv) {
    // Default values
    const char *model_weights_file = "unet_init.bin";
    const char *data_file = "data/elephant_train.bin";
    const char *log_filename = "log.txt";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model_weights") == 0 && i + 1 < argc) {
            model_weights_file = argv[++i];
        } else if (strcmp(argv[i], "--data_file") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "--log_file") == 0 && i + 1 < argc) {
            log_filename = argv[++i];
        }
    }
    printf("loading model weights from %s\n", model_weights_file);
    printf("loading data from %s\n", data_file);

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

    Unet unet;
    GaussianDiffusion diffusion;
    load_unet_and_diffusion_states(&unet, &diffusion, model_weights_file);

    int B = unet.config.B;
    int C_in = unet.config.C_in;
    int C_out = unet.config.C_out;
    int H = unet.config.H;
    int W = unet.config.W;

    DataLoader loader;
    dataloader_init(&loader, data_file, B);

    // allocate memory for target separately, since unet can still be run without targets for inference
    cudaCheck(cudaMalloc(&(unet.target), B * C_out * H * W * sizeof(float)));
    float *d_timesteps_buf;
    cudaCheck(cudaMalloc(&d_timesteps_buf, B * sizeof(float)));


    int n_iters = 100000;
    int n_log_iter = 100;
    int n_save_iter = 10000;
    //int n_iters = 100;
    //int n_log_iter = 5;
    //int n_save_iter = 50;

    // wipe existing logs
    FILE* log_file = fopen(log_filename, "w");
    fprintf(log_file, "training unet,\nmodel file %s,\ndata filename %s\n", model_weights_file, data_file);
    fprintf(log_file, "starting training for %d iterations\n", n_iters);
    fcloseCheck(log_file);

    const char *model_save_dir = "./models/";
    create_directory(model_save_dir);

    LossCounter loss_counter;
    clear_loss_counter(&loss_counter);

    printf("Training for %d iterations\n", n_iters);
    float time_elapsed = 0.0;
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventRecord(start));
    cudaCheck(cudaEventSynchronize(start));
    for (int iter = 1; iter <= n_iters; iter++) {
        unet_zero_grad(&unet);
        dataloader_next_batch(&loader);
        // copy loaded data into input
        cudaCheck(cudaMemcpy(unet.input, loader.input, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));

        // sample timesteps and get timestep embeddings
        sample_timesteps(&diffusion, d_timesteps_buf);
        get_timestep_embeddings(&unet.time_emb_state, d_timesteps_buf, unet.timestep_emb);
        
        // target will be injected random noise
        diffusion_draw_normal(&diffusion, unet.target);

        // push input through forward diffusion process by t to get x_t
        diffusion_forward_by_t(&diffusion, unet.input, d_timesteps_buf, unet.target);

        unet_forward(cublas_handle, &unet);
        unet_backward(cublas_handle, &unet);
        unet_update(&unet, 0.0001f, 0.9f, 0.999f, 1e-8f, 0.0f, iter);

        update_loss_counter(&loss_counter, unet.mean_loss);

        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end));
        cudaCheck(cudaEventElapsedTime(&time_elapsed, start, end));

        if (iter % n_log_iter == 0) {
            printf("step %4d/%d | loss %7.6f | mean loss %7.6f | cur time %.4f s\n", iter, n_iters, unet.mean_loss, loss_counter.mean_loss, time_elapsed / 1000);
            FILE *log_file = fopen(log_filename, "a");
            fprintf(log_file, "step %4d/%d | loss %7.6f | mean loss %7.6f | cur time %.4f s\n", iter, n_iters, unet.mean_loss, loss_counter.mean_loss, time_elapsed / 1000);
            fclose(log_file);
            clear_loss_counter(&loss_counter);
        }

        if (iter % n_save_iter == 0) {
            char save_filename[256];
            sprintf(save_filename, "./models/model_%d.bin", iter);
            save_unet_states(&unet, &diffusion, save_filename);
        }
    }
    printf("average time per iteration: %f ms\n", time_elapsed / n_iters);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(end));
    cublasCheck(cublasDestroy(cublas_handle));
    cudaCheck(cudaFree(d_timesteps_buf));
    cudaCheck(cudaFree(unet.target));

    dataloader_free(&loader);
    diffusion_free(&diffusion);
    free_unet(&unet);

    printf("success\n");
}