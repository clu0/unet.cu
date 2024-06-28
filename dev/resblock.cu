#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"
#include "groupnorm.cuh"
#include "upsample.cuh"
#include "avgpool.cuh"
#include "silu.cuh"
#include "conv2d_k1.cuh"
#include "conv2d_k3.cuh"
#include "linear.cuh"
#include "broadcast.cuh"
#include "add.cuh"
#include "resblock.cuh"



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
    //conv2d_k3_forward2(cublas_handle, acts->ud_h, params->cv3_1_w, params->cv3_1_b, acts->cv3_1, B, C, C_out, H_out, W_out, block_size);
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
    //conv2d_k3_forward2(cublas_handle, acts->silu2, params->cv3_2_w, params->cv3_2_b, acts->cv3_2, B, C_out, C_out, H_out, W_out, block_size);
    conv2d_k3_forward3(acts->silu2, params->cv3_2_w, params->cv3_2_b, acts->cv3_2, B, C_out, C_out, H_out, W_out);
    cudaCheck(cudaGetLastError());

    // add residual (change channels with 1x1 conv if neccessary)
    if (C_out == C) {
        add_forward(acts->cv3_2, acts->ud_x, acts->add2, B * C_out * H_out * W_out, block_size);
        cudaCheck(cudaGetLastError());
    } else {
        //conv2d_k1_forward1(cublas_handle, acts->res_cv1, acts->ud_x, params->res_cv1_w, params->res_cv1_b, B, C, H_out, W_out, C_out, block_size);
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
    //conv2d_k3_backward1(cublas_handle, dout, acts->silu2, params->cv3_2_w, buf1_BCoHoWo, grads->cv3_2_w, grads->cv3_2_b, B, C_out, C_out, H_out, W_out, block_size);
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
    //conv2d_k3_backward1(cublas_handle, buf1_BCoHoWo, acts->ud_h, params->cv3_1_w, buf1_BCHoWo, grads->cv3_1_w, grads->cv3_1_b, B, C, C_out, H_out, W_out, block_size);
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


// ----------------------------------------------------------------------------


void resblock_count_backs(
    ResBlockBackwardActivations* back_acts,
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

    size_t* back_act_sizes = back_acts->back_sizes;
    back_act_sizes[0] = B * C_emb; // buf_BCemb
    back_act_sizes[1] = B * C * H_out * W_out; // buf_BCHoWo
    back_act_sizes[2] = B * C * H * W; // buf1_BCHW
    back_act_sizes[3] = B * C * H * W; // buf2_BCHW
    back_act_sizes[4] = B * C_out * H_out * W_out; // dout
    back_act_sizes[5] = C_out * C * 9 * B * 32; // dweight_buf
    back_act_sizes[6] = C_out * B * 32; // dbias_buf
    size_t num_back_acts = 0;
    for (size_t i = 0; i < NUM_RES_BACKWARD_TENSORS; i++) {
        num_back_acts += back_act_sizes[i];
    }

    back_acts->n_backs = num_back_acts;
}

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


#define NUM_DEBUG_STATES 16
typedef struct {
    float* input;
    float* emb;
    float* h_gn1;
    float* h_silu1;
    float* h_1;
    float* x_1;
    float* emb_1;
    float* h_plus_emb;
    float* h_gn2;
    float* h_silu2;
    float* h_2;
    float* out;
    float* dout;
    float* dx;
    float* demb;
    float* emb_broad;
    float* h_ud;
} DebugStates;


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


#ifndef LINKING
int main(int argc, char **argv) {
    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    // read header
    FILE *model_file = fopenCheck("resblock_params.bin", "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 12345678) {
        fprintf(stderr, "Invalid magic model file\n");
        exit(1);
    }
    int B = model_header[1];
    int C = model_header[2];
    int C_emb = model_header[3];
    int C_out = model_header[4];
    int H = model_header[5];
    int W = model_header[6];
    int up = model_header[7];
    int down = model_header[8];
    int gn_n_groups = model_header[9];
    int H_out = H;
    int W_out = W;
    if (up) {
        H_out *= 2;
        W_out *= 2;
    } else if (down) {
        H_out /= 2;
        W_out /= 2;
    }
    // not setting to 1024 because sometimes we run out of resources
    int block_size = 512;
    
    ResBlockParameters params;
    resblock_count_params(&params, C, C_emb, C_out, B, H, W, up, down, gn_n_groups);
    ResBlockParameters grads;
    resblock_count_params(&grads, C, C_emb, C_out, B, H, W, up, down, gn_n_groups);
    ResBlockActivations acts;
    resblock_count_acts(&acts, C, C_emb, C_out, B, H, W, up, down, gn_n_groups);
    ResBlockBackwardActivations back_acts;
    resblock_count_backs(&back_acts, C, C_emb, C_out, B, H, W, up, down, gn_n_groups);

    size_t num_parameters = params.n_params;
    // allocate host memory 
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // allocate device memory
    float* params_memory_gpu;
    cudaCheck(cudaMalloc(&params_memory_gpu, num_parameters * sizeof(float)));
    cudaCheck(cudaMemcpy(params_memory_gpu, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    printf("Params: total param size: %.3f MB\n", (float)num_parameters * sizeof(float) / 1024 / 1024);

    float* grads_memory_gpu;
    cudaCheck(cudaMalloc(&grads_memory_gpu, num_parameters * sizeof(float)));
    cudaCheck(cudaMemset(grads_memory_gpu, 0, num_parameters * sizeof(float)));

    // set param pointers
    set_resblock_params_ptrs(C, C_out, &params, params_memory_gpu);
    set_resblock_params_ptrs(C, C_out, &grads, grads_memory_gpu);

    // open states file and read input and activations
    FILE *states_file = fopenCheck("resblock_states.bin", "rb");
    int num_debug_states = NUM_DEBUG_STATES;
    num_debug_states += (up || down) ? 1 : 0;
    size_t state_sizes[num_debug_states];
    state_sizes[0] = B * C * H * W; // input
    state_sizes[1] = B * C_emb; // emb
    state_sizes[2] = B * C * H * W; // h_gn1
    state_sizes[3] = B * C * H * W; // h_silu1
    state_sizes[4] = B * C_out * H_out * W_out; // h_1
    state_sizes[5] = B * C * H_out * W_out; // x_1, channel is still C because we only apply the skip connection at the end
    state_sizes[6] = B * C_out; // emb_1
    state_sizes[7] = B * C_out * H_out * W_out; // h_plus_emb
    state_sizes[8] = B * C_out * H_out * W_out; // h_gn2
    state_sizes[9] = B * C_out * H_out * W_out; // h_silu2
    state_sizes[10] = B * C_out * H_out * W_out; // h_2
    state_sizes[11] = B * C_out * H_out * W_out; // out
    state_sizes[12] = B * C_out * H_out * W_out; // dout
    state_sizes[13] = B * C * H * W; // dx
    state_sizes[14] = B * C_emb; // demb
    state_sizes[15] = B * C_out * H_out * W_out; // emb_broad
    if (up || down) {
        state_sizes[16] = B * C * H_out * W_out; // h_ud
    }

    size_t num_debug_state_params = 0;
    for (size_t i = 0; i < num_debug_states; i++) {
        num_debug_state_params += state_sizes[i];
    }

    printf("num debug params: %ld\n", num_debug_state_params);
    float* debug_states_cpu = (float*)mallocCheck(num_debug_state_params * sizeof(float));
    float* grads_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    printf("H_out %d, W_out %d\n", H_out, W_out);
    freadCheck(debug_states_cpu, sizeof(float), num_debug_state_params, states_file);
    freadCheck(grads_memory_cpu, sizeof(float), num_parameters, states_file);
    fcloseCheck(states_file);

    DebugStates debug_states;

    float** debug_ptrs[] = {
        &debug_states.input, &debug_states.emb, &debug_states.h_gn1, &debug_states.h_silu1,
        &debug_states.h_1, &debug_states.x_1, &debug_states.emb_1, &debug_states.h_plus_emb,
        &debug_states.h_gn2, &debug_states.h_silu2, &debug_states.h_2, &debug_states.out,
        &debug_states.dout, &debug_states.dx, &debug_states.demb, &debug_states.emb_broad,
        &debug_states.h_ud
    };

    float* debug_ptr = debug_states_cpu;
    for (int i = 0; i < num_debug_states; i++) {
        *(debug_ptrs[i]) = debug_ptr;
        debug_ptr += state_sizes[i];
    }

    ResBlockParameters grads_cpu;
    resblock_count_params(&grads_cpu, C, C_emb, C_out, B, H, W, up, down, gn_n_groups);
    set_resblock_params_ptrs(C, C_out, &grads_cpu, grads_memory_cpu);

    // allocate device state memory
    size_t num_acts_params = acts.n_acts;
    float* acts_memory_gpu;
    cudaCheck(cudaMalloc(&acts_memory_gpu, num_acts_params * sizeof(float)));
    cudaCheck(cudaMemset(acts_memory_gpu, 0, num_acts_params * sizeof(float)));
    printf("Activations: total param size: %.3f MB\n", (float)num_acts_params * sizeof(float) / 1024 / 1024);
    set_resblock_acts_ptrs(C, C_out, &acts, acts_memory_gpu);

    // create input and emb buffers and copy states
    cudaCheck(cudaMalloc(&acts.input, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&acts.emb, B * C_emb * sizeof(float)));
    // copy the input and emb state to the GPU
    cudaCheck(cudaMemcpy(acts.input, debug_states.input, state_sizes[0] * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(acts.emb, debug_states.emb, state_sizes[1] * sizeof(float), cudaMemcpyHostToDevice));


    // run the forward pass
    printf("Checking forward pass\n");
    resblock_forward(cublas_handle, C, C_emb, C_out, B, H, W, block_size, up, down, gn_n_groups, &params, &acts);

    // validate the results
    float acc = 1e-5;
    printf("\nChecking h_gn1\n");
    validate_result(acts.gn1, debug_states.h_gn1, "h_gn1", B * C * H * W, acc);
    printf("\nChecking h_silu1\n");
    validate_result(acts.silu1, debug_states.h_silu1, "h_silu1", B * C * H * W, acc);
    if (up || down) {
        printf("\nChecking h_ud\n");
        validate_result(acts.ud_h, debug_states.h_ud, "h_ud", B * C * H_out * W_out, acc);
    }
    printf("\nChecking h_1\n");
    validate_result(acts.cv3_1, debug_states.h_1, "h_1", B * C_out * H_out * W_out, acc);
    printf("\nChecking x_1\n");
    validate_result(acts.ud_x, debug_states.x_1, "x_1", B * C * H_out * W_out, acc);
    printf("\nChecking emb_1\n");
    validate_result(acts.l_emb, debug_states.emb_1, "emb_1", B * C_out, acc);
    printf("\nChecking emb_broad\n");
    validate_result(acts.broad_emb, debug_states.emb_broad, "emb_broad", B * C_out * H_out * W_out, acc);
    printf("\nChecking h_plus_emb\n");
    validate_result(acts.add1, debug_states.h_plus_emb, "h_plus_emb", B * C_out * H_out * W_out, acc);
    printf("\nChecking h_gn2\n");
    validate_result(acts.gn2, debug_states.h_gn2, "h_gn2", B * C_out * H_out * W_out, acc);
    printf("\nChecking h_silu2\n");
    validate_result(acts.silu2, debug_states.h_silu2, "h_silu2", B * C_out * H_out * W_out, acc);
    printf("\nChecking h_2\n");
    validate_result(acts.cv3_2, debug_states.h_2, "h_2", B * C_out * H_out * W_out, acc);
    printf("\nCheckout out\n");
    validate_result(acts.add2, debug_states.out, "out", B * C_out * H_out * W_out, acc);
    printf("Forward pass successful\n");

    // allocate backward pass activation memory on device
    size_t num_back_acts = back_acts.n_backs;
    float* back_acts_memory_gpu;
    cudaCheck(cudaMalloc(&back_acts_memory_gpu, num_back_acts * sizeof(float)));
    cudaCheck(cudaMemset(back_acts_memory_gpu, 0, num_back_acts * sizeof(float)));
    float** back_acts_ptrs[] = {
        &back_acts.buf_BCemb, &back_acts.buf_BCHoWo, &back_acts.buf1_BCHW, &back_acts.buf2_BCHW, &back_acts.dout,
        &back_acts.dweight_buf, &back_acts.dbias_buf
    };
    float* back_acts_mem_ptr = back_acts_memory_gpu;
    for (int i = 0; i < NUM_RES_BACKWARD_TENSORS; i++) {
        *(back_acts_ptrs[i]) = back_acts_mem_ptr;
        back_acts_mem_ptr += back_acts.back_sizes[i];
    }
    back_acts.dx = acts.input;
    back_acts.demb = acts.emb;
    // copy dout to device
    cudaCheck(cudaMemcpy(back_acts.dout, debug_states.dout, B * C_out * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Checking backward pass\n");
    resblock_backward(cublas_handle, C, C_emb, C_out, B, H, W, block_size, up, down, gn_n_groups, &params, &grads, &acts, &back_acts);

    // validate the results
    float back_acc = 1e-4;
    if (C_out != C) {
        printf("\nChecking res_cv1_w\n");
        validate_result(grads.res_cv1_w, grads_cpu.res_cv1_w, "res_cv1_w", C_out * C, back_acc);
        printf("\nChecking res_cv1_b\n");
        validate_result(grads.res_cv1_b, grads_cpu.res_cv1_b, "res_cv1_b", C_out, back_acc);
    }

    printf("\nChecking grad cv3_2_w\n");
    validate_result(grads.cv3_2_w, grads_cpu.cv3_2_w, "cv3_2_w", C_out * C_out * 9, back_acc);
    printf("\nChecking grad cv3_2_b\n");
    validate_result(grads.cv3_2_b, grads_cpu.cv3_2_b, "cv3_2_b", C_out, back_acc);

    printf("\nChecking grad gn2_w\n");
    validate_result(grads.gn2_w, grads_cpu.gn2_w, "gn2_w", C_out, back_acc);
    printf("\nChecking grad gn2_b\n");
    validate_result(grads.gn2_b, grads_cpu.gn2_b, "gn2_b", C_out, back_acc);

    printf("\nChecking grad l_emb_w\n");
    validate_result(grads.l_emb_w, grads_cpu.l_emb_w, "l_emb_w", C_out * C_emb, back_acc);
    printf("\nChecking grad l_emb_b\n");
    validate_result(grads.l_emb_b, grads_cpu.l_emb_b, "l_emb_b", C_out, back_acc);

    printf("\nChecking grad cv3_1_w\n");
    validate_result(grads.cv3_1_w, grads_cpu.cv3_1_w, "cv3_1_w", C_out * C * 9, back_acc);
    printf("\nChecking grad cv3_1_b\n");
    validate_result(grads.cv3_1_b, grads_cpu.cv3_1_b, "cv3_1_b", C_out, back_acc);

    printf("\nChecking grad gn1_b\n");
    validate_result(grads.gn1_b, grads_cpu.gn1_b, "gn1_b", C, back_acc);
    printf("\nChecking grad gn1_w\n");
    validate_result(grads.gn1_w, grads_cpu.gn1_w, "gn1_w", C, back_acc);

    printf("\nChecking demb\n");
    validate_result(back_acts.demb, debug_states.demb, "demb", B * C_emb, back_acc);
    printf("\nChecking dx\n");
    validate_result(back_acts.dx, debug_states.dx, "dx", B * C * H * W, back_acc);

    printf("\nAll results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmark\n");
    int repeat_times = 100;

    float elapsed_time = benchmark_kernel(
        repeat_times, resblock_forward,
        cublas_handle, C, C_emb, C_out, B, H, W, block_size, up, down, gn_n_groups, &params, &acts
    );
    printf("forward pass | time %.4f ms\n", elapsed_time);

    printf("Backward pass benchmark\n");
    elapsed_time = benchmark_kernel(
        repeat_times, resblock_backward,
        cublas_handle, C, C_emb, C_out, B, H, W, block_size, up, down, gn_n_groups, &params, &grads, &acts, &back_acts
    );
    printf("backward pass | time %.4f ms\n", elapsed_time);



    cublasCheck(cublasDestroy(cublas_handle));
    // free memory
    free(params_memory_cpu);
    free(debug_states_cpu);
    free(grads_memory_cpu);
    cudaCheck(cudaFree(params_memory_gpu));
    cudaCheck(cudaFree(acts_memory_gpu));
    cudaCheck(cudaFree(grads_memory_gpu));
    cudaCheck(cudaFree(back_acts_memory_gpu));
    // free input and emb
    cudaCheck(cudaFree(acts.input));
    cudaCheck(cudaFree(acts.emb));
    return 0;
}
#endif