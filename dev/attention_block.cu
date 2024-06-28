#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
#include "attention.cuh"
#include "groupnorm.cuh"
#include "linear.cuh"
#include "add.cuh"
#include "attention_block.cuh"


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

void set_attention_back_acts_pointers(
    AttentionBackwardActs* back_acts,
    float* back_acts_memory,
    size_t* back_act_sizes
) {
    float** ptrs[] = {
        &back_acts->buf1_BCHW,
        &back_acts->buf2_BCHW,
        &back_acts->buf_B3CHW,
        &back_acts->dqkvr,
        &back_acts->dpreatt,
        &back_acts->datt,
        &back_acts->dout,
    };
    for (int i = 0; i < NUM_ATT_BACKWARD_ACTS_TENSORS; i++) {
        *(ptrs[i]) = back_acts_memory;
        back_acts_memory += back_act_sizes[i];
    }
}


void attention_block_count_backs(
    AttentionBackwardActs* back_acts,
    int B, int C, int H, int W, int HS, int gn_n_groups
){
    int NH = C / HS;
    int C3 = 3 * C;
    int T = H * W;

    // count back acts
    size_t *back_act_sizes = back_acts->back_sizes;
    back_act_sizes[0] = B * C * H * W; // buf1_BCHW
    back_act_sizes[1] = B * C * H * W; // buf2_BCHW
    back_act_sizes[2] = B * C3 * H * W; // buf_B3CHW
    back_act_sizes[3] = B * T * C3; // dqkvr
    back_act_sizes[4] = B * NH * T * T; // dpreatt
    back_act_sizes[5] = B * NH * T * T; // datt
    back_act_sizes[6] = B * C * H * W; // dout
    
    size_t num_back_act_params = 0;
    for (int i = 0; i < NUM_ATT_BACKWARD_ACTS_TENSORS; i++) {
        num_back_act_params += back_act_sizes[i];
    }
    back_acts->n_backs = num_back_act_params;
}

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

#ifndef LINKING
int main(int argc, char **argv) {
    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    // read header
    FILE *model_file = fopenCheck("attention_block_params.bin", "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 12345678) {
        fprintf(stderr, "Invalid magic model file, magic number %d\n", model_header[0]);
        exit(1);
    }
    int B = model_header[1];
    int C = model_header[2];
    int H = model_header[3];
    int W = model_header[4];
    int HS = model_header[5];
    int gn_n_groups = model_header[6];

    AttentionParams params, grads, params_cpu;
    attention_block_count_params(&params, C);
    attention_block_count_params(&grads, C);
    attention_block_count_params(&params_cpu, C);

    size_t num_parameters = params.n_params;
    // read params to host
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // copy params to device
    float* params_memory_gpu;
    cudaCheck(cudaMalloc(&params_memory_gpu, num_parameters * sizeof(float)));
    cudaCheck(cudaMemcpy(params_memory_gpu, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));

    // create device memory for grads
    float* grads_memory_gpu;
    cudaCheck(cudaMalloc(&grads_memory_gpu, num_parameters * sizeof(float)));
    cudaCheck(cudaMemset(grads_memory_gpu, 0, num_parameters * sizeof(float)));

    // set param pointers
    set_attention_params_pointers(&params, params_memory_gpu);
    set_attention_params_pointers(&grads, grads_memory_gpu);
    set_attention_params_pointers(&params_cpu, params_memory_cpu);

    // calculate activation sizes
    int C3 = 3 * C;
    int T = H * W;
    AttentionActs acts;
    attention_block_count_acts(&acts, B, C, H, W, gn_n_groups, HS);
    size_t num_acts_params = acts.n_acts;

    // allocate device act memory
    float* acts_memory_gpu;
    cudaCheck(cudaMalloc(&acts_memory_gpu, num_acts_params * sizeof(float)));
    cudaCheck(cudaMemset(acts_memory_gpu, 0, num_acts_params * sizeof(float)));

    // set acts pointers
    set_attention_acts_pointers(&acts, acts_memory_gpu);

    // malloc input, which is not part of the large block of acts memory
    cudaCheck(cudaMalloc(&acts.input, B * C * H * W * sizeof(float)));

    // create debug states
    size_t debug_state_sizes[NUM_ATT_DEBUG_STATES];
    debug_state_sizes[0] = B * C * H * W; // inp
    debug_state_sizes[1] = B * C * H * W; // gn
    debug_state_sizes[2] = B * C * H * W; // perm1
    debug_state_sizes[3] = B * C3 * H * W; // qkv
    debug_state_sizes[4] = B * C * H * W; // att
    debug_state_sizes[5] = B * C * H * W; // proj
    debug_state_sizes[6] = B * C * H * W; // out
    debug_state_sizes[7] = B * C * H * W; // dout
    debug_state_sizes[8] = B * C * H * W; // dinp

    size_t num_debug_states = 0;
    for (int i = 0; i < NUM_ATT_DEBUG_STATES; i++) {
        num_debug_states += debug_state_sizes[i];
    }

    // allocate host debug state memory and host grads memory
    float* debug_states_memory_cpu = (float*)mallocCheck(num_debug_states * sizeof(float));
    float* grads_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    AttentionParams grads_cpu;
    attention_block_count_params(&grads_cpu, C);
    set_attention_params_pointers(&grads_cpu, grads_memory_cpu);
    FILE *debug_states_file = fopenCheck("attention_block_states.bin", "rb");
    printf("num debug states: %ld\n", num_debug_states);
    freadCheck(debug_states_memory_cpu, sizeof(float), num_debug_states, debug_states_file);
    freadCheck(grads_memory_cpu, sizeof(float), num_parameters, debug_states_file);
    fcloseCheck(debug_states_file);

    // set debug states pointers
    AttentionDebugStates debug_states;
    float* debug_states_ptr = debug_states_memory_cpu;
    float** debug_states_ptrs[] = {
        &debug_states.inp,
        &debug_states.gn,
        &debug_states.perm1,
        &debug_states.qkv,
        &debug_states.att,
        &debug_states.proj,
        &debug_states.out,
        &debug_states.dout,
        &debug_states.dinp
    };
    for (int i = 0; i < NUM_ATT_DEBUG_STATES; i++) {
        *(debug_states_ptrs[i]) = debug_states_ptr;
        debug_states_ptr += debug_state_sizes[i];
    }

    // create attention backward acts
    AttentionBackwardActs back_acts;
    attention_block_count_backs(&back_acts, B, C, H, W, HS, gn_n_groups);
    size_t num_back_act_params = back_acts.n_backs;

    float* back_acts_memory_gpu;
    cudaCheck(cudaMalloc(&back_acts_memory_gpu, num_back_act_params * sizeof(float)));
    cudaCheck(cudaMemset(back_acts_memory_gpu, 0, num_back_act_params * sizeof(float)));

    // set pointers
    set_attention_back_acts_pointers(&back_acts, back_acts_memory_gpu, back_acts.back_sizes);
    back_acts.dinp = acts.input;

    // copy input and dout to device
    cudaCheck(cudaMemcpy(acts.input, debug_states.inp, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(back_acts.dout, debug_states.dout, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 512;

    printf("Checking forward pass\n");
    attention_block_forward(cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, &params, &acts);

    // validate results
    float acc = 1e-4;
    printf("\nChecking gn\n");
    validate_result(acts.gn, debug_states.gn, "gn", B * C * H * W, acc);
    printf("\nChecking perm1\n");
    validate_result(acts.perm1, debug_states.perm1, "perm1", B * T * C, acc);
    // don't check qkv because it is used as a buffer in attention step so values are wrong
    //printf("\nChecking qkv\n");
    //validate_result(acts.qkv1, debug_states.qkv, "qkv", B * C3 * H * W, acc);
    printf("\nChecking att\n");
    validate_result(acts.att_out, debug_states.att, "att_out", B * T * C, acc);
    printf("\nChecking proj\n");
    validate_result(acts.proj, debug_states.proj, "proj", B * T * C, acc);
    printf("\nChecking out\n");
    validate_result(acts.add, debug_states.out, "out", B * C * H * W, acc);

    printf("\nForward successful\n");

    printf("Checking backward pass\n");
    attention_block_backward(cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, &params, &acts, &back_acts, &grads);

    printf("\nChecking dproj_w\n");
    validate_result(grads.proj_w, grads_cpu.proj_w, "proj_w", C * C, acc);
    printf("\nChecking dproj_b\n");
    validate_result(grads.proj_b, grads_cpu.proj_b, "proj_b", C, acc);
    printf("\nChecking dqkv_w\n");
    validate_result(grads.qkv_w, grads_cpu.qkv_w, "qkv_w", 3 * C * C, acc);
    printf("\nChecking dqkv_b\n");
    validate_result(grads.qkv_b, grads_cpu.qkv_b, "qkv_b", 3 * C, acc);
    printf("\nChecking dgn_w\n");
    validate_result(grads.gn_w, grads_cpu.gn_w, "gn_w", C, acc);
    printf("\nChecking dgn_b\n");
    validate_result(grads.gn_b, grads_cpu.gn_b, "gn_b", C, acc);
    printf("\nChecking dinp\n");
    validate_result(back_acts.dinp, debug_states.dinp, "dinp", B * C * H * W, acc);

    printf("\nAll results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmark\n");
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(
        repeat_times, attention_block_forward,
        cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, &params, &acts
    );
    printf("forward pass | time %.4f ms\n", elapsed_time);

    printf("Backward pass benchmark\n");
    elapsed_time = benchmark_kernel(
        repeat_times, attention_block_backward,
        cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, &params, &acts, &back_acts, &grads
    );
    printf("backward pass | time %.4f ms\n", elapsed_time);


    cublasCheck(cublasDestroy(cublas_handle));
    // free memory
    cudaCheck(cudaFree(params_memory_gpu));
    cudaCheck(cudaFree(acts_memory_gpu));
    cudaCheck(cudaFree(back_acts_memory_gpu));
    cudaCheck(cudaFree(grads_memory_gpu));
    cudaCheck(cudaFree(acts.input));
    free(params_memory_cpu);
    free(debug_states_memory_cpu);
    free(grads_memory_cpu);

}
#endif