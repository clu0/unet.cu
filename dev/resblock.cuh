#include <cstddef>
#include <cublas_v2.h>
#include <type_traits>
#include "common.h"


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


void set_resblock_params_ptrs(
    int C,
    int C_out,
    ResBlockParameters *params,
    float* params_memory_gpu
);


void set_resblock_acts_ptrs(
    int C,
    int C_out,
    ResBlockActivations *acts,
    float* acts_memory
);


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
);


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
);


void resblock_count_params(
    ResBlockParameters* params,
    int C, int C_emb, int C_out, int B, int H, int W, int up, int down, int gn_n_groups
);

void resblock_count_acts(
    ResBlockActivations* acts,
    int C, int C_emb, int C_out, int B, int H, int W, int up, int down, int gn_n_groups
);