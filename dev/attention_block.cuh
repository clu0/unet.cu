#include <cublas_v2.h>

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


#define NUM_ATT_DEBUG_STATES 9
typedef struct {
    float* inp; // B, C, H, W
    float* gn; // B, C, H, W
    float* perm1; // B, HxW, C
    float* qkv; // B, 3xC, H, W
    float* att; // B, NH, T, T
    float* proj; // B, C, H, W
    float* out; // B, C, H, W
    float* dout; // B, C, H, W
    float* dinp; // B, C, H, W
} AttentionDebugStates;


typedef struct {
    int C;
    int H;
    int W;
    int HS;
    int B;
    int gn_n_groups;
    size_t param_sizes[NUM_ATT_PARAM_TENSORS];
    size_t act_sizes[NUM_ATT_ACT_TENSORS];
    size_t back_sizes[NUM_ATT_BACKWARD_ACTS_TENSORS];
    size_t n_params;
    size_t n_acts;
    size_t n_backs;
} AttentionConfig;


void attention_block_count_params(AttentionParams* params, int C);


void attention_block_count_acts(
    AttentionActs* acts,
    int B, int C, int H, int W, int gn_n_groups, int HS
);


void attention_block_forward
(
    cublasHandle_t cublas_handle,
    int B, int C, int H, int W, int HS, int gn_n_groups, int block_size,
    AttentionParams* params, AttentionActs* acts
);


void attention_block_backward
(
    cublasHandle_t cublas_handle,
    int B, int C, int H, int W, int HS, int gn_n_groups, int block_size,
    AttentionParams* params, AttentionActs* acts,
    AttentionBackwardActs* back_acts, AttentionParams* grads
);


void set_attention_params_pointers(
    AttentionParams* params,
    float* params_memory
);

void set_attention_acts_pointers(
    AttentionActs* acts,
    float* acts_memory
);