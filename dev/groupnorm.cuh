#pragma once

void groupnorm_forward(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int H, int W, int n_groups
);

void groupnorm_backward(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int H, int W, int n_groups
);

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