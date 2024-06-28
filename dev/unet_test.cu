#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <unistd.h> // For POSIX systems to use getcwd
#include <limits.h> // For PATH_MAX
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.h"
#include "linear.cuh"
#include "silu.cuh"
#include "conv2d_k3.cuh"
#include "resblock.cuh"
#include "add.cuh"
#include "avgpool.cuh"
#include "attention_block.cuh"
#include "concat_channel.cuh"
#include "add.cuh"
#include "upsample.cuh"
#include "groupnorm.cuh"
#include "mse.cuh"
#include "timestep_embedding.cuh"
#include "rand.h"
#include <chrono>



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

// ------------------------------------------------------------------------------------------------
// DataLoader

const char* DATASET_FILE = "../datasets/elephant_train.bin";
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
    mt19937_state host_rng;
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
    int total_states = B * C_in * H * W;
    int n_blocks = ceil_div(total_states, block_size);
    cudaCheck(cudaMalloc(&diffusion->curand_states, total_states * sizeof(curandState)));
    init_curand_states<<<n_blocks, block_size>>>(diffusion->curand_states, seed, total_states);

    // init host rng for drawing random timesteps
    manual_seed(&diffusion->host_rng, seed);
}

// gridDim will be of shape (N / block_size, B)
// expect x and x_t to be of shape (B, N)
// and timesteps to be of shape (B,)
// expect x_t to be filled with standard normals
__global__ void diffusion_sample_kernel(
    float* timesteps, float* x, float* x_t,
    float* sqrt_alphas_cumprod, float* sqrt_one_minus_alphas_cumprod,
    int B, int N
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && b < B) {
        int t = (int)timesteps[b];
        float sqrt_alpha_t = sqrt_alphas_cumprod[t];
        float sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t];
        float x_t_val = x_t[b * N + n]; // x_t starts out as standard normals
        float x_val = x[b * N + n];
        x_t[b * N + n] = x_val * sqrt_alpha_t + x_t_val * sqrt_one_minus_alpha_t;
    }
}

void diffusion_sample(
    float* timesteps, float* x, float* x_t,
    GaussianDiffusion *diffusion,
    int B, int N
) {
    dim3 gridDim(ceil_div(N, diffusion->block_size), B);
    diffusion_sample_kernel<<<gridDim, diffusion->block_size>>>(timesteps, x, x_t, diffusion->sqrt_alphas_cumprod, diffusion->sqrt_one_minus_alphas_cumprod, B, N);
    cudaCheck(cudaDeviceSynchronize());
}

// a kernel that creates x_t = sqrt_alpha * x + sqrt(1 - alpha) * noise in one go
// that draws the noise from standard normals, gets alpha from the timesteps and the precomputed alpha cumprod
// and saves output to x_t
// if the input x, timesteps and the alpha buffers are null, then it will just fill x_t with standard normals
// to be called with B * N total threads
__global__ void diffusion_draw_and_sample_xt_kernel(
    float* x_t, int B, int N, curandState* curand_states,
    float* sqrt_alphas_cumprod, float* sqrt_one_minus_alphas_cumprod,
    float* timesteps = nullptr, float* x = nullptr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        float out_val = curand_normal(&curand_states[idx]);
        if (timesteps != nullptr && x != nullptr) {
            int b = idx / N;
            int n = idx % N;
            float x_val = x[b * N + n];
            int t = (int)timesteps[b];
            float sqrt_alpha_t = sqrt_alphas_cumprod[t];
            float sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t];
            out_val = x_val * sqrt_alpha_t + out_val * sqrt_one_minus_alpha_t;
        }
        x_t[idx] = out_val;
    }
}

void diffusion_draw_and_sample_xt(
    GaussianDiffusion *diffusion,
    float* x_t,
    float* timesteps = nullptr, float* x = nullptr
) {
    int total_states = diffusion->B * diffusion->img_size;
    int n_blocks = ceil_div(total_states, diffusion->block_size);
    diffusion_draw_and_sample_xt_kernel<<<n_blocks, diffusion->block_size>>>(
        x_t, diffusion->B, diffusion->img_size, diffusion->curand_states, diffusion->sqrt_alphas_cumprod, diffusion->sqrt_one_minus_alphas_cumprod, timesteps, x
    );
}

void diffusion_free(GaussianDiffusion* diffusion) {
    cudaCheck(cudaFree(diffusion->betas));
    cudaCheck(cudaFree(diffusion->sqrt_alphas_cumprod));
    cudaCheck(cudaFree(diffusion->sqrt_one_minus_alphas_cumprod));
    cudaCheck(cudaFree(diffusion->curand_states));
}

// samples B timesteps in range [0, max_period), convert them to floats and
// store in timesteps
void sample_timesteps(GaussianDiffusion *diffusion, float* timesteps, int N = -1) {
    mt19937_state* rng = &diffusion->host_rng;
    if (N < 0) {
        N = diffusion->B;
    }
    int max_period = diffusion->max_period;
    for (int i = 0; i < N; i++) {
        float rand_float = randfloat32(rng);
        timesteps[i] = (float)(int)(rand_float * max_period);
    }
}

// ------------------------------------------------------------------------------------------------
// Unet init

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
    Unet* unet,
    float* t_res = nullptr, float* t_att = nullptr
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

    cudaEvent_t start, end;
    if (t_res != nullptr) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
    }

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
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(start));
            }
            resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, OC, B, H, W, gn_n_groups, block_size);
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(end));
                cudaCheck(cudaEventSynchronize(start));
                cudaCheck(cudaEventSynchronize(end));
                float elapsed_time;
                cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                *t_res += elapsed_time;
                printf("down resblock: l %d, r %d, Cin %d, Cout %d, H %d, W %d, time %f\n", l, r, C, OC, H, W, elapsed_time);
            }
            C = OC;
            if (l >= att_start_level) {
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(start));
                }
                attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(end));
                    cudaCheck(cudaEventSynchronize(start));
                    cudaCheck(cudaEventSynchronize(end));
                    float elapsed_time;
                    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                    *t_att += elapsed_time;
                    printf("down att: l %d, r %d, C %d, H %d, W %d, time %f\n", l, r, C, H, W, elapsed_time);
                }
            }
        }
        if (l < n_levels - 1) {
            avgpool_layer_forward(unet, &l_info, B, C, H, W, block_size);
            H /= 2;
            W /= 2;
        }
    }

    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(start));
    }
    resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        *t_res += elapsed_time;
        printf("middle resblock 1: Cin %d, Cout %d, H %d, W %d, time %f\n", C, OC, H, W, elapsed_time);
        cudaCheck(cudaEventRecord(start));
    }
    attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        *t_att += elapsed_time;
        printf("middle att: Cin %d, H %d, W %d, time %f\n", C, H, W, elapsed_time);
        cudaCheck(cudaEventRecord(start));
    }
    resblock_layer_forward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        printf("middle resblock 2: Cin %d, Cout %d, H %d, W %d, time %f\n", C, OC, H, W, elapsed_time);
        *t_res += elapsed_time;
    }

    // upward
    for (int l = n_levels - 1; l >= 0; l--) {
        int C_skip = C_model * config.channel_mult[l];
        for (int r = 0; r <= n_res; r++) {
            if (r == n_res && l > 0) {
                C_skip = C_model * config.channel_mult[l - 1];
            }
            concat_layer_forward(unet, &l_info, B, C, C_skip, H, W, block_size);
            OC = C_model * config.channel_mult[l];
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(start));
            }
            resblock_layer_forward(unet, &l_info, cublas_handle, C + C_skip, C_time, OC, B, H, W, gn_n_groups, block_size);
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(end));
                cudaCheck(cudaEventSynchronize(start));
                cudaCheck(cudaEventSynchronize(end));
                float elapsed_time;
                cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                printf("up resblock: l %d, r %d, Cin %d, Cskip %d, Cout %d, H %d, W %d, time %f\n", l, r, C, C_skip, OC, H, W, elapsed_time);
                *t_res += elapsed_time;
            }
            C = OC;
            if (l >= att_start_level) {
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(start));
                }
                attention_layer_forward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(end));
                    cudaCheck(cudaEventSynchronize(start));
                    cudaCheck(cudaEventSynchronize(end));
                    float elapsed_time;
                    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                    printf("up att: l %d, r %d, C %d, H %d, W %d, time %f\n", l, r, C, H, W, elapsed_time);
                    *t_att += elapsed_time;
                }
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
// backward pass
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
    Unet* unet,
    float* t_res = nullptr, float* t_att = nullptr
) {
    if (unet->mean_loss == -1.0f or unet->target == nullptr) {
        printf("Error: must run forward with targets before calling backward\n");
        exit(EXIT_FAILURE);
    }
    cudaEvent_t start, end;
    if (t_res != nullptr) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
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
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(start));
                }
                attention_layer_backward(unet, &l_info, cublas_handle, B, OC, H, W, HS, gn_n_groups, block_size);
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(end));
                    cudaCheck(cudaEventSynchronize(start));
                    cudaCheck(cudaEventSynchronize(end));
                    float elapsed_time;
                    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                    *t_att += elapsed_time;
                    printf("up att back: l %d, r %d, Cout %d, H %d, W %d, time %f\n", l, r, OC, H, W, elapsed_time);
                }
            }
            if (r == 0 && l < n_levels - 1) {
                C = C_model * config.channel_mult[l + 1];
            }
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(start));
            }
            resblock_layer_backward(unet, &l_info, cublas_handle, C + C_skip, C_time, OC, B, H, W, gn_n_groups, block_size);
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(end));
                cudaCheck(cudaEventSynchronize(start));
                cudaCheck(cudaEventSynchronize(end));
                float elapsed_time;
                cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                *t_res += elapsed_time;
                printf("up res back: l %d, r %d, Cin %d, Cskip %d, Cout %d, H %d, W %d, time %f\n", l, r, C, C_skip, OC, H, W, elapsed_time);
            }
            concat_layer_backward(unet, &l_info, B, C, C_skip, H, W, block_size);
        }
        if (l < n_levels - 1) {
            H /= 2;
            W /= 2;
            upsample_layer_backward(unet, &l_info, B, C, H, W, block_size);
            OC = C_model * config.channel_mult[l + 1];
        }
    }

    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(start));
    }
    resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        *t_res += elapsed_time;
        printf("middle resblock 2: Cin %d, Cout %d, H %d, W %d, time %f\n", C, OC, H, W, elapsed_time);
        cudaCheck(cudaEventRecord(start));
    }
    attention_layer_backward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        *t_att += elapsed_time;
        printf("middle att: Cin %d, H %d, W %d, time %f\n", C, H, W, elapsed_time);
        cudaCheck(cudaEventRecord(start));
    }
    resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, C, B, H, W, gn_n_groups, block_size);
    if (t_res != nullptr) {
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(end));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        printf("middle resblock 1: Cin %d, Cout %d, H %d, W %d, time %f\n", C, OC, H, W, elapsed_time);
        *t_res += elapsed_time;
    }

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
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(start));
                }
                attention_layer_backward(unet, &l_info, cublas_handle, B, C, H, W, HS, gn_n_groups, block_size, 1);
                if (t_res != nullptr) {
                    cudaCheck(cudaEventRecord(end));
                    cudaCheck(cudaEventSynchronize(start));
                    cudaCheck(cudaEventSynchronize(end));
                    float elapsed_time;
                    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                    printf("down att back: l %d, r %d, C %d, H %d, W %d, time %f\n", l, r, C, H, W, elapsed_time);
                    *t_att += elapsed_time;
                }
            }
            int res_add_skip = (l < att_start_level) ? 1 : 0;
            if (r == n_res - 1 && l > 0) {
                int prev_mult = config.channel_mult[l - 1];
                C = prev_mult * C_model;
            }
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(start));
            }
            resblock_layer_backward(unet, &l_info, cublas_handle, C, C_time, OC, B, H, W, gn_n_groups, block_size, res_add_skip);
            if (t_res != nullptr) {
                cudaCheck(cudaEventRecord(end));
                cudaCheck(cudaEventSynchronize(start));
                cudaCheck(cudaEventSynchronize(end));
                float elapsed_time;
                cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
                printf("down res back: l %d, r %d, Cin %d, Cout %d, H %d, W %d, time %f\n", l, r, C, OC, H, W, elapsed_time);
                *t_res += elapsed_time;
            }
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
// functions to update gradient

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
// run tests

#define NUM_DEBUG_STATES 5
typedef struct {
    float* input;
    float* timestep_emb;
    float* out;
    float* dinp;
    float* target;
} DebugStates;


int main(int argc, char **argv) {
    // setup cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        //printf("Current working dir: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }

    // read header
    FILE *model_file = fopenCheck("unet_test_params.bin", "rb");
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
    int C_time = C_model * 4;
    const int n_levels = 4;
    int channel_mult[n_levels] = {1, 2, 3, 4};

    // dataloader
    DataLoader loader;
    dataloader_init(&loader, DATASET_FILE, B);


    Unet unet;
    UnetConfig* config = &(unet.config);
    config->B = B;
    config->C_in = C_in;
    config->C_model = C_model;
    config->C_time = C_time;
    config->C_out = C_out;
    config->H = H;
    config->W = W;
    config->block_size = 512;
    config->gn_n_groups = 32;
    config->n_levels = n_levels;
    config->channel_mult = channel_mult;
    config->HS = 32;
    config->n_res = 2;
    config->max_period = max_period;
    config->att_start_level = 2;

    unet_init(&unet);

    unet_count_layers(config);
    unet_make_ptrs_and_count_memory(&unet);
    malloc_and_point(&unet);

    // allocate host memory for params
    float* params_mem_cpu = (float*)mallocCheck(unet.params.total_size * sizeof(float));
    freadCheck(params_mem_cpu, sizeof(float), unet.params.total_size, model_file);
    fcloseCheck(model_file);
    UnetBlock params_cpu;
    unet_setup_and_count_params(config, &params_cpu);
    params_cpu.mem_ptr = params_mem_cpu;
    point_unet_params(config, &params_cpu);

    // copy params to device
    cudaCheck(cudaMemcpy(unet.params.mem_ptr, params_mem_cpu, unet.params.total_size * sizeof(float), cudaMemcpyHostToDevice));

    // allocate host memory for debug states
    FILE *debug_file = fopenCheck("unet_test_acts.bin", "rb");
    size_t state_sizes[NUM_DEBUG_STATES];
    int out_size = B * C_out * H * W;
    state_sizes[0] = B * C_in * H * W; // input
    state_sizes[1] = B; // timestep emb
    state_sizes[2] = out_size; // out
    state_sizes[3] = B * C_in * H * W; // dinp
    state_sizes[4] = out_size; // target

    size_t num_debug_params = 0;
    for (int i = 0; i < NUM_DEBUG_STATES; i++) {
        num_debug_params += state_sizes[i];
    }

    float* debug_states_cpu = (float*)mallocCheck(num_debug_params * sizeof(float));
    freadCheck(debug_states_cpu, sizeof(float), num_debug_params, debug_file);
    // point to the debug states
    DebugStates debug_states;
    float** debug_ptrs[] = {
        &(debug_states.input),
        &(debug_states.timestep_emb),
        &(debug_states.out),
        &(debug_states.dinp),
        &(debug_states.target),
    };

    float* debug_mem_ptr = debug_states_cpu;
    for (int i = 0; i < NUM_DEBUG_STATES; i++) {
        *(debug_ptrs[i]) = debug_mem_ptr;
        debug_mem_ptr += state_sizes[i];
    }

    // read the groud truth grads
    float* grads_mem_cpu = (float*)mallocCheck(unet.grads.total_size * sizeof(float));
    freadCheck(grads_mem_cpu, sizeof(float), unet.grads.total_size, debug_file);
    fcloseCheck(debug_file);


    // read the losses
    float losses[10];
    FILE *loss_file = fopenCheck("unet_test_losses.bin", "rb");
    freadCheck(losses, sizeof(float), 10, loss_file);
    fcloseCheck(loss_file);

    // create a grads struct and point to cpu grads for easier validation
    UnetBlock grads_cpu;
    unet_setup_and_count_params(config, &grads_cpu);
    grads_cpu.mem_ptr = grads_mem_cpu;
    point_unet_params(config, &grads_cpu);

    //// copy input to device
    //cudaCheck(cudaMemcpy(unet.input, debug_states.input, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));
    // load input
    dataloader_next_batch(&loader);
    cudaCheck(cudaMemcpy(unet.input, loader.input, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));

    // try out the random number generator
    GaussianDiffusion diffusion;
    diffusion_init(&diffusion, B, C_in, H, W, 1000);
    printf("successful gaussian init\n");

    float rand_nums[B * 10];
    sample_timesteps(&diffusion, rand_nums, B * 10);

    // compute timestep emb
    float* d_timesteps;
    cudaCheck(cudaMalloc(&d_timesteps, B * sizeof(float)));
    cudaMemcpy(d_timesteps, rand_nums, B * sizeof(float), cudaMemcpyHostToDevice);
    get_timestep_embeddings(&unet.time_emb_state, d_timesteps, unet.timestep_emb);

    // generate all the gaussian noise
    float *all_noise = (float*)malloc(10 * B * C_in * H * W * sizeof(float));
    normal_(all_noise, 10 * B * C_in * H * W, 0.0f, 1.0f, &diffusion.host_rng);


    cudaCheck(cudaMalloc(&(unet.target), out_size * sizeof(float)));

    //printf("drawing and saving x_t for verification\n");
    //float* x_t_cpu = (float*)mallocCheck(B * C_in * H * W * sizeof(float));
    //diffusion_draw_and_sample_xt(&diffusion, unet.target, d_timesteps, unet.input);
    //print_gpu_first_5("x_t", unet.target);
    //cudaCheck(cudaMemcpy(x_t_cpu, unet.target, B * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost));
    //FILE *sample_debug_file = fopenCheck("./sample_debug.bin", "wb");
    //size_t result = fwrite(rand_nums, sizeof(float), B, sample_debug_file);
    //if (result != B) {
    //    fprintf(stderr, "Error writing to file\n");
    //    exit(1);
    //}
    //result = fwrite(loader.input, sizeof(float), B * C_in * H * W, sample_debug_file);
    //if (result != B * C_in * H * W) {
    //    fprintf(stderr, "Error writing to file\n");
    //    exit(1);
    //}
    //result = fwrite(x_t_cpu, sizeof(float), B * C_in * H * W, sample_debug_file);
    //if (result != B * C_in * H * W) {
    //    fprintf(stderr, "Error writing to file\n");
    //    exit(1);
    //}
    //fcloseCheck(sample_debug_file);


    // copy the targets to gpu
    //cudaCheck(cudaMemcpy(unet.target, debug_states.target, out_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(unet.target, all_noise, out_size * sizeof(float), cudaMemcpyHostToDevice));
    diffusion_sample(d_timesteps, unet.input, unet.target, &diffusion, B, C_in * H * W);
    //diffusion_draw_and_sample_xt(&diffusion, unet.target, d_timesteps, unet.input);



    printf("\nCheck that unet targets match the debug states\n");
    validate_result(unet.target, debug_states.target, "target", out_size);


    float acc = 1e-4;
    float med_acc = 1e-2;
    printf("Running 10 update loops\n");
    // loop 1
    printf("Checking forward pass\n");
    float t_res = 0.0f;
    float t_att = 0.0f;
    unet_forward(cublas_handle, &unet, &t_res, &t_att);
    printf("resblock time: %f ms, attention time: %f ms\n", t_res, t_att);
    printf("\nChecking out\n");
    validate_result(unet.output, debug_states.out, "out", out_size, med_acc);
    printf("cpu loss: %f, gpu loss: %f\n", losses[0], unet.mean_loss);
    printf("\nForward pass successful\n");

    printf("\nChecking backward pass\n");
    t_res = 0.0f;
    t_att = 0.0f;
    unet_backward(cublas_handle, &unet, &t_res, &t_att);
    printf("resblock back time: %f ms, attention back time: %f ms\n", t_res, t_att);
    printf("\nChecking grads\n");
    validate_result(unet.grads.mem_ptr, grads_mem_cpu, "grads", unet.grads.total_size, med_acc);
    printf("\nChecking dinp\n");
    validate_result(unet.dinp, debug_states.dinp, "dinp", B * C_in * H * W, acc);
    unet_update(&unet, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
    unet_zero_grad(&unet);

    // loop 2-9
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaProfilerStart());
    cudaCheck(cudaEventRecord(start));
    for (int i = 1; i < 10; i++) {
        dataloader_next_batch(&loader);
        cudaCheck(cudaMemcpy(unet.input, loader.input, B * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));
        cudaMemcpy(d_timesteps, rand_nums + i * B, B * sizeof(float), cudaMemcpyHostToDevice);
        get_timestep_embeddings(&unet.time_emb_state, d_timesteps, unet.timestep_emb);

        cudaCheck(cudaMemcpy(unet.target, all_noise + i * out_size, out_size * sizeof(float), cudaMemcpyHostToDevice));
        diffusion_sample(d_timesteps, unet.input, unet.target, &diffusion, B, C_in * H * W);
        //diffusion_draw_and_sample_xt(&diffusion, unet.target, d_timesteps, unet.input);

        unet_forward(cublas_handle, &unet);
        unet_backward(cublas_handle, &unet);
        printf("step %d, cpu loss: %f, gpu loss: %f\n", i+1, losses[i], unet.mean_loss);
        unet_update(&unet, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i+1);
        unet_zero_grad(&unet);
    }
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(end));
    float time_elapsed;
    cudaCheck(cudaEventElapsedTime(&time_elapsed, start, end));
    time_elapsed /= 9;
    printf("Average time per step: %f ms\n", time_elapsed);



    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmark\n");
    int repeat_times = 50;
    float elapsed_time = benchmark_kernel(repeat_times, unet_forward, cublas_handle, &unet, nullptr, nullptr);
    printf("Forward pass time: %f ms\n", elapsed_time);

    printf("\nBackward pass benchmark\n");
    elapsed_time = benchmark_kernel(repeat_times, unet_backward, cublas_handle, &unet, nullptr, nullptr);
    printf("Backward pass time: %f ms\n", elapsed_time);

    // free memory
    free(params_mem_cpu);
    free(debug_states_cpu);
    free_unet(&unet);
    cudaCheck(cudaFree(unet.target));
    cudaCheck(cudaFree(d_timesteps));
    dataloader_free(&loader);
    diffusion_free(&diffusion);
    free(all_noise);

    // free cublas
    cublasCheck(cublasDestroy(cublas_handle));
}