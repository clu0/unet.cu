#include <cuda_runtime.h>
#include "common.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include "timestep_embedding.cuh"




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


#ifndef LINKING
int main(int argc, char **argv) {
    int B = 32;
    int dim = 64;
    int max_period = 1000;

    // read ground truth
    float* timesteps = (float*)malloc(B * sizeof(float));
    float* embeddings = (float*)malloc(B * dim * sizeof(float));
    float* freqs = (float*)malloc(dim / 2 * sizeof(float));
    FILE *file = fopenCheck("time_emb.bin", "rb");
    freadCheck(timesteps, sizeof(float), B, file);
    freadCheck(embeddings, sizeof(float), B * dim, file);
    freadCheck(freqs, sizeof(float), dim / 2, file);
    fcloseCheck(file);

    TimestepEmbedding t_emb;
    init_timestep_embedding(&t_emb, dim, B, max_period);
    float *d_timesteps, *d_embeddings;
    cudaCheck(cudaMalloc(&d_timesteps, B * sizeof(float)));
    cudaCheck(cudaMalloc(&d_embeddings, B * dim * sizeof(float)));

    cudaCheck(cudaMemcpy(d_timesteps, timesteps, B * sizeof(float), cudaMemcpyHostToDevice));


    printf("Checking freqs\n");
    validate_result(t_emb.freqs, freqs, "freqs", dim / 2);


    get_timestep_embeddings(&t_emb, d_timesteps, d_embeddings);

    printf("Checking timestep embeddings\n");
    float emb_acc = 1e-3;
    validate_result(d_embeddings, embeddings, "embeddings", B * dim, emb_acc);

    free_timestep_embedding(&t_emb);
    free(timesteps);
    free(embeddings);
    free(freqs);
    cudaFree(d_timesteps);
}
#endif