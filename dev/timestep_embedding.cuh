typedef struct {
    float* freqs;
    int half_dim;
    int B;
    int max_period;
} TimestepEmbedding;

void init_timestep_embedding(TimestepEmbedding* emb, int dim, int B, int max_period = 10000);

void free_timestep_embedding(TimestepEmbedding* emb);

void get_timestep_embeddings(
    TimestepEmbedding* emb, const float* timesteps, float* out
);