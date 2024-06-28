void mse_forward(const float* inp, const float* y, float* loss, int N, int block_size);

void mse_backward(const float* inp, const float* y, float* dinp, int N, int block_size);