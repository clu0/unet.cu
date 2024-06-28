#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

inline int max_int(int a, int b) {
    return a > b ? a : b;
}

inline int min_int(int a, int b) {
    return a < b ? a : b;
}

// many utils below copied from llm.c

// CUDA error checking
inline void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
inline void cublas_check(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublas_check((status), __FILE__, __LINE__); }

template<const int K>
void print_gpu_first_k(const char* name, const float* gpu_ptr) {
    float cpu_ptr[K];
    cudaCheck(cudaMemcpy(cpu_ptr, gpu_ptr, K * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s first %d:\n", name, K);
    for (int i = 0; i < K; i++) {
        printf("%f ", cpu_ptr[i]);
    }
    printf("\n");
}

#define print_gpu_first_5(name, gpu_ptr) print_gpu_first_k<5>(name, gpu_ptr)

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4, int n_print=5, int check_all=0) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    if (check_all) {
        // compute the sum of absolute differences and print
        float abs_sum = 0.0f;
        for (int i = 0; i < num_elements; i++) {
            abs_sum += fabs(cpu_reference[i] - (T)out_gpu[i]);
        }
        printf("%s: sum of absolute differences: %f\n", name, abs_sum);
    }
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < n_print) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // ensure correctness for all elements. We can set an "ignore" mask by writing NaN
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && isfinite(cpu_reference[i])) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= max_int(10, n_print)) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

FILE *fopen_check(const char *path, const char *mode, const char *file, int line);

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

void fclose_check(FILE *fp, const char *file, int line);

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

void fseek_check(FILE *fp, long off, int whence, const char *file, int line);

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

void *malloc_check(size_t size, const char *file, int line);

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)
