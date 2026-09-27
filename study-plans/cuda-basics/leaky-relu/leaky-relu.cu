#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write code here
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int cvn = idx;cvn < N;cvn+=stride){
        output[cvn] = fmaxf(input[cvn],input[cvn]*alpha);
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}