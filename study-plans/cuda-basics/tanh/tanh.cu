#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int xi = idx; xi< N;xi+=stride){
        output[xi] = tanhf(input[xi]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}