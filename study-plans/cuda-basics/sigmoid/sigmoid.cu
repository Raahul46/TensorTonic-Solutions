#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int loop_idx = idx; loop_idx < N; loop_idx+=stride){
        output[loop_idx] = 1/(1+expf(-input[loop_idx]));   
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}