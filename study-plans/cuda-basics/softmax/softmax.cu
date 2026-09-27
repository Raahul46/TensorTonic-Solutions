#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    // write a for loop for every thread to go through the stride
    // first find the max value 
    // next add the difference to the global sum
    // parallel calc of softmax
    int idx = threadIdx.x;
    float local_max = -INFINITY;
    __shared__ float shared_max[256];
    for(int cvn = idx;cvn<N;cvn+=blockDim.x){
        local_max = fmaxf(local_max,input[cvn]);
    }
    shared_max[idx] = local_max;
    __syncthreads();
    for(int fm = blockDim.x/2;fm > 0; fm/=2){
        if(idx < fm){
            shared_max[idx] = fmaxf(shared_max[idx],shared_max[idx+fm]);
        }
        __syncthreads();
    }
    float max = shared_max[0];
    float local_sum = 0;
    for(int bnm = idx; bnm < N;bnm+=blockDim.x){
        local_sum += expf(input[bnm]-max);
    }
    __shared__ float shared_sum[256];
    shared_sum[idx] = local_sum;
    __syncthreads();
    for(int bnm = blockDim.x/2;bnm>0;bnm/=2){
        if(idx<bnm){
         shared_sum[idx] += shared_sum[idx+bnm];
        }
        __syncthreads();
    }
    float full_sum = shared_sum[0];
    for(int bnm = idx;bnm < N;bnm+=blockDim.x){
        output[bnm] = expf(input[bnm]-max)/full_sum;
    }
}
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}