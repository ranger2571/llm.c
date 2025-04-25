/*
Kernels for a global norm.
Global norm in this context means that we want to calculate a single norm cooperatively using all avalailable SMs, instead
 of multiple norms that can be handled by separate blocks.

Compile example:
nvcc -O3 --use_fast_math global_norm.cu -o global_norm
*/


#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// turn on bf16 as default, done up here for now
#define ENABLE_BF16
#include "common.h"

cudaDeviceProp deviceProp;

float global_norm_cpu(const float* data, size_t count) {
    // accumulate in double so we have an accurate numerical reference
    double acc = 0.0;
    for(size_t i = 0; i < count; ++i) {
        acc  += (double)data[i] * (double)data[i];
    }
    return (float)acc;
}

// meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
template<class T>
__global__ void norm_kernel1(float* out, const T* data, size_t count) {
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.
    namespace cg = cooperative_groups;
    // 取当前线程所在的线程块
    cg::thread_block block = cg::this_thread_block();
    // 将整个线程块划分为大小为 32 的子组（即 warp），便于后续 warp 内的归约  
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float block_result[32];

    // out will be updated atomically from all thread blocks
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;//整个grid的thread id
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    //grid-stride 循环的基本思想是：每个线程从 index 开始，每隔 grid_width 个元素处理一次，确保所有数据都能被遍历。这样做可以适应数据量与线程数不一致的情况，同时均衡负载。
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];//将count大小的数据量，reduce到grid的所有线程中
    }
    // warp-level reduce
    float warp_result = cg::reduce(warp, accumulator, cg::plus<float>{});//将一个warp内的数据reduce到一个寄存器中,warp内所有线程的warp_result相同
    block_result[warp.meta_group_rank()] = warp_result;//寄存器转移到共享内存中，
    //warp.meta_group_rank() 返回当前 warp 的编号。每个 warp 的所有线程执行这行代码时，其编号相同（也即所有线程得到的 warp_result 是一样的）注意：通常做法是让 warp 中的一个线程（如 thread_lane==0）写入共享内存；此处代码写法可能暗示所有线程都有机会写入同一位置，但由于 warp 内归约结果相同，所以效果等同于取其中一个。
    block.sync();

    // 只有 warp 中编号为0的线程负责块内的最终归约,调用 atomicAdd 将结果写入全局内存
    if(warp.meta_group_rank() == 0) {
        // 对于当前 warp 内的线程，如果线程编号小于 warp 的有效线程数，则取共享内存中对应的累加结果，否则设为 0
        float gather = warp.thread_rank() < warp.meta_group_size() ? block_result[warp.thread_rank()] : 0.f;
        // 对这些结果做归约，得到整个线程块的累积值
        float block_sum = cg::reduce(warp, gather, cg::plus<float>{});
        // 再由 warp 中第 0 号线程执行原子加，将块的累积值加到全局输出中  
        if(warp.thread_rank() ==  0) {
            atomicAdd(out, block_sum);
        }
    }
}

template<class T>
__global__ void norm_kernel2(float* out, const T* data, size_t count) {
    // concrete example for an A100 GPU (108 SMs, 2048 max threads each)
    // so there are 2048 * 108 = 221,184 threads total
    // say the block_size is 512, then we would launch 432 blocks in total
    // say num_params is ~100M, each thread will process ~500 elements

    // warps reduce with warp-level reduce, we have 221,184/32 = 6,912 warps
    // and then each warp atomicAdd's to global memory, total of 6,912 atomics

    // no shared memory; but one atomic per warp instead of per block
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // out will be updated atomically from all thread blocks
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // warp-level reduce
    float warp_result = cg::reduce(warp, accumulator, cg::plus<float>{});
    // and atomic in global buffer
    if(warp.thread_rank() == 0) {
        atomicAdd(out, warp_result);
    }
}

template<class T>
__global__ void norm_kernel3(float* out, const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    float block_sum = blockReduce<warpReduceSum>(accumulator);
    if(threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

// Same as kernel3 but without atomic adds -> this allows us to have determinism due to the
// non associativity of floating point operations. Roughly same performance as kernel3.
template<class T>
__global__ void norm_kernel4(float* out, const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    float block_sum = blockReduce<warpReduceSum>(accumulator);
    // each block accumulates its partial sum to out[blockIdx.x]
    // we want to avoid using atomic add here so we combine this kernel with the aggregate kernel call
    // that sums up the partial block sums
    if(threadIdx.x == 0) {
        out[blockIdx.x] = block_sum;
    }
}

__global__ void global_norm_aggregate_kernel(float* out, size_t count) {
    size_t index = threadIdx.x;
    // grab block sums from the previous kernel, use 0. as the neutral sum element
    float block_sum = (index < count) ? out[index] : 0.f;
    float sum = blockReduce<warpReduceSum>(block_sum);
    if(threadIdx.x == 0) {
        out[0] = sum;  // out[0] ends up with the final norm squared
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

template<typename T>
void global_norm1(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    /*
    启动足够的块来填充网格。故意不使用 DIV_CEIL。少一个块会对性能产生很小的影响，而多一个块则会造成灾难性的后果，因为它只能在所有其他块完成后才能启动。无论如何，我认为 cuda_threads_per_SM 在所有 gpu 上都应该是 512 的倍数，因此除法确实是精确的。???没看懂什么意思

    作者的意思可能是，现在能启动的thread的数目已经确定了就是cuda_threads_per_SM * cuda_num_SMs，如果grid的数目比当前的值更大，就会导致资源短缺，有一个block就会需要等待到资源存在空闲才能启动，这会显著的拖慢计算速度
    */
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    norm_kernel1<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void global_norm2(float* out, const T* values, size_t count, int block_size) {
    // ditto
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    norm_kernel2<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void global_norm3(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);  // gives a better error than letting the call below fail
    norm_kernel3<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void global_norm4(float* out, const T* values, size_t count, int block_size) {
    if (block_size <= 64) {
        block_size = 128;  // to avoid triggering the assert below
    }
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    assert(grid_size < 1024);  // we want to later accumulate the block sums in a single block
    norm_kernel4<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
    global_norm_aggregate_kernel<<<1, 1024>>>(out, grid_size);
    cudaCheck(cudaGetLastError());
}

void global_norm(int kernel_num, float* out, const floatX* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(out, values, count, block_size);
        case 2:
            return global_norm2(out, values, count, block_size);
        case 3:
            return global_norm3(out, values, count, block_size);
        case 4:
            return global_norm4(out, values, count, block_size);
    }
}

int main(int argc, const char **argv) {
    setup_main();
    cudaGetDeviceProperties(&deviceProp, 0);

    int C = 768;
    int L = 12;

    size_t num_params = (size_t)(C * 4*C + C*C) * 2 * L;

    // create host memory of random numbers
    float* inp = make_random_float(num_params);
    // scale them down
    for(size_t i = 0; i < num_params; ++i) {
        inp[i] *= 1e-3;
    }

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    float out = global_norm_cpu(inp, num_params);

    // move to GPU
    float* d_out;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_out,  1024 * sizeof(float)));  // 1024 needed for kernel 4
    cudaCheck(cudaMalloc(&d_inp, num_params * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, num_params));

    int block_sizes[] = {32, 64, 128, 256, 512, 768, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        global_norm(kernel_num, d_out, d_inp, num_params, block_size);
        validate_result(d_out, &out, "out", 1, 1e-2f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, global_norm,
                                              kernel_num, d_out, d_inp,
                                              num_params, block_size);
        size_t memory_ops = num_params * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
}