/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;//b是batch，t是time_step,c是channel
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;//输出位置
    const floatX* wte_ix = wte + ix * C + c;//词矩阵
    const floatX* wpe_tc = wpe + t * C + c;//位置编码

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}//编码的函数

template <int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    /*
    Grid层：每个block处理一个独立的桶（bucket），通过blockIdx.x直接映射到桶的索引。这种设计确保不同桶之间的计算完全独立，无需同步。
    Block层：一个block处理单个桶内的所有数据。block的大小由模板参数BLOCK_SIZE（默认256线程）决定，内部通过warp划分任务。

    Warp作为基本执行单元：每个warp（32线程）负责处理桶中一个元素（item）的部分梯度。
        warp_id：标识block内的warp编号（0~BLOCK_SIZE/WARP_SIZE-1）。
        lane_id：标识warp内的线程编号（0~31）。
    动态负载均衡：
        循环步长BLOCK_SIZE/WARP_SIZE：每个warp处理item = warp_id, warp_id + step, ...，确保桶内所有元素被均匀分配到block的warps。
        条件提前退出：若warp_id >= bucket_size，该warp无任务，立即返回。
    
    向量化计算：每个线程处理x128::size个通道（如8个float32），通过load128cs和store128实现向量化内存访问。
    c的计算：全局通道偏移由桶的w字段（通道组索引）和lane_id共同决定，确保线程间连续访问。
    寄存器累加：每个线程在寄存器accum[x128::size]中局部累加梯度，避免频繁访问全局内存。

    Partial Results收集：
        非warp0的线程将累加结果写入共享内存accum_shared，结构为[x128::size][BLOCK_SIZE]。
        Warp0的线程从共享内存中收集所有warps的部分结果，累加到自己的寄存器。
    同步机制：通过__syncthreads()确保所有共享内存写入完成后再进行读取。

    读-修改-写操作：Warp0线程从全局内存加载原始梯度值packed_in_out，与累加结果相加后，通过stochastic_rounding进行随机舍入。
    确定性保证：种子seed + bucket * WARP_SIZE + threadIdx.x + k为每个参数的舍入生成唯一随机数，确保结果可复现。

    内存访问优化：
        Coalesced Access：通过load128cs实现跨线程的连续内存访问（cs后缀暗示缓存流操作，优先L2缓存）。
        共享内存Bank冲突规避：accum_shared的布局（k * BLOCK_SIZE + threadIdx.x）可能优化Bank访问模式。
        计算隐藏延迟：
        Warp0在等待其他warps写入共享内存时，提前加载dwte的原始值（load128(dwte_ix)），隐藏内存延迟。
        负载均衡：
        CPU预处理将大桶排在前面，优先处理计算密集型任务，减少GPU空闲时间。
    */
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    /*
    • 为了保证确定性（determinism），输入数据已由 CPU 预处理成桶，每个桶对应于单个词的一个特定通道范围。
    • 每个桶覆盖 (WARP_SIZE * x128::size) 个通道；
    • 每个线程负责处理 x128::size 个通道，一整个 warp （通常 32 线程）就能处理一大块数据；
    • 每个 CUDA 块中，线程被划分为若干 warp，每个 warp 负责处理桶中一个元素（比如一个 token 的部分梯度），所以一个块可并行处理 (BLOCK_SIZE / WARP_SIZE) 个桶内的元素；(BLOCK_SIZE / WARP_SIZE）个warp
    • 针对小桶（元素数量少）部分 warp 会直接返回；若桶内元素较多，则需要循环处理所有元素；
    • 桶在 CPU 预排序，较大的桶排在前边，以便尽早处理高工作量的数据。
    */
    int bucket = blockIdx.x;//一个block对应一个桶
    int warp_id = threadIdx.x / WARP_SIZE;//warp的id
    int lane_id = threadIdx.x % WARP_SIZE;//warp内thread的id
    int c_per_warp = WARP_SIZE * x128::size;//每个 warp 负责处理的通道总数
    /*
    • 从 bucket_info 数组中取出第 bucket 个桶的各个字段：  
    – bucket_start_idx：此桶在 workload_indices 中的起始偏移，用于定位桶中各条目的索引。  
    – bucket_size：该桶内包含的元素（token）数量。  
    – bucket_ix：桶对应的目标梯度 dwte 中的偏移索引（例如对应 embedding 矩阵的行号）。 
    – bucket_info[bucket].w：用于计算起始通道组索引，每个桶可能只处理部分通道。
    • 变量 c 计算了当前线程负责处理的起始通道索引。计算方式为：桶所对应的通道组编号乘以每个组的通道数（c_per_warp），再加上当前线程 lane_id 对应的偏移（乘以单线程处理的通道数 x128::size）。
    */
    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};//每个线程的accum
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];//每一个block的线程数*128个通道的梯度，所有通道的梯度
    //warp是什么？是一个线程束，是物理上的多个线程组成的一个组，一个warp中的线程是同步，warp之间是异步的
    
    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/WARP_SIZE) {
        /*
        item < bucket_size，说明item是当前block内的，item = warp_id作为起始，说明一个线程会遍历多个warp

        同一个warp内的线程，不就意味着相同的item，会有相同的bt，但是c不同，所以dout_btc不同，accum累加的值不同

        不同的warp的线程，就存在一个前后次序的问题，前面的warp的线程循环的次数多，后面的线程循环的次数少

        • 循环变量 item 初始为当前 warp_id。原因在于同一 block 内的线程 按不同 warp 分布任务，步长为 BLOCK_SIZE/WARP_SIZE（即块内 warps 数），这样一个块内所有 warp 能并行处理桶内所有元素。
        • 对于每个 item：
          – 利用 bucket_start_idx 和 item 取得实际数据索引 bt，该索引对应 dout 数据的行。
          – 计算 dout_btc 指针，指向当前梯度数据的起点。偏移计算为：bt * C + c，其中 C 是每个样本的总通道数，c 是当前线程负责的起始通道。
          – 调用 load128cs(dout_btc) 以流式缓存方式加载 128 位数据（多个连续通道的数据）到一个 x128 类型变量 packed_inp1。
          – 内部循环遍历 packed_inp1 中的每个分量，累加到 accum 数组中，用以累加不同 item 的梯度。
        */
        //希望一个warp处理一个词？warp中的线程，处理一个词的不同channel
        int bt = workload_indices[bucket_start_idx + item];
        //这里不是对线程处理，而是对warp处理，所以是bt*C，而不是b*T*C+t*C
        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    /*
    • 仅留在 warp0 的线程继续下面的工作：
      – 根据 bucket_ix 和 c 计算 dwte_ix 指针，此指针指向目标梯度数组 dwte 中当前桶和当前通道分片的位置。
      – 调用 load128(dwte_ix) 加载该位置已有的 128 位梯度数据到 packed_in_out 中，用于后续的累加更新（这是一种读-修改-写操作）。
    */
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = load128(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    
    /*
    • warp0 的线程（仍在进行计算）需要将共享内存中来自其他 warps 的累加结果汇总到自身累加器中。
    • 循环变量 i 从 threadIdx.x + WARP_SIZE 开始，每次加上 WARP_SIZE，遍历整个块中非 warp0 写入共享内存的数据。上限取 min(BLOCK_SIZE, bucket_size*WARP_SIZE)，确保遍历的范围不会超过块内实际参与计算的线程数。
    • 内层循环对每个通道（k 从 0 到 x128::size-1）将共享内存中对应位置的值累加到累加数组 accum 中。
    */
    // 外层for循环是读取一个block内的不同的warp
    for (int i = threadIdx.x+WARP_SIZE; i < min(BLOCK_SIZE, bucket_size*WARP_SIZE); i += WARP_SIZE) {
        // 内层for循环是对不同的block做循环？？
        // accum是什么存储层级
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        /*
        • 循环遍历每个通道（k 范围为 0 到 x128::size-1），对当前累积结果进行更新：
          – 将累加结果 accum[k] 与原先加载的 dwte 数据 packed_in_out[k] 相加，得到待更新值。
          – 为了将 FP32（32 位浮点）值转换成 BF16（或其他精度较低格式），使用随机舍入（stochastic_rounding）函数。在转换过程中引入随机性，可以减少舍入误差的系统误差。调用时传入：
            • 待舍入的数值：累加后的结果；
            • 输出指针：指向当前 packed_in_out 中对应分量的位置；
            • 调整过的随机种子：seed + bucket * WARP_SIZE + threadIdx.x + k，保证每个位置的随机性都是唯一且确定的。
        • 最后，调用 store128(dwte_ix, packed_in_out) 将更新后的 128 位数据写回全局内存对应位置，实现梯度的写回更新。
        */
        stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * WARP_SIZE + threadIdx.x + k);
    }
    store128(dwte_ix, packed_in_out);
}

__global__ void wpe_backward_kernel(floatX* dwpe,
                                    const floatX* dout, const int* inp,
                                    int B, int T, int C, unsigned int seed) {
    // Each thread handles x128::size "channel positions", e.g. 256 per warp for BF16
    // For gpt2-124M BF16, C=768 and T=1024, so 3 warps per channel and 3072 warps in total
    // For each "channel position" we sum the gradients for every batch at that C/T element
    // This way each dwte element is only updated once, and the kernel is fully deterministic!
    // The previous kernel was not deterministic, as batches were aggregated with atomicAdd
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= T * C) { return; }

    // if C is not a multiple of WARP_SIZE*x128::size, it's OK for some warps to handle multiple t
    int t = idx / C;
    int c = idx % C;
    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        x128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c); // will never be read again
        for (int k = 0; k < x128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    floatX* dwpe_tc = dwpe + (t * C) + c;
    x128 packed_dwpe = load128(dwpe_tc);
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + idx + k);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N = T * C / x128::size;
    const int grid_size = CEIL_DIV(N, block_size);
    wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    // check the GPU scratch buffer is large enough to hold the bucket info and workload indices
    // todo - this is trivially true given hardcoded scratch buffer size here, is this useful?
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}
