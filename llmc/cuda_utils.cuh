// Utilities for use in __device__ code

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
/*
关键字 template 表示这是一个模板（泛型）结构体，ElementType 是一个类型参数，使用者可以指定具体的类型，例如 float、int 等。
关键字 struct 表示定义了一个结构体（可以看作一个类，成员默认是公共的）。
alignas(16) 表示该结构体的实例在内存中需要按照 16 字节进行对齐，这对于某些需要特定内存对齐要求（例如 SIMD 优化或者 GPU 编程）的场合非常重要。
Packed128 是该结构体的名字。
*/
    Packed128() = default;//自动构造函数
    __device__ explicit Packed128(int4 bits) {
        /*
        定义了一个构造函数，它接收一个 int4 类型的参数（int4 是四个整数组成的数据结构，经常用于 CUDA 代码中表示 128 位数据）。
        关键字 __device__ 表明这个构造函数仅在 GPU 的设备端可用。
        explicit 表示这个构造函数是明确调用的，避免隐式转换。??什么叫明确调用？？
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");这是一条编译期间检查语句，确保传入的 bits 变量和结构体中存储数据的 payload 数组大小一致，不匹配时编译器将报错，并显示 “Size mismatch.” 错误信息。
        memcpy(&payload, &bits, sizeof(bits));利用 memcpy 将 bits 中的内存内容复制到 payload 数组中，从而把 int4 数据拷贝到结构体内部。
        */
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }


    __device__  static Packed128 constant(ElementType value) {
        /*这个函数是静态函数，意思是你不需要创建结构体对象就可以调用它。返回构造好的 result 对象。*/
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);//生成一个所有值都为0的Packed128对象
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        /*（用于非 const 对象）
        函数重载了“[]”运算符，使得用户可以通过对象[index]的形式访问 payload 数组具体的元素。
        返回的是对 payload[index] 的引用，可以用来修改对应的元素。
        */
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        /*（用于 const 对象）
        函数重载了“[]”运算符，用于当 Packed128 对象被定义为常量时使用,
        用户可以通过对象[index]的形式访问 payload 数组具体的元素。
        返回的是对 payload[index] 的常量引用，这样确保调用该函数时不会修改 payload 中的数据。*/
        return payload[index];
    }
    __device__ int4 get_bits() const {
        /*成员函数
        • 作用是将存储在 payload 数组中的数据转换为 int4 类型并返回。
        • 声明一个 int4 类型的变量 bits，用来存储结果。
        • 同样使用 static_assert 检查 bits 与 payload 的大小是否一致。
        • 使用 memcpy 将 payload 的内存内容复制到 bits 中。
        • 返回复制后的 bits。这个函数可以用作将内部二进制数据提取为一个四整数的数据块。*/
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
    /*
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    这一行定义了一个静态常量 size，它表示 payload 数组中包含多少个 ElementType 类型的元素。计算方法为：将 int4 类型的总字节数除以 ElementType 的字节数。    
    constexpr 表示这个值在编译期间就会确定，可以用于编译时检查。
    这样当 ElementType 类型不同时，数组的大小会自动调整以保证整体数据大小与 int4 相同（即 16 字节）。
    声明了一个数组 payload，该数组每个元素的类型为 ElementType，总共包含 size 个元素。    
    这就是 Packed128 用来存储实际数据的地方
    */
};

// load a Packed128 from an aligned memory address
// 从对齐的内存地址中load一个packed128对象
template<class ElementType>//address是指向元素类型ElementType 的常量指针，要求这个地址对齐，便于 128 位的读取操作。
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
    //使用 reinterpret_cast 将给出的 ElementType 指针转换为 const int4（int4 表示 128 位数据的向量类型，包含4个int型成员）。对转换后的指针进行解引用（*），将其内容读取出来，并利用花括号初始化语法构造出一个 Packed128 实例返回。注意：这个操作要求传入的地址必须按 int4（128 位）要求对齐，否则可能发生未定义行为。
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
    //CUDA 内置函数 __ldcs，该函数用于从全局内存加载数据，并且提供“流式缓存提示”（cs 表示 cache streaming），加载的数据不会保存在常规 L1 缓存中，适合一次性读入而不必频繁重用的数据。
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
    //将target指针转化为int4指针，然后解引用，将value的数据存储到target中
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
    //__stcs 是 CUDA 内置的存储函数，专门用于存储数据到全局内存时给予一个流式缓存提示：
    //该函数调用告诉硬件按照流式缓存方式管理数据，即这种存储方式常用于数据一次写入而不频繁读取的场合，从而避免占用 L1 缓存、减少缓存污染。
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
    //__stcg 会将数据存入全局内存，但采用的缓存策略是在写入时绕过 L1 缓存，仅缓存到 L2。这在某些应用中可以减少对 L1 缓存的占用，改善内存数据的共享效果或满足特定的硬件优化需求。
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// DType support

// enumerator to indentify the datatype of a tensor.
enum class DType : uint8_t {
    FP32, FP16, BF16
};

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }



// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
// warp-level reduction for finding the maximum value
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// Performs a _deterministic_ sum reduction. determinism is achieved by requiring that only
// a single block be used.
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// memory management

// allocate memory, preferrably on the device
// returns a status code. 0 = OK, 1 = fell back to managed memory
int cudaMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    // try to allocate
    cudaError_t err = cudaMalloc(out, bytes);
    if(err == cudaErrorMemoryAllocation) {
        // if we OOM, fallback to a managed allocation. slower but at least won't crash.
        cudaGetLastError(); // reset the error before the next API call
        cudaCheck_(cudaMallocManaged(out, bytes), file, line);
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
        return 1;
    } else {
        cudaCheck_(err, file, line);
        return 0;
    }
}

#define cudaMallocConditionallyManaged(out, bytes)\
(cudaMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    // makes sure each thread gets a different random number
    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

#endif