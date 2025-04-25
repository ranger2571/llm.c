#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

#define ENABLE_BF16
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

#define TILE_SIZE 64
#define HEAD_SIZE 64
//gQ(i,j)代表的是[t,channel]的数据
#define gQ(i,j) gQ[(i) * 3 * NH * HS + (j)]//[t,hidden_size]
#define gK(i,j) gK[(i) * 3 * NH * HS + (j)]//[t,hidden_size]
#define gV(i,j) gV[(i) * 3 * NH * HS + (j)]//[t,hidden_size]
#define gO(i,j) gO[(i) * 1 * NH * HS + (j)]//[t,hidden_size]
#define gL(i) gL[(i) * NH]
#define gD(i) gD[(i) * NH]

#define sQ(i,j) sQ[(i) + (j) * TILE_SIZE]

#define sK_T(i,j) sK[(i) * TILE_SIZE + (j)]//[]
#define sV(i,j) sV[(i) * HEAD_SIZE + (j)]//[heads,hidden_size]
#define FLOAT4(value) reinterpret_cast<float4*>(&(value))[0]

__global__ __launch_bounds__(256)
void flash_attention_forward_kernel1(float* out, float* inp, float* l,
                                int B, int T, int NH, int HS) {

    // inp (B, T, 3, NH, HS)
    // out (B, T, NH, HS)
    // l (B, T, NH)

    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B

    // we use 256 threads = 8 warps in each threadblock
    // we use 64KB of shared memory for K, V so each uses 32KB of shared memory
    // Q is stored in registers
    // 32KB of shared memory can store 32 * 1024 / 4 = 8192 floats = 128 * 64 floats
    // so each threadblock computes a 128 * 64 tile of O, and each warp does a 16 * 64 tile of O
    // following flash attention 2, we only store (m + log l) instead of (m, l) for the backward pass
    // 每个warp处理 16*64的tile，所以warp_id是0-7 
    // 一个线程处理4个float，所以一次处理32个线程能处理 2*64的tile
    // 所以需要进行逻辑上的八次重复，才能实现一个warp处理16*64的tile，不论这个重复是如何进行时间上schedule的

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int q_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset = blockIdx.z * T * 3 * NH * HS +                      0 * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset = blockIdx.z * T * 3 * NH * HS +                      0 * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int l_global_offset = blockIdx.z * T * NH + blockIdx.y * TILE_SIZE * NH + blockIdx.x;
    // T存在的意义就是，对Q做第一次切分，然后根据切分后的结果，来进行第二次切分

    float* gQ = &inp[q_global_offset];
    float* gK = &inp[k_global_offset];
    float* gV = &inp[v_global_offset];
    float* gO = &out[o_global_offset];
    float* gL = &l[l_global_offset];

    extern __shared__ float sharedMemory[];

    float* sK = &sharedMemory[0 * TILE_SIZE * 64];//sK size = TILE_SIZE * 64
    float* sV = &sharedMemory[1 * TILE_SIZE * 64];

    int tile_increment = TILE_SIZE * 3 * NH * HS;

    // addresses for loading data from global to shared
    // as well as for register tiling

    int thread_row = warp_id * 16 + (lane_id / 16) * 4;// 一个warp内的32线程，会在此时分成两个thread_row的部分，
    // lane_id小于16的，处理warp_id * 16这一行的数据，列是0-15列，一次读4个数据，所以是0-63个数据。
    // lane_id大于16的，处理warp_id * 16 + 4 这一行的数据，（为什么是+4）因为要使用for进行四次重复

    int thread_col = (lane_id % 16) * 4;//(0-32 ==== 0-15 0-15 *4== 0-60 0-60)两个laneid会处理听一个thread_col


    // main loop
    float tQ[8][4];
    float tK[8][4];
    float rQ[8] = {0.0f};
    float rK[8] = {0.0f};
    float rV[4] = {0.0f};
    float tS[8][8] = {0.0f};
    float (&tP)[8][8] = tS;
    //float tP[4][4] = {0.0f};
    float rP[8] = {0.0f};
    float rO[8][4] = {0.0f};
    float rM_old[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float rM[8] = {0.0f};
    float rL_old[8] = {0.0f};
    float rL[8] = {0.0f};
    // this stores sum(rP) across the half-warps
    // in order to compute rL = exp(rM_old - rM) * rL_old + sum(rP)
    float rD[8] = {0.0f};
    unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
    int lane_id_to_read_from = (lane_id < 16) ? 0 : 16; // Lane to read from

    // load gQ to sQ 加载Q到寄存器 这就是一个线程要处理的所有元素了吗？
    // 哦，这里知道为什么要进行warp_id * 16 + 4了。因为是要进行四次重复。
    // 因为一个warp的线程，分配到了两个行
    for (int i = 0; i < 4; i++) {
        FLOAT4(tQ[i][0]) = FLOAT4(gQ(thread_row + i, thread_col));//不同的col，代表一个col切分了四个float的数据
        FLOAT4(tQ[i+4][0]) = FLOAT4(gQ(thread_row + i + 8, thread_col));
    }


    // For auto regressive mask, need to check when kv_tile = blockIdx.y
    // 意味着，将T进行按照tile切分，一个tile是128个token的长度，channel的长度是64
    for (int tile = 0; tile <= blockIdx.y; tile++) {


        for (int i = 0; i < 4; i++) {
            //(这里包不包含合并访存呢？？？)
            FLOAT4(tK[i][0]) = FLOAT4(gK(thread_row + i, thread_col));
            //因为一个warp是16*64的数据，这样的数据，一个warp能同时处理两行，所以进行了切分，第一行处理0-3，第二行处理4-7。然后复用线程，处理8-15行。
            FLOAT4(tK[i+4][0]) = FLOAT4(gK(thread_row + i + 8, thread_col));
            // 每个线程加载8行(thread_row + i和thread_row + i + 8)和4列(thread_col)的Q值
        }

        for (int i = 0; i < 4; i++) {
            for (int j=0; j < 4; j++) {
                sK_T(thread_col + j, thread_row + i) = tK[i][j];
                sK_T(thread_col + j, thread_row + i + 8) = tK[i+4][j];
            }
        }
        // load gV to sV
        for (int i = 0; i < 4; i++) {
            FLOAT4(sV(thread_row + i, thread_col)) = FLOAT4(gV(thread_row + i, thread_col));
            FLOAT4(sV(thread_row + i + 8, thread_col)) = FLOAT4(gV(thread_row + i + 8, thread_col));
        }
        __syncthreads();
        //
        // compute rS
        //

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                tS[i][j] = 0;
            }
        }
        // We use warp shuffling to directly load data to each fragment from tQ to compute the outer product tS.
        // To do this, there is some array indexing involving the modulo operator for tQ to compute the lane id that we want to load data from.
        // For some reason, in this case the compiler will put tQ into local memory, causing register spillage,
        // even though the array indexing can be computed at compile time.
        // To resolve this, we use nested for loops to remove the use of modulo operator.
        // 我们使用warp shuffling操作直接将tQ的数据加载到每个片段，用于计算外积tS。
        // 为此，需要通过模运算计算数组索引，以确定从tQ的哪个lane id加载数据。
        // 但发现这种情况下编译器会将tQ放入本地内存，导致寄存器溢出，
        // 尽管这个数组索引其实可以在编译时计算出来。
        // 为了解决这个问题，我们改用嵌套for循环来消除模运算的使用。
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            // 将K矩阵的列分成16个大块（16×4=64，对应HEAD_SIZE=64）
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // 处理每个大块中的4个连续元素,4次循环对应FLOAT4的4个float，实现向量化加载
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                FLOAT4(rK[0]) = FLOAT4(sK_T(k_fragment, thread_col));
                FLOAT4(rK[4]) = FLOAT4(sK_T(k_fragment, thread_col + 64));//为什么这里是+64


                for (int i = 0; i < 4; i++) {
                    //rdO[i] = sdO(thread_row_64_x_128 + i, k_fragment);
                    rQ[i] = __shfl_sync(mask, tQ[i][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                    rQ[i+4] = __shfl_sync(mask, tQ[i+4][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                }// 这里的Q是一个数值
                // 行为：通过warp shuffle在半个warp内（16线程）广播Q片段 mask=0xFFFF或0xFFFF0000将warp分为两个16线程组
                // 设计原因：避免共享内存访问（寄存器→寄存器直接传输）每个线程只需保存自己的Q片段，按需获取其他片段
                
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (tile == blockIdx.y  && thread_row + i < thread_col + j) {
                            tS[i][j] = -FLT_MAX;
                        } else {
                            tS[i][j] += rQ[i] * rK[j];
                        }

                        if (tile == blockIdx.y  && thread_row + i + 8 < thread_col + j) {
                            tS[i + 4][j] = -FLT_MAX;
                        } else {
                            tS[i + 4][j] += rQ[i + 4] * rK[j];
                        }

                        if (tile == blockIdx.y  && thread_row + i < thread_col + j + 64) {
                            tS[i][j+4] = -FLT_MAX;
                        } else {
                            tS[i][j+4] += rQ[i] * rK[j+4];
                        }

                        if (tile == blockIdx.y  && thread_row + i + 8 < thread_col + j + 64) {
                            tS[i+4][j+4] = -FLT_MAX;
                        } else {
                            tS[i+4][j+4] += rQ[i+4] * rK[j+4];
                        }
                    }
                }
            }
        }


        // rescale preatt by 1/sqrt(HS)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        //
        // compute m
        //

        // inter-thread reduction
        // 每个线程处理8行(tS[0-7][])×8列(tS[][0-7])的S矩阵分块
        // 对每行8个元素求最大值，结果存入rM[0-7] 存储在寄存器文件中，每个线程独立维护
        // rM_old[i]携带来自前一个tile的最大值
        for (int i = 0; i < 8; i++) {
            rM[i] = rM_old[i];// 计算每行的最大值m
            for (int j = 0; j < 8;j++) {
                rM[i] = fmaxf(rM[i], tS[i][j]);
            }
        }

        // inter-warp reduction
        for (int i=0; i < 8; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }
        /*
        操作分解：
            offset=8：比较当前线程与+8位置线程的值
            硬件行为：
            __shfl_down_sync：在warp内将当前线程的rM[i]值与offset距离的线程交换数据
            mask=0xFFFF或0xFFFF0000：限定在半个warp(16线程)内操作
        */

        // now threads 0, 16 have the correct m[i],
        // so we broadcast m back to the other lanes in the half warp
        for (int i=0; i<8; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }
        // 参数解析：lane_id_to_read_from：0（前16线程）或16（后16线程）
        // mask：保持与之前相同的half-warp划分
        // 物理意义：经过前一步归约后，只有lane_id=0和16的线程持有正确最大值
        // __shfl_sync(mask, var, src_lane)：所有在mask内的线程都会复制src_lane线程的var值
        // 硬件上通过warp内的寄存器网络直接复制，无内存访问

        //
        // compute P
        //
        for (int i=0;i<8;i++) {
            for (int j=0;j<8;j++){
                tP[i][j] = expf(tS[i][j] - rM[i]);
            }
        }

        //store to sP


        //
        // compute l
        //

        // rescale l and also reset rD to 0
        for (int i = 0; i < 8; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0;
        }

        // inter-thread reduction
        for (int i = 0; i < 8; i++) {
            for (int j=0;j<8;j++){
                rD[i] += tP[i][j];
            }
        }//一个线程处理的8列的总和

        // inter-warp reduction
        for (int i=0; i < 8; i++) {
            for (int offset = 8; offset > 0; offset /= 2) {
               rD[i] += __shfl_down_sync(mask, rD[i], offset);
            }
        }//16个线程跑一行的数据，所以是半个warp进行规约获得总的求和的值

        // now threads 0, 16 have the correct rD[i],
        // so we compute rL[i] and broadcast it back to the warp
        for (int i=0; i<8; i++) {
            rL[i] += rD[i];
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



        //
        // compute O
        //

        // first rescale O by exp(m_old - m)
        for (int i=0; i<8; i++) {
            for (int j=0;j<4;j++) {
                rO[i][j] = expf(rM_old[i] - rM[i]) * rO[i][j];
            }
        }//ro会复用，所以要对前面的进行处理

        // add PV to rO
        for (int step = 0; step < 2; step++) {
            // 将V矩阵分为上下两半	匹配共享内存容量
            for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
                for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                    // position is h * 64 + l * 4 + k
                    int k_fragment = k_fragment_outer * 4 + k_fragment_inner;//具体到了一行的第x个元素
                    FLOAT4(rV[0]) = FLOAT4(sV(step * 64 + k_fragment, thread_col));
                    for (int i=0;i<4;i++) {
                        rP[i] = __shfl_sync(mask, tP[i][k_fragment_inner + step * 4], (lane_id /16) * 16  + k_fragment_outer );
                        rP[i + 4] = __shfl_sync(mask, tP[i + 4][k_fragment_inner + step * 4], (lane_id /16) * 16  + k_fragment_outer);
                        //rV[i] = sV(step * 64 + k_fragment, thread_col + i);
                    }

                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 4; j++) {
                            rO[i][j] += rP[i] * rV[j];
                        }
                    }
                }
            }
        }


        // update m and l
        for (int i = 0; i < 8; i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        gK += tile_increment;
        gV += tile_increment;
        __syncthreads();
    }


    //rescale rO
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            rO[i][j] /= rL[i];
        }
    }


    // store l back to gL
    if (lane_id == 0 || lane_id == 16) {
        for (int i = 0; i < 4; i++) {
            gL(thread_row + i) = rM[i] + logf(rL[i]);
            gL(thread_row + 8 + i) = rM[i + 4] + logf(rL[i + 4]);
        }
    }

    // store rO to gO
    for (int i=0; i < 4; i++) {
        FLOAT4(gO(thread_row + i, thread_col)) = FLOAT4(rO[i][0]);
        FLOAT4(gO(thread_row + 8 + i, thread_col)) = FLOAT4(rO[i+4][0]);
    }

}

#undef sK_T

#define Q_TILE_SIZE 64
#define KV_TILE_SIZE 128
#define gdQ(i,j) gdQ[(i) * 3 * NH * HS + (j)]
#define gdK(i,j) gdK[(i) * 3 * NH * HS + (j)]
#define gdV(i,j) gdV[(i) * 3 * NH * HS + (j)]
#define gdO(i,j) gdO[(i) * 1 * NH * HS + (j)]
#define sK(i,j) sK[(i) * HEAD_SIZE + (j)]
#define sK_T(i,j) sK[(i) + (j) * HEAD_SIZE]
#define sQ_row(i,j) sQ[(i) * HEAD_SIZE + (j)]
#define sQ_col(i,j) sQ[(i) + (j) * Q_TILE_SIZE]
#define sdO_row(i,j) sdO[(i) * HEAD_SIZE + (j)]
#define sdO_col(i,j) sdO[(i) + (j) * Q_TILE_SIZE]
#define sdS(i,j) sdS[(i) + (j) * Q_TILE_SIZE]
#define sdQ(i,j) sdQ[(i) * HEAD_SIZE + (j)]

// preprocessing D = rowsum(dO * O)
__global__ void flash_attention_backward_preprocessing_kernel1(float* d, float* dout, float* out,
                                int B, int T, int NH, int HS) {
    // dout (B, T, NH, HS)
    // out (B, T, NH, HS)
    // d (B, T, NH)

    // blockDim.x = NH
    // blockDim.y = T / 256
    // blockDim.z = B

    // Each half-warps compute 4 rows,
    // so each warp computes 8 rows
    // We use 1024 threads = 32 warps per block, so each block computes 256 rows
    // so we have B * T / 256 * NH blocks

    int o_global_offset = blockIdx.z * T * NH * HS + blockIdx.y * 256 * NH * HS + blockIdx.x * HS;
    int d_global_offset = blockIdx.z * T * NH + blockIdx.y * 256 * NH + blockIdx.x;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves

    float* gO = &out[o_global_offset];
    float* gdO = &dout[o_global_offset];
    float* gD = &d[d_global_offset];

    int thread_row = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col = (lane_id % 16) * 4;

    float tO[4][4];
    float tdO[4][4];
    float sum[4] = {0.0f};

    for (int i=0;i<4;i++){
        FLOAT4(tO[i][0]) = FLOAT4(gO(thread_row + i, thread_col));
        FLOAT4(tdO[i][0]) = FLOAT4(gdO(thread_row + i, thread_col));
    }

    // inter-thread reduction
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4;j++) {
            sum[i] += tO[i][j] * tdO[i][j];
        }
    }

    // inter-warp reduction
    for (int i=0; i < 4; i++) {
        for (int offset = 8; offset > 0; offset /= 2) {
           sum[i] += __shfl_down_sync(mask, sum[i], offset);
        }
    }

    if (lane_id == 0 || lane_id == 16) {
        for (int i=0; i<4; i++) {
            gD(thread_row + i) = sum[i];
        }
    }
}


__global__ __launch_bounds__(256)
void flash_attention_backward_kernel1(float* dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int NH, int HS) {
    // dinp (B, T, 3, NH, HS)
    // inp  (B, T, 3, NH, HS)
    // out  (B, T, NH, HS)
    // dout (B, T, NH, HS)
    // l    (B, T, NH)
    // d    (B, T, NH)

    // blockDim.x = NH
    // blockDim.y = T
    // blockDim.z = B

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // offset for the q,k,v of the corresponding head
    int q_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * 2 * Q_TILE_SIZE * 3 * NH * HS + 0 * NH * HS + blockIdx.x * HS;
    int k_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * KV_TILE_SIZE * 3 * NH * HS + 1 * NH * HS + blockIdx.x * HS;
    int v_global_offset = blockIdx.z * T * 3 * NH * HS + blockIdx.y * KV_TILE_SIZE * 3 * NH * HS + 2 * NH * HS + blockIdx.x * HS;
    int o_global_offset = blockIdx.z * T * 1 * NH * HS + blockIdx.y * 2 * Q_TILE_SIZE * 1 * NH * HS + 0 * NH * HS + blockIdx.x * HS;

    int ld_global_offset = blockIdx.z * T * NH + blockIdx.y * 2 * Q_TILE_SIZE * NH + blockIdx.x;

    int q_increment = Q_TILE_SIZE * 3 * NH * HS;
    int o_increment = Q_TILE_SIZE * NH * HS;
    int ld_increment = Q_TILE_SIZE * NH;

    float* gQ = &inp[q_global_offset];
    float* gK = &inp[k_global_offset];
    float* gV = &inp[v_global_offset];
    float* gdO = &dout[o_global_offset];
    float* gL = &l[ld_global_offset];
    float* gD = &d[ld_global_offset];

    // output
    float* gdQ = &dinp[q_global_offset];
    float* gdK = &dinp[k_global_offset];
    float* gdV = &dinp[v_global_offset];

    extern __shared__ float sharedMemory[];

    float* sQ = &sharedMemory[0];
    float* sdO = sQ + Q_TILE_SIZE * Q_TILE_SIZE;
    float* sK = sdO + Q_TILE_SIZE * Q_TILE_SIZE;
    float* sdS = sQ;
    float* sdQ = sQ;

    // offset for register tiling for dK and dV
    int thread_row_128_x_64 = warp_id * 16 + (lane_id / 16) * 4;
    int thread_col_128_x_64 = (lane_id % 16) * 4;

    // offset for register tiling for S and dP
    int thread_row_64_x_128 = thread_col_128_x_64;
    int thread_col_64_x_128 = thread_row_128_x_64;

    // offset for register tiling for dQ
    int thread_row_64_x_64 = warp_id * 8 + (lane_id / 16) * 4;
    int thread_col_64_x_64 = (lane_id % 16) * 4;


    // offset for atomic add for dQ
    int thread_row_atomic_add = warp_id * 8;
    int thread_col_atomic_add = lane_id;

    unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves

    float rL[4];
    float rD[4];
    float rQ[4];
    float rK[8];
    float rV[8];
    float rdO[4];
    float rP[8];
    float rdS[8];
    float tV[8][4];
    float tQ[4][4];
    float tdO[4][4];
    float tdQ[4][4] = {0.0f};
    float tdK[8][4] = {0.0f};
    float tdV[8][4] = {0.0f};
    float tS[4][8] = {0.0f};
    float (&tP)[4][8] = tS;
    float tdP[4][8] = {0.0f};
    float (&tdS)[4][8] = tdP;

    for (int i=0; i < 4;i ++){
        FLOAT4(sK(thread_row_128_x_64 + i, thread_col_128_x_64)) = FLOAT4(gK(thread_row_128_x_64 + i, thread_col_128_x_64));
        FLOAT4(sK(thread_row_128_x_64 + 8 + i,  thread_col_128_x_64)) = FLOAT4(gK(thread_row_128_x_64 + 8 + i, thread_col_128_x_64));
        FLOAT4(tV[i][0]) = FLOAT4(gV(thread_row_128_x_64 + i, thread_col_128_x_64));
        FLOAT4(tV[i+4][0]) = FLOAT4(gV(thread_row_128_x_64 + 8 + i, thread_col_128_x_64));
    }

    for (int q_tile = 2 * blockIdx.y; q_tile < T / Q_TILE_SIZE; q_tile++) {

        // load Q, dO into shared memory
        for (int i=0; i < 4;i ++){
            FLOAT4(tQ[i][0]) = FLOAT4(gQ(thread_row_64_x_64 + i, thread_col_64_x_64));
            FLOAT4(tdO[i][0]) = FLOAT4(gdO(thread_row_64_x_64 + i, thread_col_64_x_64));
        }

        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                sQ_col(thread_row_64_x_64 + i, thread_col_64_x_64+j) = tQ[i][j];
                sdO_col(thread_row_64_x_64 + i, thread_col_64_x_64+j) = tdO[i][j];
            }
        }


        // load l, d into registers
        for (int i=0; i< 4;i ++){
            rL[i] = gL(thread_row_64_x_128 + i);
            rD[i] = gD(thread_row_64_x_128 + i);
        }

        __syncthreads();


        //
        // compute S and P
        //

        // reset tS back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tS[i][j] = 0;
            }
        }

        // compute S = Q * K^T
        for (int k_fragment = 0; k_fragment < HEAD_SIZE; k_fragment++) {
            FLOAT4(rQ[0]) = FLOAT4(sQ_col(thread_row_64_x_128, k_fragment));
            for (int i = 0; i < 4; i++) {
                rK[i] = sK_T(k_fragment, thread_col_64_x_128 + i);
                rK[i+4] = sK_T(k_fragment, thread_col_64_x_128 + 8 + i);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + j) {
                        tS[i][j] = -FLT_MAX;
                    } else {
                        tS[i][j] += rQ[i] * rK[j];
                    }

                    if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + 8 + j) {
                        tS[i][j + 4] = -FLT_MAX;
                    } else {
                        tS[i][j + 4] += rQ[i] * rK[j + 4];
                    }
                }
            }
        }


        // rescale S by 1/sqrt(HS)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                if (tS[i][j] != -FLT_MAX) {
                    tS[i][j] *= 1.0f / sqrtf(HS);
                }
            }
        }

        // compute P = exp(Q * K^T - l)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
               tP[i][j] = expf(tS[i][j] - rL[i]);
            }
        }

        //
        // compute dP and dS
        //

        // reset tdP back to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tdP[i][j] = 0;
            }
        }

        // compute dP = dO * V^T
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                FLOAT4(rdO[0]) = FLOAT4(sdO_col(thread_row_64_x_128, k_fragment));
                for (int i = 0; i < 4; i++) {
                    rV[i] = __shfl_sync(mask, tV[i][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                    rV[i+4] = __shfl_sync(mask, tV[i+4][k_fragment_inner], (lane_id / 16) * 16  + k_fragment_outer);
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + j) {
                            tdP[i][j] = 0;
                        } else {
                            tdP[i][j] += rdO[i] * rV[j];
                        }
                    }
                    for (int j = 0; j < 4; j++) {
                        if (q_tile * Q_TILE_SIZE + thread_row_64_x_128 + i < blockIdx.y * KV_TILE_SIZE + thread_col_64_x_128 + 8 + j) {
                            tdP[i][j+4] = 0;
                        } else {
                            tdP[i][j+4] += rdO[i] * rV[j+4];
                        }
                    }

                }
            }
        }


        // compute dS = P \circ (dP - D)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                tdS[i][j] = tP[i][j] * (tdP[i][j] - rD[i]);
            }
        }

        //
        // retile Q and dO to minimize bank conflicts
        //

        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                tQ[i][j] = sQ_col(thread_row_64_x_64 + i, thread_col_64_x_64+j);
                tdO[i][j] = sdO_col(thread_row_64_x_64 + i, thread_col_64_x_64+j);
            }
        }
        __syncthreads();

        for (int i=0;i<4;i++) {
            FLOAT4(sQ_row(thread_row_64_x_64 + i, thread_col_64_x_64)) = FLOAT4(tQ[i][0]);
            FLOAT4(sdO_row(thread_row_64_x_64 + i, thread_col_64_x_64)) = FLOAT4(tdO[i][0]);
        }
        __syncthreads();

        //
        //  compute dV
        //

        // compute dV = P^T * dO
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rP[i] = __shfl_sync(mask, tP[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rP[i+4] = __shfl_sync(mask, tP[k_fragment_inner][i + 4], (lane_id / 16) * 16  + k_fragment_outer);
                }
                FLOAT4(rdO[0]) = FLOAT4(sdO_row(k_fragment, thread_col_128_x_64 ));
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 4; j++) {
                        tdV[i][j] += rP[i] * rdO[j];
                    }
                }
            }
        }

        //
        // dK
        //

        // compute dK = dS^T * Q
        for (int k_fragment_outer = 0; k_fragment_outer < 16; k_fragment_outer++) {
            for (int k_fragment_inner = 0; k_fragment_inner < 4; k_fragment_inner++) {
                // position is k_fragment_outer * 4 + k_fragment_inner
                int k_fragment = k_fragment_outer * 4 + k_fragment_inner;
                for (int i = 0; i < 4; i++) {
                    rdS[i] = __shfl_sync(mask, tdS[k_fragment_inner][i], (lane_id / 16) * 16  + k_fragment_outer);
                    rdS[i+4] = __shfl_sync(mask, tdS[k_fragment_inner][i + 4], (lane_id / 16) * 16  + k_fragment_outer);
                }
                FLOAT4(rQ[0]) = FLOAT4(sQ_row(k_fragment, thread_col_128_x_64 ));

                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 4; j++) {
                        tdK[i][j] += rdS[i] * rQ[j];
                    }
                }
            }
        }


        __syncthreads();

        //
        // compute dQ
        //

        // reset tdQ back to zero
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] = 0;
            }
        }

        // store dS to shared memory
        for (int i = 0; i< 4; i++) {
            for (int j=0; j < 4;j++) {
                sdS(thread_row_64_x_128 + i, thread_col_64_x_128 + j) = tdS[i][j];
                sdS(thread_row_64_x_128 + i, thread_col_64_x_128 + j + 8) = tdS[i][j + 4];
            }
        }

        __syncthreads();


        //compute dQ = dS * K
        for (int k_fragment = 0; k_fragment < KV_TILE_SIZE; k_fragment++) {

            for (int i=0;i<4;i++) {
                //rdS[i] = sdS(thread_row_64_x_64 + i, k_fragment);
                //rK[i] = sK(k_fragment, thread_col_64_x_64 + i);
                FLOAT4(rdS[0]) = FLOAT4(sdS(thread_row_64_x_64, k_fragment));
                FLOAT4(rK[0]) = FLOAT4(sK(k_fragment, thread_col_64_x_64));
            }

            for (int i=0;i<4;i++) {
                for (int j=0; j<4; j++) {
                    tdQ[i][j] += rdS[i] * rK[j];
                }
            }
        }

        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                tdQ[i][j] *= 1.0f / sqrtf(HS);
            }
        }

        __syncthreads();

        // store dQ
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                sdQ(thread_row_64_x_64 + i, thread_col_64_x_64 + j) = tdQ[i][j];
            }
        }
        __syncthreads();

        for (int i=0;i<8;i++) {
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add ), sdQ(thread_row_atomic_add + i, thread_col_atomic_add));
            atomicAdd(&gdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32), sdQ(thread_row_atomic_add + i, thread_col_atomic_add + 32));
        }

        gQ += q_increment;
        gdQ += q_increment;
        gdO += o_increment;
        gL += ld_increment;
        gD += ld_increment;
        __syncthreads();
    }

    // rescale dK
    for (int i=0;i<8;i++) {
        for (int j=0; j<4; j++) {
            tdK[i][j] *= 1.0f / sqrtf(HS);
        }
    }

    // store dK to global memory

    for (int i=0;i<4;i++) {
        FLOAT4(gdK(thread_row_128_x_64 + i ,thread_col_128_x_64)) = FLOAT4(tdK[i][0]);
        FLOAT4(gdK(thread_row_128_x_64 + 8 + i ,thread_col_128_x_64)) = FLOAT4(tdK[i+4][0]);
    }


    // store dV to global memory
    for (int i=0;i<4;i++) {
        FLOAT4(gdV(thread_row_128_x_64 + i ,thread_col_128_x_64)) = FLOAT4(tdV[i][0]);
        FLOAT4(gdV(thread_row_128_x_64 + 8 + i ,thread_col_128_x_64)) = FLOAT4(tdV[i+4][0]);
    }
}



// ----------------------------------------------------------------------------
// kernel launchers


// use att to store log l + m
void flash_attention_forward(float* out, float* inp, float* l,
                                int B, int T, int C, int NH) {
    // head size
    int HS = C / NH;

    // inp (B, T, 3, NH, HS)
    // out (B, T, NH, HS)
    // l (B, T, NH)

    dim3 dimGrid(NH, T / 128, B);
    dim3 dimBlock(256);
    int maxbytes = 65536;
    cudaFuncSetAttribute(flash_attention_forward_kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    flash_attention_forward_kernel1<<<dimGrid, dimBlock, maxbytes>>>(out, inp, l, B, T, NH, HS);
    cudaGetLastError();
}


void flash_attention_backward(float *dinp, float* inp, float* dout, float* out, float* l, float* d,
                                int B, int T, int C, int NH) {

    // head size
    int HS = C / NH;

    // dinp (B, T, 3, NH, HS)
    // inp (B, T, 3, NH, HS)
    // dout (B, T, NH, HS)
    // out (B, T, NH, HS)
    // l (B, T, NH)
    // d (B, T, NH)

    // preprocess d = rowsum(dout * out)
    dim3 dimGrid_preprocessing(NH, T / 256, B);
    dim3 dimBlock_preprocessing(1024);
    flash_attention_backward_preprocessing_kernel1<<<dimGrid_preprocessing, dimBlock_preprocessing>>>(d, dout, out, B, T, NH, HS);

    dim3 dimGrid(NH, T / 128, B);
    dim3 dimBlock(256);
    int maxbytes = 65536;
    cudaFuncSetAttribute(flash_attention_backward_kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    flash_attention_backward_kernel1<<<dimGrid, dimBlock, maxbytes>>>(dinp, inp, dout, out, l, d, B, T, NH, HS);

    cudaGetLastError();
}
int main(){
    int B=10;
    int T=1280; 
    int C=512; 
    int NH =128;
    size_t N=B*T*3*C;
    float* input = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    float* output=(float*)malloc(B*T*C* sizeof(float));
    float* out_l=(float*)malloc(B*T*C* sizeof(float));
    

    // move to GPU
    float* d_out;
    float* d_l;
    float* d_inp;
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_l, B * T * C * sizeof(float));
    cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float));
    cudaMemcpy(d_inp, input, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice);
    flash_attention_forward(d_out,d_inp,d_l,B,T,C,NH);
}