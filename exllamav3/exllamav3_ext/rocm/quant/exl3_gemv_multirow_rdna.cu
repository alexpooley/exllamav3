// =============================================================================
// Multi-row GEMV for RDNA, m = 2..8
// =============================================================================
//
// The GEMV fast paths take m == 1 only; at m = 2..8 -- speculative-decoding
// verify steps (MTP, DFlash, n-gram drafts) -- every dense linear used to fall
// to the cooperative tile GEMM, which pads the rows to a 16-row tile and walks
// the weights at a fraction of the GEMV's bandwidth, making the dense linears
// the largest kernel class of a verify step.
//
// This path reads the weights once for up to 8 rows. The dot core is the
// barrier-free m == 1 core (exl3_gemv_dot_tile_direct) with a row tile M in
// {2, 4, 8} chosen as the smallest >= m: one B dequant per k-tile feeds M
// pairs of fdot2 accumulators, and each row's chain is exactly the m == 1
// chain, so row r of an m-row call is bit-identical to a separate m == 1 call
// on that row. Rows past m are
// computed on row m - 1's data and never stored.
//
// Structure is the pre-fusion three-kernel form, deliberately. The fused
// m == 1 kernels rotate the input into LDS in their prologue, which does not
// extend to m rows on a 64 KB-per-workgroup part (8 x 12288 halves is 196 KB,
// and the rotation is redundant across a matrix's N-tile blocks). So:
//
//   1. exl3_gemv_mr_had_in_{single,multi}   rotates the m rows of every slab
//      into A_had (global; L2-resident thereafter) and hosts the graph patch
//      sites, republishing the patchable pointers the dot kernel needs through
//      a per-device Exl3GemvMrParams block (the pattern of exl3_mgemv_rdna.cu)
//   2. exl3_gemv_mr_dot_{single,multi}      split-K GEMV, one block per
//      (slot, N-tile), M rows per warp; the fused output epilogue (last-
//      arriving warp per 128-wide segment, exl3_gemv_kernel_rdna.cuh)
//      rotates the m row segments in place
//
// Two launches per call at m > 1 is irrelevant: the verify step is GPU-bound.
// A part with more LDS could rotate the rows in-kernel (the
// helpers are shared), gated on a runtime LDS budget; not done here.
//
// Routing: exl3_gemm_gr and exl3_mgemm_gr call the two try_launch entry
// points for 2 <= m <= EXL3_GEMV_MAX_M (default 8; 1 switches the path off)
// before the cooperative kernels, in and out of graph capture. The
// multi-matrix form declines sliced mode.
//
// Routing weights (the MoE down projection's weighted reduce):
// at m == 1 the multi-matrix form takes weighted calls too, with the
// exl3_mgemv fused epilogue verbatim (exl3_gemv_fused_epilogue: rotate the
// segment with scale * weights[orig_pos], then the last rotated slot runs the
// grouped num_tokens reduction in exl3_mgemv_reduce_kernel's order), so the
// output is bit-identical to the LDS-prologue exl3_mgemv_dot_kernel_splitk it
// replaces there -- the dot core, wave count and N-tiles per block are the
// same rule on the same shape; only where the rotated input lives differs
// (A_had in L2 instead of per-block LDS).
// EXL3_ROCM_MR_WEIGHTED=0 restores the decline (weighted calls go back to
// exl3_mgemv_try_launch). num_tokens > 1 is accepted without range packing
// (min_index < 0), the case the grouped reduction divides evenly.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_mgemv_rdna.cuh"
#include "exl3_gemv_kernel_rdna.cuh"
#include "exl3_gemv_multirow_rdna.cuh"

#include <cstdlib>
#include <mutex>

int exl3_gemv_max_m()
{
    const char* env = std::getenv("EXL3_GEMV_MAX_M");
    if (!env) return 8;
    int v = atoi(env);
    return v < 1 ? 1 : (v > 8 ? 8 : v);
}

// m == 1 goes through this structure too, with a row tile of 1 -- rows read
// from L2, no rotated row in LDS -- ahead of the fused m == 1 kernels
// (EXL3_GEMV_MR_M1=0 restores those): the fused kernels' LDS prologue costs
// occupancy, which outweighs the saved launch. Logits are bit-identical
// either way. The fused kernels keep the weighted-reduce MoE down projection
// (this path declines weights) and lm_head-scale widths (the m == 1 single-warp form; leaving them there
// keeps the two routes bit-identical where both split K).
static bool exl3_gemv_mr_m1_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_MR_M1");
    if (!env) return true;
    return atoi(env) != 0;
}

static bool exl3_gemv_mr_weighted_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_ROCM_MR_WEIGHTED");
        return !env || atoi(env) != 0;
    }();
    return on;
}

static inline int exl3_gemv_mr_min_m(int n_tiles)
{
    return (exl3_gemv_mr_m1_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES) ? 1 : 2;
}

// Half-integer bitrates (EXL3_HALF_BITS(K), K = 1..3, mul1): the tiles core's
// dq8_half decoder (exl3_gemv_tiles_rdna.cuh) on the same bodies. Callers
// pass the pseudo width only for half-rate weights and only while this is on;
// re-read per call (half-rate calls only -- integer calls never reach it).
bool exl3_rocm_half_gemv_enabled()
{
    const char* env = std::getenv("EXL3_ROCM_HALF_GEMV");
    return !env || atoi(env) != 0;
}

// Widths this file dispatches: integer 1..8 (any codebook), half 1.5 / 2.5 / 3.5 (mul1)
static inline bool exl3_gemv_mr_k_ok(int K, int cb)
{
    if (K >= 1 && K <= 8) return true;
    return K >= EXL3_HALF_BITS(1) && K <= EXL3_HALF_BITS(3) && cb == 2;
}

// -----------------------------------------------------------------------------
// Per-device parameter block: the patchable pointers the dot kernels read
// (republished by the rotation kernels, which host the graph sites) and the
// fused epilogue's arrival counters, [slot][segment], self-resetting.
// -----------------------------------------------------------------------------
struct Exl3GemvMrParams
{
    const uint16_t* B;      // single-matrix
    void* C;
    half* A_had;
    const half* svh;
    void* Cm;               // multi-matrix
    const int64_t* indices;
    const half* weights;
    int seg_counters[EXL3_MGEMV_SEG_COUNTERS];
    int red_counters[EXL3_MGEMV_RED_COUNTERS];   // weighted reduce, per output segment
};

static Exl3GemvMrParams* exl3_gemv_mr_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3GemvMrParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        if (cudaMemset(p, 0, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cudaFree(p);
            return nullptr;
        }
        blocks[device] = (Exl3GemvMrParams*) p;
    }
    return blocks[device];
}

void exl3_gemv_multirow_prewarm(int device)
{
    (void) exl3_gemv_mr_param_block(device, true);
}

// =============================================================================
// Multi-row dot core: exl3_gemv_dot_tile_direct with M rows per warp
// =============================================================================
// Same B path, same per-row fdot2 chain and the same quad reduction as the
// m == 1 core, so each row is bit-identical to it. A is the slab's rotated
// rows, lda halves apart, rows_valid of them (<= M): rows past that re-read
// the last valid row rather than memory beyond the slab (their outputs are
// discarded). out[r] is row r's column-lane value (lanes 0-15).

template <int bits, int cb, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_direct_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int n_tiles,
    const int tile_n,
    const int lane,
    const int kb_begin,
    const int kb_end,
    float* out,
    const int rows_valid = M
)
{
    constexpr int tile_elements = Exl3Width<bits>::tile_u16;   // 16 * bits (integer K)
    const int r0 = (lane & 3) * 2;
    const int row_last = rows_valid - 1;
    int a_row[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) a_row[r] = (r < row_last ? r : row_last) * lda;

    float accA[M], accB[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }

    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
    {
        const uint32_t* b_ptr = (const uint32_t*)
            (B + (k_tile * n_tiles + tile_n) * tile_elements);

        FragB frag0, frag1;
        dq_dispatch<Exl3Width<bits>::ka, cb, Exl3Width<bits>::half>(b_ptr, lane << 3, frag0, frag1);

        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const half2* a2 = (const half2*) (A + a_row[r] + k_tile * 16);
            half2 a01 = a2[r0 >> 1];
            half2 a89 = a2[(r0 >> 1) + 4];
            accA[r] = exl3_fdot2(a01, frag0[0], accA[r]);
            accA[r] = exl3_fdot2(a89, frag0[1], accA[r]);
            accB[r] = exl3_fdot2(a01, frag1[0], accB[r]);
            accB[r] = exl3_fdot2(a89, frag1[1], accB[r]);
        }
    }

    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// Split-K form: the m == 1 chunking and reduction order (exl3_gemv_dot_tile_
// splitk), M rows wide. sh_red holds WARPS_PER_BLOCK * M * 16 floats. Every
// thread of the block enters; out is meaningful for warp 0, lanes 0-15.
template <int bits, int cb, int WARPS_PER_BLOCK, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_splitk_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles,
    const int tile_n,
    const int warp_id,
    const int lane,
    float* sh_red,
    float* out
)
{
    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int kb0 = warp_id * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;

    float acc[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) acc[r] = 0.0f;
    if (kb0 < kb1)
        exl3_gemv_dot_tile_direct_mr<bits, cb, M>(A, lda, B, n_tiles, tile_n, lane, kb0, kb1, acc);

    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r) sh_red[(warp_id * M + r) * 16 + lane] = acc[r];
    }
    __syncthreads();

    if (warp_id == 0 && lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w)
                total += sh_red[(w * M + r) * 16 + lane];
            out[r] = total;
        }
    }
}

static inline size_t exl3_gemv_mr_smem_bytes(int warps, int M)
{
    return (size_t) warps * M * EXL3_GEMV_TILES_TMAX * 16 * sizeof(float);
}

// =============================================================================
// Multi-row fused output epilogue: the m == 1 handshake, m row segments
// =============================================================================
// Entered by the warp holding the tile's M rows in acc (lanes 0-15); stores
// rows < rows_valid, arrives once per tile, and as the last of the segment's
// 8 tiles rotates each valid row's segment in place with the shared helper --
// per row the arithmetic of the m == 1 epilogue exactly.

template <bool c_fp32, int M>
__device__ __forceinline__ void exl3_gemv_fused_epilogue_mr
(
    const float* acc,
    void* Cb,
    const int64_t row_off,       // slab base within Cb (j * m * n, or 0 for list outputs)
    const int ldc,               // row stride of C
    const int rows_valid,        // m (<= M)
    const int tile_n,
    const int lane,
    const half* __restrict__ svh,
    int* seg_counter
)
{
    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (r >= rows_valid) break;
            const int64_t out_idx = row_off + (int64_t) r * ldc + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) Cb)[out_idx] = acc[r];
            else
                ((half*) Cb)[out_idx] = __float2half(acc[r]);
        }
    }

    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;

    __threadfence();
    if (lane == 0) *seg_counter = 0;
    const int seg = tile_n / 8;
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        if (r >= rows_valid) break;
        const int64_t seg_off = row_off + (int64_t) r * ldc + (int64_t) seg * 128;
        if constexpr (c_fp32)
            exl3_gemv_had_out_128_f(((float*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
        else
            exl3_gemv_had_out_128_h(((half*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
    }
}

// =============================================================================
// Kernel 1: input rotation, m rows per slab, and the graph patch sites
// =============================================================================
// One warp per (slab, row, 128-block). The single-matrix form hosts the six
// GP_gemm_* sites (argument order 0-5 load-bearing; B, C, svh are patch hosts
// only) and the multi-matrix form the four GP_mgemm_* sites (0-3; weights is a
// patch host only -- the path declines weighted calls).

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_single
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis (patch host only)
    void* __restrict__ C,              // 2: GP_gemm_C (patch host only)
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_m
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= size_m * k_segs) return;
    const int row = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int64_t off = (int64_t) row * size_k + seg * 128;
    exl3_gemv_had_in_128(A + off, A_had + off, suh + seg * 128, lane);
}

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_multi
(
    const half* __restrict__ A,          // 0: GP_mgemm_A
    void* __restrict__ C,                // 1: GP_mgemm_C (patch host only)
    const int64_t* __restrict__ indices, // 2: GP_mgemm_indices
    const half* __restrict__ weights,    // 3: GP_mgemm_weights (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ suh_list,
    half* __restrict__ A_had,
    const int size_k,
    const int size_m,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->Cm = C;
        pb->indices = indices;
        pb->weights = weights;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= bszm * size_m * k_segs) return;
    const int j   = warp_id / (size_m * k_segs);
    const int row = (warp_id / k_segs) % size_m;
    const int seg = warp_id % k_segs;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    const half* suh = (const half*) suh_list[mat_index];
    // A slab j (or the broadcast input), row `row`; A_had slabs follow packed slot numbering
    const half* A_src = (bszm_in == 1 ? A : A + (int64_t) j * size_m * size_k)
                        + (int64_t) row * size_k + seg * 128;
    half* dst = A_had + ((int64_t) j * size_m + row) * size_k + seg * 128;
    exl3_gemv_had_in_128(A_src, dst, suh + seg * 128, lane);
}

// Weighted form, skipped slot: exl3_gemv_fused_epilogue with a zero tile and no
// rotation (the whole segment is zero, and so is its rotation), then the same
// reduction arrival -- and, as the last rotated slot, the same grouped sum.
template <bool c_fp32>
__device__ __forceinline__ void exl3_gemv_mr_zero_epilogue
(
    void* Cb,
    const int64_t row_off,
    const int size_n,
    const int tile_n,
    const int lane,
    int* seg_counter,
    int* red_counter,
    const int red_target,
    const int num_tokens
)
{
    if (lane < 16)
    {
        const int64_t out_idx = row_off + tile_n * 16 + lane;
        if constexpr (c_fp32) ((float*) Cb)[out_idx] = 0.0f;
        else                  ((half*) Cb)[out_idx] = __float2half(0.0f);
    }
    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;
    __threadfence();
    if (lane == 0) *seg_counter = 0;

    __threadfence();
    if (lane == 0) old = atomicAdd(red_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != red_target - 1) return;
    __threadfence();
    if (lane == 0) *red_counter = 0;
    const int seg = tile_n / 8;
    const int stride = red_target / num_tokens;
    const int col0 = seg * 128 + lane;
    for (int t = 0; t < num_tokens; ++t)
    {
        if constexpr (c_fp32)
        {
            const float* C_ = ((const float*) Cb) + (int64_t) t * stride * size_n + col0;
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] += C_[c * 32];
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((float*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
        else
        {
            const half* C_ = ((const half*) Cb) + (int64_t) t * stride * size_n + col0;
            half sum[4] = {};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] = __hadd(sum[c], C_[c * 32]);
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((half*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
    }
}

// =============================================================================
// Kernel 2: the split-K GEMV, M rows per warp, fused output epilogue
// =============================================================================
// Grid (n_tiles, bszm); one block per (slot, N-tile). Both forms share the body.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
__device__ __forceinline__ void exl3_gemv_mr_body
(
    const half* __restrict__ A,     // this slab's rotated rows
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles_j,
    const int tile_n,               // first of the block's tpb tiles
    void* Cb,
    const int64_t row_off,
    const int ldc,
    const int rows_valid,
    const half* __restrict__ svh,
    int* seg_counters,              // this slot's counters, indexed by segment
    const int warp_id,
    const int lane,
    const int core,
    const int tpb,
    const float w_scale = 0.0f,        // weighted form (M == 1): 1/sqrt(128) * weights[orig_pos]
    int* red_counters = nullptr,       // weighted form: per-segment rotated-slot counters
    const int red_target = 0,          // weighted form: packed slot count
    const int size_n = 0,              // weighted form: row stride of C (reduction)
    const int num_tokens = 1
)
{
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;

    // tpb adjacent N-tiles per block with the tiles core (1 otherwise); each
    // tile's rows keep the one-tile chain and reduction order
    constexpr int T = EXL3_GEMV_TILES_TMAX;
    float acc[T * M];
    exl3_gemv_dot_tile_splitk_t<bits, cb, WARPS_PER_BLOCK, M, false, T>
    (
        core, tpb, A, lda, B, size_k, n_tiles_j, tile_n, warp_id, lane, sh_red, acc,
        [&](int tn, int k0, int k1, float* o)
        {
            exl3_gemv_dot_tile_direct_mr<bits, cb, M>(A, lda, B, n_tiles_j, tn, lane, k0, k1, o, rows_valid);
        },
        rows_valid
    );
    if (warp_id != 0) return;
    if constexpr (M == 1)
    {
        if (red_counters)
        {
            // Weighted reduce: exactly exl3_mgemv_store_or_fuse's fused call
            for (int t = 0; t < tpb; ++t)
                exl3_gemv_fused_epilogue<c_fp32>
                (
                    acc[t], Cb, row_off, size_n, tile_n + t, lane, svh, w_scale,
                    seg_counters + (tile_n + t) / 8, red_counters + (tile_n + t) / 8,
                    red_target, num_tokens
                );
            return;
        }
    }
    for (int t = 0; t < tpb; ++t)
        exl3_gemv_fused_epilogue_mr<c_fp32, M>
        (
            acc + t * M, Cb, row_off, ldc, rows_valid, tile_n + t, lane, svh,
            seg_counters + (tile_n + t) / 8
        );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_single
(
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int size_m,
    const int core,
    const int tpb
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        pb->A_had, size_k, pb->B, size_k, n_tiles, tile_n,
        pb->C, 0, size_n, size_m, pb->svh,
        pb->seg_counters, warp_id, lane, core, tpb
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_multi
(
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ svh_list,
    const half* __restrict__ A_had,
    const int size_k,
    const int size_n,
    const int size_m,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (m rows, stride n_j)
    const int core,
    const int tpb,
    const int num_tokens,                 // weighted form: reduction groups (0: unweighted)
    const int weighted
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(pb->indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0)   // block-uniform
    {
        // Weighted form: a skipped slot (negative index, no packing) still has to arrive
        // at the grouped reduce, as an all-zero row -- adding +0 leaves every sum exactly
        // what skipping the row gives, which is the cooperative kernel's semantics
        if (M == 1 && weighted && min_index < 0 && warp_id == 0)
        {
            const int n_segs0 = size_n / 128;
            const int red_target = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
            for (int t = 0; t < tpb; ++t)
                exl3_gemv_mr_zero_epilogue<c_fp32>
                (
                    pb->Cm, (int64_t) j * size_n, size_n, tile_n + t, lane,
                    pb->seg_counters + j * n_segs0 + (tile_n + t) / 8,
                    pb->red_counters + (tile_n + t) / 8, red_target, num_tokens
                );
        }
        return;
    }
    const int n_j = size_n_list ? size_n_list[mat_index] : size_n;
    const int n_tiles_j = n_j / 16;
    if (tile_n >= n_tiles_j) return;   // block-uniform

    void* Cb = c_list ? c_list[mat_index] : pb->Cm;
    const int64_t row_off = c_list ? 0 : (int64_t) j * size_m * size_n;
    const int ldc = c_list ? n_j : size_n;
    const int n_segs = size_n / 128;
    if (M == 1 && weighted)   // launch-uniform; size_m == 1, no lists (entry point)
    {
        const half* weights = pb->weights;
        const float w_scale = 0.088388347648f * __half2float(weights[orig_pos]);
        const int red_target = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
        (
            A_had + (int64_t) j * size_m * size_k, size_k,
            (const uint16_t*) B_list[mat_index], size_k, n_tiles_j, tile_n,
            Cb, row_off, ldc, size_m, (const half*) svh_list[mat_index],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb,
            w_scale, pb->red_counters, red_target, size_n, num_tokens
        );
        return;
    }
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        A_had + (int64_t) j * size_m * size_k, size_k,
        (const uint16_t*) B_list[mat_index], size_k, n_tiles_j, tile_n,
        Cb, row_off, ldc, size_m, (const half*) svh_list[mat_index],
        pb->seg_counters + j * n_segs, warp_id, lane, core, tpb
    );
}

// =============================================================================
// Host dispatch
// =============================================================================

static inline int exl3_gemv_mr_row_tile(int size_m)
{
    return size_m <= 1 ? 1 : (size_m <= 2 ? 2 : (size_m <= 4 ? 4 : 8));
}

// Split-K wave count: the m == 1 rule (shape-aware; EXL3_GEMV_SPLITK_WARPS
// overrides), which keeps row r of an m-row call bit-identical to the m == 1
// split-K result for the same shape. Above the m == 1 single-warp threshold
// (lm_head-scale outputs) the m == 1 path uses one warp per tile; here the
// block count alone saturates the device, so the narrowest split is used.
static inline int exl3_gemv_mr_warps(int size_k, int n_tiles, int bszm)
{
    if (n_tiles > EXL3_GEMV_SPLITK_MAX_TILES) return 4;
    return exl3_gemv_splitk_warps(size_k / 16, n_tiles, bszm);
}

#define LAUNCH_MR_M(bits_val, codebook, warps_val, m_tile, kernel_name, ...) \
    if (c_fp32) { \
        hipLaunchKernelGGL((kernel_name<bits_val, true, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
    } else { \
        hipLaunchKernelGGL((kernel_name<bits_val, false, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
    }
#define LAUNCH_MR_W(bits_val, codebook, warps_val, kernel_name, ...) \
    switch (m_tile) { \
        case 1: LAUNCH_MR_M(bits_val, codebook, warps_val, 1, kernel_name, __VA_ARGS__) break; \
        case 2: LAUNCH_MR_M(bits_val, codebook, warps_val, 2, kernel_name, __VA_ARGS__) break; \
        case 4: LAUNCH_MR_M(bits_val, codebook, warps_val, 4, kernel_name, __VA_ARGS__) break; \
        case 8: LAUNCH_MR_M(bits_val, codebook, warps_val, 8, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_CB(bits_val, codebook, kernel_name, ...) \
    switch (warps) { \
        case 4:  LAUNCH_MR_W(bits_val, codebook, 4,  kernel_name, __VA_ARGS__) break; \
        case 8:  LAUNCH_MR_W(bits_val, codebook, 8,  kernel_name, __VA_ARGS__) break; \
        case 16: LAUNCH_MR_W(bits_val, codebook, 16, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_K(bits_val, kernel_name, ...) \
    switch (cb) { \
        case 0: LAUNCH_MR_CB(bits_val, 0, kernel_name, __VA_ARGS__); break; \
        case 1: LAUNCH_MR_CB(bits_val, 1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_CB(bits_val, 2, kernel_name, __VA_ARGS__); break; \
    }
#define LAUNCH_MR(kernel_name, ...) \
    switch (K) { \
        case 1: LAUNCH_MR_K(1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_K(2, kernel_name, __VA_ARGS__); break; \
        case 3: LAUNCH_MR_K(3, kernel_name, __VA_ARGS__); break; \
        case 4: LAUNCH_MR_K(4, kernel_name, __VA_ARGS__); break; \
        case 5: LAUNCH_MR_K(5, kernel_name, __VA_ARGS__); break; \
        case 6: LAUNCH_MR_K(6, kernel_name, __VA_ARGS__); break; \
        case 7: LAUNCH_MR_K(7, kernel_name, __VA_ARGS__); break; \
        case 8: LAUNCH_MR_K(8, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(1): LAUNCH_MR_CB(EXL3_HALF_BITS(1), 2, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(2): LAUNCH_MR_CB(EXL3_HALF_BITS(2), 2, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(3): LAUNCH_MR_CB(EXL3_HALF_BITS(3), 2, kernel_name, __VA_ARGS__); break; \
    }

bool exl3_gemv_multirow_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (!exl3_gemv_mr_k_ok(K, cb)) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (!suh_ptr || !A_had_ptr || !svh_ptr) return false;
    if (size_n / 128 > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_single,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, size_k, size_m
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_single;
            graph->record_param(k, GP_gemm_A, 0);
            graph->record_param(k, GP_gemm_B_trellis, 1);
            graph->record_param(k, GP_gemm_C, 2);
            graph->record_param(k, GP_gemm_B_suh, 3);
            graph->record_param(k, GP_gemm_A_had, 4);
            graph->record_param(k, GP_gemm_B_svh, 5);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV, M rows per warp, fused output epilogue
    {
        const int n_tiles = size_n / 16;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, 1);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        const int core = exl3_gemv_core_mode();
        const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, 1, warps);
        dim3 grid(n_tiles / tpb, 1);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        LAUNCH_MR(exl3_gemv_mr_dot_single, pb, size_k, size_n, size_m, core, tpb)
    }
    return true;
}

bool exl3_mgemv_multirow_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,
    void** c_list,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (!exl3_gemv_mr_k_ok(K, cb)) return false;
    if (size_k % 128 || size_n % 128) return false;
    // Weighted calls (MoE down) at m == 1 with the mgemv epilogue; see the header
    const bool weighted = weights_ptr != nullptr;
    if (weighted && (size_m != 1 || !exl3_gemv_mr_weighted_enabled())) return false;
    if (weighted)
    {
        // The weighted form stands in for exl3_mgemv_try_launch, so it honors that
        // path's kill switch (EXL3_MGEMV=0 pins the cooperative kernel)
        const char* mg = std::getenv("EXL3_MGEMV");
        if (mg && atoi(mg) == 0) return false;
    }
    if (num_tokens < 1 || (num_tokens != 1 && min_index >= 0)) return false;
    if (weighted && size_n_list) return false;
    if (weighted && (!exl3_gemv_fuse_out_enabled() || size_n / 128 > EXL3_MGEMV_RED_COUNTERS)) return false;
    if ((size_n_list != nullptr) != (c_list != nullptr)) return false;
    if (size_n_list && min_index >= 0) return false;
    if (!suh_ptr_ptr || !A_had_ptr || !svh_ptr_ptr) return false;
    int bszm = MAX(bszm_in, bszm_out);
    if (bszm < 1 || bszm > 128) return false;
    if (bszm * (size_n / 128) > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(bszm * size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_multi,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, C_ptr, indices_ptr, weights_ptr, pb, suh_ptr_ptr, A_had_ptr,
            size_k, size_m, bszm_in, bszm, min_index, max_index
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_multi;
            graph->record_param(k, GP_mgemm_A, 0);
            graph->record_param(k, GP_mgemm_C, 1);
            graph->record_param(k, GP_mgemm_indices, 2);
            graph->record_param(k, GP_mgemm_weights, 3);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV
    {
        const int n_tiles = size_n / 16;
        // Multi-token calls (num_tokens groups of slots, the batched MoE route) size
        // the split-K from one token's slot count, so every slot keeps the reduction
        // order of a single-token call on the same experts (bit-identical rows)
        const int bszm_rule = (num_tokens > 1 && bszm % num_tokens == 0) ? bszm / num_tokens : bszm;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, bszm_rule);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        const int core = exl3_gemv_core_mode();
        const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, bszm, warps);
        dim3 grid(n_tiles / tpb, bszm);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        LAUNCH_MR(exl3_gemv_mr_dot_multi, pb, B_ptr_ptr, svh_ptr_ptr, A_had_ptr,
                  size_k, size_n, size_m, bszm, min_index, max_index, size_n_list, c_list, core, tpb,
                  num_tokens, weighted ? 1 : 0)
    }
    return true;
}

#undef LAUNCH_MR
#undef LAUNCH_MR_K
#undef LAUNCH_MR_CB
#undef LAUNCH_MR_W
#undef LAUNCH_MR_M

// =============================================================================
// Routed-MoE decode in four launches: exl3_rocm::moe_decode
// =============================================================================
//
// The per-token MoE route runs a routed expert block as exl3_mgemm(gate) +
// exl3_mgemm(up) + silu_mul + exl3_mgemm(down, weighted): seven dependent
// launches per layer at m == 1 (each mgemm is a rotation + a dot kernel here),
// and every one of them costs a command-processor gap on top of its run time --
// decode is GPU-bound (the host runs ahead), so launches are paid in GPU time.
// This op runs the same arithmetic in four:
//
//   1. exl3_moe_dec_had_gu    rotate each token's input once per (slot, gate|up)
//                             into yh -- slot j < S is gate[sel[j]], j >= S is
//                             up[sel[j - S]]; token = (j mod S) / top_k, read in
//                             place (no replicated input rows at bsz > 1)
//   2. exl3_moe_dec_dot       gate and up as one 2S-slot split-K GEMV (the
//                             multi-row body, row tile 1, fused output rotation)
//   3. exl3_moe_dec_had_act   silu(gate) * up in registers (act_mul_kernel_h's
//                             half2 arithmetic and act_limit clamp, verbatim),
//                             rotated with the down projection's suh into A_had
//   4. exl3_moe_dec_dot       down, weighted, with the mgemv fused epilogue's
//                             grouped reduce: token t's routed sum -> out row t
//
// Every value is the one the four-mgemm route produces: the rotations, dot core
// and epilogues are the same functions; the split-K wave count is sized from one
// token's slot count per matrix (top_k), exactly the rule the separate calls
// apply; the activation is elementwise in the same half2 ops. Pointer tables:
// gate and up concatenated (2E entries, built once at load). SILU with a gated
// MLP only; the caller falls back to the mgemm route otherwise.

__device__ __forceinline__ half2 exl3_moe_silu_h2(half2 x)   // activation_kernels.cuh _silu(half2)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

// exl3_gemv_had_in_128 with the 128 input halves already in registers (v)
__device__ __forceinline__ void exl3_moe_had_in_128_v
(
    half4 v,
    half* output_ptr,
    const half* __restrict__ scale,
    const int lane
)
{
    half4 scales = ((const half4*) scale)[lane];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    ((half4*) output_ptr)[lane] = v;
}

static __global__
__launch_bounds__(256)
void exl3_moe_dec_had_gu
(
    const half* __restrict__ y,            // (bsz, size_k)
    const int64_t* __restrict__ sel,       // (S)
    const uintptr_t* __restrict__ suh_gu,  // (2E)
    half* __restrict__ yh,                 // (2S, size_k)
    const int size_k,
    const int S,
    const int top_k,
    const int E
)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= 2 * S * k_segs) return;
    const int j = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int js = j < S ? j : j - S;
    const int mat = (int) sel[js] + (j < S ? 0 : E);
    const int token = js / top_k;
    const half* suh = (const half*) suh_gu[mat];
    exl3_gemv_had_in_128(y + (int64_t) token * size_k + seg * 128,
                         yh + (int64_t) j * size_k + seg * 128, suh + seg * 128, lane);
}

static __global__
__launch_bounds__(256)
void exl3_moe_dec_had_act
(
    const half* __restrict__ gu,           // (2S, size_k): gate rows, then up rows
    const int64_t* __restrict__ sel,       // (S)
    const uintptr_t* __restrict__ suh_d,   // (E)
    half* __restrict__ a_had,              // (S, size_k)
    const int size_k,
    const int S,
    const float act_limit
)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= S * k_segs) return;
    const int j = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int64_t off = (int64_t) j * size_k + seg * 128;
    half4 g = ((const half4*) (gu + off))[lane];
    half4 u = ((const half4*) (gu + (int64_t) S * size_k + off))[lane];

    // act_mul_kernel_h<ACT_SILU>, per half2
    half2 xg[2] = { g.x, g.y };
    half2 yu[2] = { u.x, u.y };
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        half2 x2 = exl3_moe_silu_h2(xg[i]);
        half2 y2 = yu[i];
        if (act_limit != 0.0f)
        {
            y2 = __hmax2(y2, __float2half2_rn(-act_limit));
            y2 = __hmin2(y2, __float2half2_rn(act_limit));
            x2 = __hmin2(x2, __float2half2_rn(act_limit));
        }
        xg[i] = __hmul2(x2, y2);
    }
    half4 a;
    a.x = xg[0];
    a.y = xg[1];
    const half* suh = (const half*) suh_d[(int) sel[j]];
    exl3_moe_had_in_128_v(a, a_had + off, suh + seg * 128, lane);
}

// Gate|up (weighted = false: 2S slots, table split at S) or down (weighted: S slots)
template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_moe_dec_dot
(
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ svh_list,
    const int64_t* __restrict__ sel,
    const half* __restrict__ weights,      // (S) for the weighted form, else nullptr
    const half* __restrict__ A_had,        // (slots, size_k)
    void* __restrict__ C,                  // (slots, size_n)
    const int size_k,
    const int size_n,
    const int S,
    const int E,
    const int num_tokens,
    const int core,
    const int tpb
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;
    const int n_segs = size_n / 128;
    const int js = j < S ? j : j - S;
    const int mat = (int) sel[js] + (j < S ? 0 : E);
    const int64_t row_off = (int64_t) j * size_n;

    if (weights)
    {
        const float w_scale = 0.088388347648f * __half2float(weights[j]);
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, 1>
        (
            A_had + (int64_t) j * size_k, size_k, (const uint16_t*) B_list[mat], size_k, n_tiles, tile_n,
            C, row_off, size_n, 1, (const half*) svh_list[mat],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb,
            w_scale, pb->red_counters, S, size_n, num_tokens
        );
    }
    else
    {
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, 1>
        (
            A_had + (int64_t) j * size_k, size_k, (const uint16_t*) B_list[mat], size_k, n_tiles, tile_n,
            C, row_off, size_n, 1, (const half*) svh_list[mat],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb
        );
    }
}

static void exl3_moe_dec_launch_dot
(
    Exl3GemvMrParams* pb, const uintptr_t* B_list, const uintptr_t* svh_list, const int64_t* sel,
    const half* weights, const half* A_had, void* C, int size_k, int size_n, int S, int E,
    int num_tokens, int slots, int top_k, int K, int cb, bool c_fp32, int device, cudaStream_t stream
)
{
    const int n_tiles = size_n / 16;
    // One token's slots per matrix: the rule the separate per-matrix calls apply
    const int warps = exl3_gemv_mr_warps(size_k, n_tiles, top_k);
    const int core = exl3_gemv_core_mode();
    const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, slots, warps);
    dim3 grid(n_tiles / tpb, slots);
    dim3 block(warps * 32);
    size_t smem = exl3_gemv_mr_smem_bytes(warps, 1);

    #define MOE_DEC_L3(bits_val, codebook, warps_val) \
        if (c_fp32) hipLaunchKernelGGL((exl3_moe_dec_dot<bits_val, true, codebook, warps_val>), grid, block, smem, stream, \
            pb, B_list, svh_list, sel, weights, A_had, C, size_k, size_n, S, E, num_tokens, core, tpb); \
        else hipLaunchKernelGGL((exl3_moe_dec_dot<bits_val, false, codebook, warps_val>), grid, block, smem, stream, \
            pb, B_list, svh_list, sel, weights, A_had, C, size_k, size_n, S, E, num_tokens, core, tpb);
    #define MOE_DEC_L2(bits_val, codebook) \
        switch (warps) { \
            case 4:  MOE_DEC_L3(bits_val, codebook, 4)  break; \
            case 8:  MOE_DEC_L3(bits_val, codebook, 8)  break; \
            case 16: MOE_DEC_L3(bits_val, codebook, 16) break; \
        }
    #define MOE_DEC_L1(bits_val) \
        switch (cb) { \
            case 0: MOE_DEC_L2(bits_val, 0) break; \
            case 1: MOE_DEC_L2(bits_val, 1) break; \
            case 2: MOE_DEC_L2(bits_val, 2) break; \
        }
    switch (K)
    {
        case 1: MOE_DEC_L1(1) break;
        case 2: MOE_DEC_L1(2) break;
        case 3: MOE_DEC_L1(3) break;
        case 4: MOE_DEC_L1(4) break;
        case 5: MOE_DEC_L1(5) break;
        case 6: MOE_DEC_L1(6) break;
        case 7: MOE_DEC_L1(7) break;
        case 8: MOE_DEC_L1(8) break;
        case EXL3_HALF_BITS(1): MOE_DEC_L2(EXL3_HALF_BITS(1), 2) break;
        case EXL3_HALF_BITS(2): MOE_DEC_L2(EXL3_HALF_BITS(2), 2) break;
        case EXL3_HALF_BITS(3): MOE_DEC_L2(EXL3_HALF_BITS(3), 2) break;
    }
    #undef MOE_DEC_L1
    #undef MOE_DEC_L2
    #undef MOE_DEC_L3
}

// Whether exl3_rocm_moe_decode takes a call of S slots (bsz * top_k) at these widths: the same bounds the
// op checks, for callers that fall back instead of raising
// 2 * S slots: gate and up of every (row, expert) pick, MAX_BSZN = 16 rows of top_k <= 16
#define MAX_BSZN_MOE_SLOTS 128
bool exl3_rocm_moe_decode_fits(int S, int Hi, int I, int Ho, int K_gu, int cb_gu, int K_d, int cb_d)
{
    return Hi % 128 == 0 && I % 128 == 0 && Ho % 128 == 0
        && 2 * S <= 4 * MAX_BSZN_MOE_SLOTS && 2 * S * (I / 128) <= EXL3_MGEMV_SEG_COUNTERS && Ho / 128 <= EXL3_MGEMV_RED_COUNTERS
        && S * (Ho / 128) <= EXL3_MGEMV_SEG_COUNTERS
        && I / 16 <= EXL3_GEMV_SPLITK_MAX_TILES && Ho / 16 <= EXL3_GEMV_SPLITK_MAX_TILES
        && cb_gu >= 0 && cb_gu <= 2 && cb_d >= 0 && cb_d <= 2
        && exl3_gemv_mr_k_ok(K_gu, cb_gu) && exl3_gemv_mr_k_ok(K_d, cb_d);
}

// cb: 0 = 3inst, 1 = mcg, 2 = mul1 (exl3_mgemm_gr's encoding). K: integer 1..8, or
// EXL3_HALF_BITS(k) = 16 + k for a k + 0.5 bpw mul1 matrix (the caller passes it only for
// half-rate layers, and only while EXL3_ROCM_HALF_GEMV is on)
void exl3_rocm_moe_decode
(
    const at::Tensor& y,            // (bsz, Hi) half
    const at::Tensor& sel,          // (bsz, top_k) int64
    const at::Tensor& weights,      // (bsz, top_k) half
    const at::Tensor& gu_trellis,   // (2E) int64 pointer tables, gate then up
    const at::Tensor& gu_suh,
    const at::Tensor& gu_svh,
    const at::Tensor& d_trellis,    // (E)
    const at::Tensor& d_suh,
    const at::Tensor& d_svh,
    at::Tensor& yh,                 // (>= 2S, Hi) half scratch
    at::Tensor& gu,                 // (>= 2S, I) half scratch
    at::Tensor& a_had,              // (>= S, I) half scratch
    at::Tensor& out,                // (>= S, Ho) float: rows 0..bsz-1 = routed sums
    int64_t K_gu,
    int64_t cb_gu,
    int64_t K_d,
    int64_t cb_d,
    double act_limit
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(weights, kHalf);
    TORCH_CHECK_DTYPE(yh, kHalf);
    TORCH_CHECK_DTYPE(gu, kHalf);
    TORCH_CHECK_DTYPE(a_had, kHalf);
    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK(y.is_contiguous() && sel.is_contiguous() && weights.is_contiguous(), "moe_decode: inputs must be contiguous");
    TORCH_CHECK(yh.is_contiguous() && gu.is_contiguous() && a_had.is_contiguous() && out.is_contiguous(), "moe_decode: buffers must be contiguous");

    const int bsz = (int) y.size(0);
    const int Hi = (int) y.size(-1);
    const int top_k = (int) sel.size(-1);
    const int S = bsz * top_k;
    const int E = (int) d_trellis.size(0);
    const int I = (int) gu.size(-1);
    const int Ho = (int) out.size(-1);
    TORCH_CHECK(sel.numel() == S && weights.numel() == S, "moe_decode: sel/weights shape");
    TORCH_CHECK(gu_trellis.size(0) == 2 * E && gu_suh.size(0) == 2 * E && gu_svh.size(0) == 2 * E, "moe_decode: gate|up tables must hold 2E entries");
    TORCH_CHECK(Hi % 128 == 0 && I % 128 == 0 && Ho % 128 == 0, "moe_decode: dims must be multiples of 128");
    TORCH_CHECK(yh.numel() >= (int64_t) 2 * S * Hi && gu.numel() >= (int64_t) 2 * S * I
                && a_had.numel() >= (int64_t) S * I && out.numel() >= (int64_t) S * Ho, "moe_decode: scratch too small");
    TORCH_CHECK(2 * S <= 4 * MAX_BSZN_MOE_SLOTS && 2 * S * (I / 128) <= EXL3_MGEMV_SEG_COUNTERS && Ho / 128 <= EXL3_MGEMV_RED_COUNTERS
                && S * (Ho / 128) <= EXL3_MGEMV_SEG_COUNTERS, "moe_decode: shape exceeds the arrival counters");
    TORCH_CHECK(I / 16 <= EXL3_GEMV_SPLITK_MAX_TILES && Ho / 16 <= EXL3_GEMV_SPLITK_MAX_TILES, "moe_decode: output too wide");
    TORCH_CHECK(cb_gu >= 0 && cb_gu <= 2 && cb_d >= 0 && cb_d <= 2
                && exl3_gemv_mr_k_ok((int) K_gu, (int) cb_gu) && exl3_gemv_mr_k_ok((int) K_d, (int) cb_d), "moe_decode: K/cb");

    int device = y.get_device();
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, true);
    TORCH_CHECK(pb, "moe_decode: parameter block allocation failed");

    const int64_t* sel_p = (const int64_t*) sel.data_ptr();

    // 1. gate|up input rotation
    {
        int blocks = CEIL_DIVIDE(2 * S * (Hi / 128) * 32, 256);
        hipLaunchKernelGGL(exl3_moe_dec_had_gu, dim3(blocks), dim3(256), 0, stream,
            (const half*) y.data_ptr(), sel_p, (const uintptr_t*) gu_suh.data_ptr(),
            (half*) yh.data_ptr(), Hi, S, top_k, E);
    }
    // 2. gate|up GEMV, 2S slots
    exl3_moe_dec_launch_dot(pb, (const uintptr_t*) gu_trellis.data_ptr(), (const uintptr_t*) gu_svh.data_ptr(),
        sel_p, nullptr, (const half*) yh.data_ptr(), gu.data_ptr(), Hi, I, S, E, 1, 2 * S, top_k,
        (int) K_gu, (int) cb_gu, false, device, stream);
    // 3. activation + down input rotation
    {
        int blocks = CEIL_DIVIDE(S * (I / 128) * 32, 256);
        hipLaunchKernelGGL(exl3_moe_dec_had_act, dim3(blocks), dim3(256), 0, stream,
            (const half*) gu.data_ptr(), sel_p, (const uintptr_t*) d_suh.data_ptr(),
            (half*) a_had.data_ptr(), I, S, (float) act_limit);
    }
    // 4. down GEMV, weighted, grouped per token
    exl3_moe_dec_launch_dot(pb, (const uintptr_t*) d_trellis.data_ptr(), (const uintptr_t*) d_svh.data_ptr(),
        sel_p, (const half*) weights.data_ptr(), (const half*) a_had.data_ptr(), out.data_ptr(), I, Ho, S, 0, bsz,
        S, top_k, (int) K_d, (int) cb_d, true, device, stream);
    cuda_check(cudaPeekAtLastError());
}


// =============================================================================
// Persistent MoE block at bsz 1: exl3_rocm_moe_persist
// =============================================================================
//
// One launch for the whole MoE block of one token: router GEMV, top-k softmax, the routed experts' gate|up,
// activation and down, the shared expert and its sigmoid gate, and the sum. A co-resident grid (sized from the
// occupancy, at most the largest phase's task count) runs three phases separated by two grid barriers
// (counter + generation word, no float atomics):
//
//   phase 0  router scores (one warp per expert row), shared expert gate|up (it does not depend on the routing),
//            shared gate dot sigmoid(x . w)
//   phase 1  top-k + softmax, computed by every block on the stored scores (identical, no further barrier);
//            routed gate|up, one task per (slot, matrix, 128-column segment)
//   phase 2  activation + down input rotation into LDS and the weighted down GEMV, one task per (slot,
//            128-column output segment); the shared expert is slot S with weight sigmoid(x . w). The fused
//            epilogue's grouped reduce (fixed slot order) writes the block output into row 0 of out.
//
// A task rotates its own input into LDS (the input is 2.5-5 KB), so no phase needs a rotated global copy. The
// dot body, its split-K and the output-rotation epilogue are exl3_gemv_mr_body's. Fixed reduction orders
// throughout, so the result is deterministic run to run; it is not bit-identical to the multi-launch route (the
// split-K wave count, the score precision and the shared expert's weighting differ).

struct MoePersistArgs
{
    const half* y;               // (Hi)
    const half* gate_t;          // (E, Hi) router
    const half* sh_w;            // (Hi) shared gate weight
    const uintptr_t* rgu_t; const uintptr_t* rgu_s; const uintptr_t* rgu_v;   // (2E) gate then up
    const uintptr_t* rd_t; const uintptr_t* rd_s; const uintptr_t* rd_v;      // (E)
    const uintptr_t* sgu_t; const uintptr_t* sgu_s; const uintptr_t* sgu_v;   // (2) shared gate, up
    const uintptr_t* sd_t; const uintptr_t* sd_s; const uintptr_t* sd_v;      // (1) shared down
    float* scores;               // (E + 1): scores, then sigmoid(x . w)
    int64_t* sel_out;            // (K) chosen experts (diagnostics)
    half* gu;                    // (2 (S + 1), I): gate rows of slots 0..S, then up rows
    float* out;                  // (S + 1, Ho): rotated weighted partials; row 0 = result
    unsigned* bar;               // [0] arrivals, [1] generation
    int* segc;                   // segment arrival counters
    int* redc;                   // reduction counters (Ho / 128)
    int Hi, I, Ho, E, K;
    int core;
    int red_bytes;
    float act_limit_r, act_limit_s;
    int phases;
};

// exl3_gemv_mr_body at M = 1 with the rotated input in LDS: the tiles core takes A_LDS = true here (the body's
// own instantiation reads A through global-memory loads, which an LDS address faults)
template <int bits, bool c_fp32, int cb, int W>
__device__ __forceinline__ void moe_persist_body
(
    const half* A_lds, const uint16_t* __restrict__ B, const int size_k, const int n_tiles, const int tile_n,
    void* Cb, const int64_t row_off, const half* __restrict__ svh, int* seg_counters,
    const int warp_id, const int lane, const int core,
    const float w_scale, int* red_counters, const int red_target, const int size_n
)
{
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;
    constexpr int T = EXL3_GEMV_TILES_TMAX;
    float acc[T];
    exl3_gemv_dot_tile_splitk_t<bits, cb, W, 1, true, T>
    (
        core, 1, A_lds, size_k, B, size_k, n_tiles, tile_n, warp_id, lane, sh_red, acc,
        [&](int tn, int k0, int k1, float* o)
        {
            exl3_gemv_dot_tile_direct_mr<bits, cb, 1>(A_lds, size_k, B, n_tiles, tn, lane, k0, k1, o, 1);
        },
        1
    );
    if (warp_id != 0) return;
    if (red_counters)
        exl3_gemv_fused_epilogue<c_fp32>(acc[0], Cb, row_off, size_n, tile_n, lane, svh, w_scale,
            seg_counters + tile_n / 8, red_counters + tile_n / 8, red_target, 1);
    else
        exl3_gemv_fused_epilogue_mr<c_fp32, 1>(acc, Cb, row_off, size_n, 1, tile_n, lane, svh, seg_counters + tile_n / 8);
}

// One warp, one 16-column tile over the whole K (A in LDS), then the fused output epilogue: the warps of a block
// take the 8 tiles of one 128-column segment together, so a segment is one pass with no block barrier
template <int bits, bool c_fp32, int cb>
__device__ __forceinline__ void moe_persist_tile
(
    const half* A_lds, const uint16_t* __restrict__ B, const int size_k, const int n_tiles, const int tile_n,
    void* Cb, const int64_t row_off, const half* __restrict__ svh, int* seg_counters,
    const int lane, const int core,
    const float w_scale, int* red_counters, const int red_target, const int size_n
)
{
    float acc[1];
    if (core == EXL3_GEMV_CORE_TILES)
        exl3_gemv_dot_tile_tiles<bits, cb, 1, true, exl3_tiles_u_single<bits>(), 1>
            (A_lds, size_k, B, n_tiles, tile_n, lane, 0, size_k / 16, acc, 1);
    else
        exl3_gemv_dot_tile_direct_mr<bits, cb, 1>(A_lds, size_k, B, n_tiles, tile_n, lane, 0, size_k / 16, acc, 1);
    if (red_counters)
        exl3_gemv_fused_epilogue<c_fp32>(acc[0], Cb, row_off, size_n, tile_n, lane, svh, w_scale,
            seg_counters + tile_n / 8, red_counters + tile_n / 8, red_target, 1);
    else
        exl3_gemv_fused_epilogue_mr<c_fp32, 1>(acc, Cb, row_off, size_n, 1, tile_n, lane, svh, seg_counters + tile_n / 8);
}

// One 128-column output segment (8 adjacent N-tiles) per block: every warp reads all 8 tiles (T = 8, 8 * 32 * bits
// contiguous bytes per k-tile) over its own K chunk, the chunks are summed in warp order through LDS, and warp t
// stores tile t through the fused output epilogue. Every thread enters. U: k-tiles in flight per load batch
#ifndef EXL3_MOE_PERSIST_U
#define EXL3_MOE_PERSIST_U 2
#endif
template <int bits, bool c_fp32, int cb, int W>
__device__ __forceinline__ void moe_persist_seg
(
    const half* A_lds, const uint16_t* __restrict__ B, const int size_k, const int n_tiles, const int seg,
    void* Cb, const int64_t row_off, const half* __restrict__ svh, int* seg_counters,
    const int warp, const int lane,
    const float w_scale, int* red_counters, const int red_target, const int size_n
)
{
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;    // [W][8][16]
    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + W - 1) / W;
    const int kb0 = exl3_uni(warp) * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    if (kb0 < kb1)
        exl3_gemv_dot_tile_tiles<bits, cb, 1, true, EXL3_MOE_PERSIST_U, 8>
            (A_lds, size_k, B, n_tiles, seg * 8, lane, kb0, kb1, acc, 1);
    if (lane < 16)
    {
        #pragma unroll
        for (int t = 0; t < 8; ++t) sh_red[(warp * 8 + t) * 16 + lane] = acc[t];
    }
    __syncthreads();
    if (warp < 8)
    {
        float total = 0.0f;
        if (lane < 16)
        {
            #pragma unroll
            for (int w = 0; w < W; ++w) total += sh_red[(w * 8 + warp) * 16 + lane];
        }
        const int tile_n = seg * 8 + warp;
        if (red_counters)
            exl3_gemv_fused_epilogue<c_fp32>(total, Cb, row_off, size_n, tile_n, lane, svh, w_scale,
                seg_counters + seg, red_counters + seg, red_target, 1);
        else
            exl3_gemv_fused_epilogue_mr<c_fp32, 1>(&total, Cb, row_off, size_n, 1, tile_n, lane, svh, seg_counters + seg);
    }
}

__device__ __forceinline__ void moe_persist_barrier(unsigned* bar)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned gen = __hip_atomic_load(bar + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
        const unsigned arrived = __hip_atomic_fetch_add(bar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == gridDim.x - 1)
        {
            __hip_atomic_store(bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(bar + 1, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        else
        {
            // Bounded wait: a barrier that never completes (a launch the device cannot hold co-resident) records
            // itself in bar[2..3] and falls through instead of hanging the device
            unsigned spins = 0;
            while (__hip_atomic_load(bar + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen)
            {
                __builtin_amdgcn_s_sleep(1);
                if (++spins == (1u << 22))
                {
                    __hip_atomic_store(bar + 2, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    __hip_atomic_store(bar + 3, __hip_atomic_load(bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    break;
                }
            }
        }
    }
    __syncthreads();
    __threadfence();
}

// Rotate x (LDS, Hi halves) with suh into dst (LDS); every thread enters
template <int W>
__device__ __forceinline__ void moe_persist_rotate(const half* x_s, half* dst, const half* suh, int Hi, int warp, int lane)
{
    for (int seg = warp; seg < Hi / 128; seg += W)
        exl3_gemv_had_in_128(x_s + seg * 128, dst + seg * 128, suh + seg * 128, lane);
    __syncthreads();
}

template <int BR, int BS, int CB, int W>
static __global__
__launch_bounds__(W * 32)
__attribute__((amdgpu_flat_work_group_size(W * 32, W * 32)))
void exl3_moe_persist_kernel(const MoePersistArgs a)
{
    static_assert(W == 8, "one warp per tile of a 128-column segment");
    extern __shared__ char shared_mem[];
    half* x_s = (half*) (shared_mem + a.red_bytes);
    half* a_s = x_s + a.Hi;
    __shared__ int sel_s[32];
    __shared__ float w_s[32];
    __shared__ float red_s[W];
    __shared__ float sig_s;

    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int Hi = a.Hi, I = a.I, Ho = a.Ho, E = a.E, K = a.K;
    const int S = K;                 // routed slots (bsz 1)
    const int R = S + 1;             // gu rows per matrix, out rows
    const int segs_i = I / 128;
    const int segs_o = Ho / 128;

    for (int i = t; i < Hi / 8; i += W * 32) ((uint4*) x_s)[i] = ((const uint4*) a.y)[i];
    __syncthreads();
    // Work list: tickets in [0, n0) are phase 0 tasks, [n0, n0 + n1) phase 1, then phase 2. A block takes the next
    // ticket from a global counter, and a phase 1 (2) task waits until every phase 0 (1) task has completed. Every
    // ticket below a waiting block's own was taken by a block that is already running, so the waits always resolve
    // whatever part of the grid is resident (no co-residency assumption, unlike a grid barrier). The last block
    // to exit resets the counters for the next launch.
    unsigned* tk_ctr = a.bar + 4;
    unsigned* done0 = a.bar + 5;
    unsigned* done1 = a.bar + 6;
    unsigned* exitc = a.bar + 7;
    const int n_rt = (E + W - 1) / W;
    const int n_sgu = 2 * segs_i;
    const int n0 = n_rt + n_sgu + 1;
    const int n1 = 2 * S * segs_i;
    const int n2 = R * segs_o;
    const int n_total = a.phases >= 3 ? n0 + n1 + n2 : (a.phases == 2 ? n0 + n1 : n0);
    __shared__ int tk_s;
    bool have_topk = false, have_p1 = false;

    auto wait_count = [&](unsigned* c, unsigned target)
    {
        if (t == 0)
        {
            unsigned spins = 0;
            while (__hip_atomic_load(c, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target)
            {
                __builtin_amdgcn_s_sleep(1);
                if (++spins == (1u << 24)) { __hip_atomic_store(a.bar + 2, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); break; }
            }
        }
        __syncthreads();
        __threadfence();
    };

    for (;;)
    {
        if (t == 0) tk_s = (int) __hip_atomic_fetch_add(tk_ctr, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        const int tk = tk_s;
        __syncthreads();
        if (tk >= n_total) break;

        unsigned* done = nullptr;
        if (tk < n0)
        {
            const int task = tk;
            if (task < n_rt)
            {
                const int row = task * W + warp;
                if (row < E)
                {
                    const uint4* w4 = (const uint4*) (a.gate_t + (size_t) row * Hi);
                    float sum = 0.0f;
                    for (int i = lane; i < Hi / 8; i += 32)
                    {
                        uint4 wv = w4[i];
                        uint4 xv = ((const uint4*) x_s)[i];
                        const half2* wh = (const half2*) &wv;
                        const half2* xh = (const half2*) &xv;
                        #pragma unroll
                        for (int q = 0; q < 4; ++q)
                        {
                            float2 wf = __half22float2(wh[q]);
                            float2 xf = __half22float2(xh[q]);
                            sum = fmaf(xf.x, wf.x, sum);
                            sum = fmaf(xf.y, wf.y, sum);
                        }
                    }
                    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
                    if (lane == 0) a.scores[row] = __half2float(__float2half_rn(sum));
                }
            }
            else if (task < n_rt + n_sgu)
            {
                const int q = task - n_rt;
                const int m = q / segs_i;
                const int seg = q % segs_i;
                moe_persist_rotate<W>(x_s, a_s, (const half*) a.sgu_s[m], Hi, warp, lane);
                const int row = m * R + S;
                moe_persist_seg<BS, false, CB, W>
                (
                    a_s, (const uint16_t*) a.sgu_t[m], Hi, I / 16, seg,
                    a.gu, (int64_t) row * I, (const half*) a.sgu_v[m],
                    a.segc + row * segs_i, warp, lane, 0.0f, nullptr, 0, I
                );
                __syncthreads();
            }
            else
            {
                float v = 0.0f;
                for (int i = t; i < Hi; i += W * 32)
                    v = fmaf(__half2float(a.sh_w[i]), __half2float(x_s[i]), v);
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
                if (lane == 0) red_s[warp] = v;
                __syncthreads();
                if (t == 0)
                {
                    float s = 0.0f;
                    for (int w = 0; w < W; ++w) s += red_s[w];
                    a.scores[E] = 1.0f / (1.0f + __expf(-s));
                }
                __syncthreads();
            }
            done = done0;
        }
        else
        {
            if (!have_topk)
            {
                wait_count(done0, (unsigned) n0);
            if (warp == 0)
            {
                constexpr int MAXC = 512 / 32;
                float v[MAXC];
                #pragma unroll
                for (int i = 0; i < MAXC; ++i)
                {
                    const int e = lane + 32 * i;
                    float sv = e < E ? __hip_atomic_load(a.scores + e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : (-__builtin_inff());
                    v[i] = sv == sv ? sv : (-__builtin_inff());     // NaN scores (warmup garbage) rank lowest
                }
                float top0 = 0.0f, esum = 0.0f;
                uint32_t taken = 0;
                float ev[32];
                for (int k = 0; k < K; ++k)
                {
                    // Highest score, lowest index on ties; taken marks this lane's picks, so every pick is a distinct
                    // in-range expert even when scores tie at -inf
                    float bv = (-__builtin_inff()); int bi = 0x7fffffff;
                    #pragma unroll
                    for (int i = 0; i < MAXC; ++i)
                    {
                        const int e = lane + 32 * i;
                        if (e < E && !((taken >> i) & 1u) && (v[i] > bv || bi == 0x7fffffff)) { bv = v[i]; bi = e; }
                    }
                    for (int o = 16; o > 0; o >>= 1)
                    {
                        float ov = __shfl_xor(bv, o, 32);
                        int oi = __shfl_xor(bi, o, 32);
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                    }
                    if (lane == (bi & 31)) taken |= 1u << (bi >> 5);
                    if (k == 0) top0 = bv;
                    ev[k] = __expf(bv - top0);
                    esum += ev[k];
                    if (lane == 0) sel_s[k] = bi;
                }
                if (lane == 0)
                {
                    const float inv = 1.0f / (esum + 1e-20f);
                    for (int k = 0; k < K; ++k) w_s[k] = __half2float(__float2half_rn(ev[k] * inv));
                    sig_s = __hip_atomic_load(a.scores + E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    if (blockIdx.x == 0) for (int k = 0; k < K; ++k) a.sel_out[k] = sel_s[k];
                }
            }
                __syncthreads();
                have_topk = true;
            }
            if (tk < n0 + n1)
            {
                const int task = tk - n0;
                const int m = task / (S * segs_i);
                const int rem = task % (S * segs_i);
                const int j = rem / segs_i;
                const int seg = rem % segs_i;
                const int mat = sel_s[j] + m * E;
                moe_persist_rotate<W>(x_s, a_s, (const half*) a.rgu_s[mat], Hi, warp, lane);
                const int row = m * R + j;
                moe_persist_seg<BR, false, CB, W>
                (
                    a_s, (const uint16_t*) a.rgu_t[mat], Hi, I / 16, seg,
                    a.gu, (int64_t) row * I, (const half*) a.rgu_v[mat],
                    a.segc + row * segs_i, warp, lane, 0.0f, nullptr, 0, I
                );
                __syncthreads();
                done = done1;
            }
            else
            {
                if (!have_p1) { wait_count(done1, (unsigned) n1); have_p1 = true; }
                const int task = tk - n0 - n1;
                const int j = task / segs_o;
                const int oseg = task % segs_o;
                const bool sh = j == S;
                const half* suh_d = sh ? (const half*) a.sd_s[0] : (const half*) a.rd_s[sel_s[j]];
                const float act_limit = sh ? a.act_limit_s : a.act_limit_r;
                for (int seg = warp; seg < segs_i; seg += W)
                {
                    const int64_t off = (int64_t) j * I + seg * 128;
                    half4 g = ((const half4*) (a.gu + off))[lane];
                    half4 u = ((const half4*) (a.gu + (int64_t) R * I + off))[lane];
                    half2 xg[2] = { g.x, g.y };
                    half2 yu[2] = { u.x, u.y };
                    #pragma unroll
                    for (int i = 0; i < 2; ++i)
                    {
                        half2 x2 = exl3_moe_silu_h2(xg[i]);
                        half2 y2 = yu[i];
                        if (act_limit != 0.0f)
                        {
                            y2 = __hmax2(y2, __float2half2_rn(-act_limit));
                            y2 = __hmin2(y2, __float2half2_rn(act_limit));
                            x2 = __hmin2(x2, __float2half2_rn(act_limit));
                        }
                        xg[i] = __hmul2(x2, y2);
                    }
                    half4 av;
                    av.x = xg[0];
                    av.y = xg[1];
                    exl3_moe_had_in_128_v(av, a_s + seg * 128, suh_d + seg * 128, lane);
                }
                __syncthreads();
                const float w_scale = 0.088388347648f * (sh ? sig_s : w_s[j]);
                if (sh)
                    moe_persist_seg<BS, true, CB, W>
                    (
                        a_s, (const uint16_t*) a.sd_t[0], I, Ho / 16, oseg,
                        a.out, (int64_t) j * Ho, (const half*) a.sd_v[0],
                        a.segc + j * segs_o, warp, lane, w_scale, a.redc, R, Ho
                    );
                else
                    moe_persist_seg<BR, true, CB, W>
                    (
                        a_s, (const uint16_t*) a.rd_t[sel_s[j]], I, Ho / 16, oseg,
                        a.out, (int64_t) j * Ho, (const half*) a.rd_v[sel_s[j]],
                        a.segc + j * segs_o, warp, lane, w_scale, a.redc, R, Ho
                    );
                __syncthreads();
            }
        }
        // Completion: every wave's stores are visible before the count
        __threadfence();
        __syncthreads();
        if (t == 0 && done) __hip_atomic_fetch_add(done, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }

    if (t == 0)
    {
        if (__hip_atomic_fetch_add(exitc, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == gridDim.x - 1)
        {
            __hip_atomic_store(tk_ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(done0, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(done1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(exitc, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#define MOE_PERSIST_W 8

bool exl3_rocm_moe_persist_fits(int E, int K, int Hi, int I, int Ho, int K_r, int K_s, int cb)
{
    const bool kr = K_r == 2 || K_r == EXL3_HALF_BITS(2) || K_r == 3 || K_r == 4;
    return kr && K_s == 6 && cb == 2 && E <= 512 && K <= 16 && Hi % 128 == 0 && I % 128 == 0 && Ho % 128 == 0
        && Hi <= 8192 && I <= Hi && (2 * (K + 1)) * (I / 128) <= 2048 && (K + 1) * (Ho / 128) <= 2048 && Ho / 128 <= 1024;
}

void exl3_rocm_moe_persist
(
    const at::Tensor& y, const at::Tensor& gate_t, const at::Tensor& sh_w,
    const at::Tensor& rgu_t, const at::Tensor& rgu_s, const at::Tensor& rgu_v,
    const at::Tensor& rd_t, const at::Tensor& rd_s, const at::Tensor& rd_v,
    const at::Tensor& sgu_t, const at::Tensor& sgu_s, const at::Tensor& sgu_v,
    const at::Tensor& sd_t, const at::Tensor& sd_s, const at::Tensor& sd_v,
    at::Tensor& scores, at::Tensor& sel_out, at::Tensor& gu, at::Tensor& out, at::Tensor& ctr,
    int64_t K, int64_t K_r, int64_t K_s, int64_t cb, double act_limit_r, double act_limit_s
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MoePersistArgs a;
    a.Hi = (int) y.size(-1);
    a.E = (int) gate_t.size(0);
    a.I = (int) gu.size(-1);
    a.Ho = (int) out.size(-1);
    a.K = (int) K;
    TORCH_CHECK(y.numel() == a.Hi && gate_t.size(1) == a.Hi, "moe_persist: bsz 1 only");
    TORCH_CHECK(exl3_rocm_moe_persist_fits(a.E, a.K, a.Hi, a.I, a.Ho, (int) K_r, (int) K_s, (int) cb), "moe_persist: shape");
    TORCH_CHECK(gu.numel() >= (int64_t) 2 * (K + 1) * a.I && out.numel() >= (int64_t) (K + 1) * a.Ho
                && scores.numel() >= a.E + 1 && ctr.numel() >= 64 + 2048 + 1024, "moe_persist: scratch");
    a.y = (const half*) y.data_ptr();
    a.gate_t = (const half*) gate_t.data_ptr();
    a.sh_w = (const half*) sh_w.data_ptr();
    a.rgu_t = (const uintptr_t*) rgu_t.data_ptr(); a.rgu_s = (const uintptr_t*) rgu_s.data_ptr(); a.rgu_v = (const uintptr_t*) rgu_v.data_ptr();
    a.rd_t = (const uintptr_t*) rd_t.data_ptr(); a.rd_s = (const uintptr_t*) rd_s.data_ptr(); a.rd_v = (const uintptr_t*) rd_v.data_ptr();
    a.sgu_t = (const uintptr_t*) sgu_t.data_ptr(); a.sgu_s = (const uintptr_t*) sgu_s.data_ptr(); a.sgu_v = (const uintptr_t*) sgu_v.data_ptr();
    a.sd_t = (const uintptr_t*) sd_t.data_ptr(); a.sd_s = (const uintptr_t*) sd_s.data_ptr(); a.sd_v = (const uintptr_t*) sd_v.data_ptr();
    a.scores = (float*) scores.data_ptr();
    a.sel_out = (int64_t*) sel_out.data_ptr();
    a.gu = (half*) gu.data_ptr();
    a.out = (float*) out.data_ptr();
    int* c = (int*) ctr.data_ptr();
    a.bar = (unsigned*) c;
    a.segc = c + 64;
    a.redc = c + 64 + 2048;
    a.core = exl3_gemv_core_mode();
    a.red_bytes = MOE_PERSIST_W * 8 * 16 * (int) sizeof(float);
    a.act_limit_r = (float) act_limit_r;
    a.act_limit_s = (float) act_limit_s;
    { const char* e = std::getenv("EXL3_MOE_PERSIST_PHASES"); a.phases = e ? atoi(e) : 3; }
    const size_t smem = a.red_bytes + (size_t) 2 * a.Hi * sizeof(half);

    const int device = y.get_device();
    const int max_tasks = std::max((a.E + MOE_PERSIST_W - 1) / MOE_PERSIST_W + 2 * (a.I / 128) + 1,
                          std::max(2 * a.K * (a.I / 128), (a.K + 1) * (a.Ho / 128)));

    #define MOE_PERSIST_LAUNCH(BR) \
    { \
        auto kern = exl3_moe_persist_kernel<BR, 6, 2, MOE_PERSIST_W>; \
        static int occ_cache[64] = {}; \
        int& occ = occ_cache[device]; \
        if (!occ) \
        { \
            int n = 0; \
            cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, MOE_PERSIST_W * 32, smem)); \
            int cus = 0; cuda_check(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device)); \
            occ = std::max(1, n) * cus; \
        } \
        const char* g_env = std::getenv("EXL3_MOE_PERSIST_GRID"); \
        int grid = std::min(occ, max_tasks); \
        if (g_env) grid = std::min(grid, std::max(1, atoi(g_env))); \
        hipLaunchKernelGGL(kern, dim3(grid), dim3(MOE_PERSIST_W * 32), smem, stream, a); \
    }
    switch ((int) K_r)
    {
        case 2: MOE_PERSIST_LAUNCH(2) break;
        case 3: MOE_PERSIST_LAUNCH(3) break;
        case 4: MOE_PERSIST_LAUNCH(4) break;
        case EXL3_HALF_BITS(2): MOE_PERSIST_LAUNCH(EXL3_HALF_BITS(2)) break;
    }
    #undef MOE_PERSIST_LAUNCH
    cuda_check(cudaPeekAtLastError());
}
