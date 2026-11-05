#pragma once
#include <cstring>

// GatedResidual decode dots for RDNA:
//
// gr_dots_kernel with RB fn rows per block instead of one. The thread -> column map (8-half column c = tid +
// GR_THREADS_A k), each per-stream fmaf chain, the warp tree and the 4-warp sum are gr_dots_kernel's, so every
// dot is bit-identical to it; what changes is that a block loads its stream values once into registers (weighted
// there when no pre-weighted copy exists) and then streams RB fn rows with all of a row's loads in flight, where
// the one-row kernel re-reads the stream stack in every block. The sum-of-squares row runs in its own block.
// gr_dots_kernel2 (the bytes-in-flight form) runs far off the bandwidth bound on these parts; both finalize
// kernels read the dots in the same layout.

template <int H, int RB, int NI, bool WEIGHTED>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_rows_kernel
(
    const float* __restrict__ streams,   // (R, H, D) raw (sum of squares)
    const float* __restrict__ wstreams,  // (R, H, D) streams * w, or null (WEIGHTED = false)
    const half* __restrict__ fn,         // (M, H * D) half projection
    const half* __restrict__ wn,         // (H * D) half norm weight, applied here when not WEIGHTED
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int D4 = D / 4;
    const int D8 = D / 8;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    const float4* d4 = (const float4*) ((WEIGHTED ? wstreams : streams) + (size_t) r * H * D);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ float red[RB][H][GR_THREADS_A / 32];

    const int nrb = (M + RB - 1) / RB;
    if (blockIdx.x == nrb)
    {
        // Sum-of-squares row: gr_dots_kernel's j == M branch
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[0][h][warp] = a;
        }
        __syncthreads();
        if (threadIdx.x < H)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[0][threadIdx.x][w];
            dots[((size_t) r * (M + 1) + M) * H + threadIdx.x] = v;
        }
        return;
    }

    const int j0 = blockIdx.x * RB;

    // This thread's (weighted) stream values, every stream: 8 columns at c = tid + GR_THREADS_A k
    float4 sa[H][NI], sb[H][NI];
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int k = 0; k < NI; ++k)
        {
            const int c = threadIdx.x + k * GR_THREADS_A;
            if (c < D8)
            {
                float4 s0 = d4[(size_t) h * D4 + 2 * c];
                float4 s1 = d4[(size_t) h * D4 + 2 * c + 1];
                if constexpr (!WEIGHTED)
                {
                    int4 wk = ((const int4*) (wn + (size_t) h * D))[c];
                    half4 v0 = *(half4*) &wk.x;
                    half4 v1 = *(half4*) &wk.z;
                    s0.x = __fmul_rn(s0.x, LOW_TO_FLOAT(v0.x)); s0.y = __fmul_rn(s0.y, HIGH_TO_FLOAT(v0.x));
                    s0.z = __fmul_rn(s0.z, LOW_TO_FLOAT(v0.y)); s0.w = __fmul_rn(s0.w, HIGH_TO_FLOAT(v0.y));
                    s1.x = __fmul_rn(s1.x, LOW_TO_FLOAT(v1.x)); s1.y = __fmul_rn(s1.y, HIGH_TO_FLOAT(v1.x));
                    s1.z = __fmul_rn(s1.z, LOW_TO_FLOAT(v1.y)); s1.w = __fmul_rn(s1.w, HIGH_TO_FLOAT(v1.y));
                }
                sa[h][k] = s0;
                sb[h][k] = s1;
            }
        }

    #pragma unroll
    for (int rr = 0; rr < RB; ++rr)
    {
        const int j = j0 + rr;
        if (j >= M) break;                                   // block-uniform
        int4 f[H][NI];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8) f[h][k] = ((const int4*) (fn + ((size_t) j * H + h) * D))[c];
            }
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8)
                {
                    const float4 s0 = sa[h][k];
                    const float4 s1 = sb[h][k];
                    half4 w0 = *(half4*) &f[h][k].x;
                    half4 w1 = *(half4*) &f[h][k].z;
                    a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                    a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                    a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                    a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                    a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                    a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                    a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                    a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
                }
            }
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[rr][h][warp] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < RB * H)
    {
        const int rr = threadIdx.x / H;
        const int h = threadIdx.x % H;
        const int j = j0 + rr;
        if (j < M)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[rr][h][w];
            dots[((size_t) r * (M + 1) + j) * H + h] = v;
        }
    }
}

// Launches the rows dots for D up to 3 * 8 * GR_THREADS_A columns; false otherwise (nothing launched)
static bool gr_dots_rows_try
(
    const float* s_p,
    const float* ws_p,
    const half* fn_p,
    const half* w_p,
    float* dots_p,
    const int M,
    const int D,
    const int R,
    cudaStream_t stream
)
{
    constexpr int RB = 4;
    const int ni = (D / 8 + GR_THREADS_A - 1) / GR_THREADS_A;
    if (ni < 1 || ni > 3 || D % 8) return false;
    dim3 grid((M + RB - 1) / RB + 1, R);
    #define ROWS(NI, WTD) gr_dots_rows_kernel<4, RB, NI, WTD><<<grid, GR_THREADS_A, 0, stream>>>(s_p, ws_p, fn_p, w_p, dots_p, M, D)
    if (ws_p) { if (ni == 1) ROWS(1, true); else if (ni == 2) ROWS(2, true); else ROWS(3, true); }
    else      { if (ni == 1) ROWS(1, false); else if (ni == 2) ROWS(2, false); else ROWS(3, false); }
    #undef ROWS
    cuda_check(cudaPeekAtLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Decode mix, load-pipelined pair for RDNA (gr_mix_rdna_try). The kernels above are latency-bound on gfx1100:
// their guarded loads (c < D8 with 128 threads over 320 chunks) compile to exec-masked branches with a
// vmcnt(0) wait in each, and the finalize waits on every 8-byte up load of its rank loop. Here:
//   gr_dots_rows2: NT = D / 8 / NI threads, so every thread owns exactly NI 8-column chunks per stream and no
//     load is guarded; the RB fn rows of a block are all issued before the first wait (the whole table is in
//     flight once), then the stream values. The sum-of-squares row runs in its own block.
//   gr_finalize2: one warp per column quad, lane l owns ranks (2 l, 2 l + 1) + 64 it, read as 16-byte loads;
//     all of a lane's up loads are issued before the dots are read, and the lane reduction runs on DPP.
// FOLD: the previous site's apply (x += post_prev * y_prev, hc_apply without comb) runs inside both kernels
// on the fly; the dots kernel uses the updated values, the finalize uses them and writes them back (each
// finalize warp owns its column quad, so the in-place write never races a read).
// Fixed reduction order everywhere, no atomics: run-to-run bit-reproducible.

#define GR2_FOLD_NONE 0
#define GR2_FOLD_W 1        // raw streams, norm weight applied here (no weighted copy)
#define GR2_FOLD_WTD 2      // pre-weighted copy given
#define GR2_FOLD_APPLY 3    // raw streams + pending apply, norm weight applied here

template <typename Y_T>
__device__ __forceinline__ void gr2_load_y8(const Y_T* y, int c8, float* v)
{
    if constexpr (std::is_same_v<Y_T, half>)
    {
        int4 k = ((const int4*) y)[c8];
        half2* h2 = (half2*) &k;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float2 f = __half22float2(h2[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
    }
    else
    {
        float4 a = ((const float4*) y)[2 * c8];
        float4 b = ((const float4*) y)[2 * c8 + 1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
}

template <int H, int NT, int RB, int NI, int MODE, typename Y_T>
__global__ __launch_bounds__(NT)
void gr_dots_rows2_kernel
(
    const float* __restrict__ streams,   // (R, H, D) raw (pre-apply when MODE == APPLY)
    const float* __restrict__ wstreams,  // (R, H, D) streams * w (MODE == WTD)
    const half* __restrict__ fn,         // (M, H * D)
    const half* __restrict__ wn,         // (H * D)
    const Y_T* __restrict__ py,          // (R, D) pending apply's sublayer output (MODE == APPLY)
    const float* __restrict__ ppost,     // (R, H) pending apply's gates (MODE == APPLY)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    constexpr int NW = NT / 32;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int D8 = D / 8;
    const float* xs = streams + (size_t) r * H * D;
    __shared__ float red[RB][H][NW];

    const int nrb = (M + RB - 1) / RB;
    const bool ss_block = blockIdx.x == nrb;
    const int j0 = blockIdx.x * RB;

    // fn rows first: independent of the activations, so the DRAM fetch starts at once
    int4 f[RB][H][NI];
    if (!ss_block)
    {
        #pragma unroll
        for (int rr = 0; rr < RB; ++rr)
        {
            const int j = min(j0 + rr, M - 1);
            #pragma unroll
            for (int h = 0; h < H; ++h)
                #pragma unroll
                for (int k = 0; k < NI; ++k)
                    f[rr][h][k] = ((const int4*) (fn + ((size_t) j * H + h) * D))[tid + k * NT];
        }
    }

    float pp[H];
    if constexpr (MODE == GR2_FOLD_APPLY)
    {
        #pragma unroll
        for (int h = 0; h < H; ++h) pp[h] = ppost[(size_t) r * H + h];
    }

    // Stream values (raw, updated when folding) of this thread's chunks
    float s[H][NI][8];
    #pragma unroll
    for (int k = 0; k < NI; ++k)
    {
        const int c = tid + k * NT;
        float yv[8];
        if constexpr (MODE == GR2_FOLD_APPLY) gr2_load_y8<Y_T>(py + (size_t) r * D, c, yv);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            const float* src = (MODE == GR2_FOLD_WTD && !ss_block) ? wstreams + (size_t) r * H * D : xs;
            float4 a = ((const float4*) (src + (size_t) h * D))[2 * c];
            float4 b = ((const float4*) (src + (size_t) h * D))[2 * c + 1];
            float* v = s[h][k];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
            if constexpr (MODE == GR2_FOLD_APPLY)
            {
                #pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = fmaf(pp[h], yv[e], v[e]);
            }
        }
    }

    if (ss_block)
    {
        // Sum of squares of the (updated) raw streams
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < NI; ++k)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    a = fmaf(s[h][k][e], s[h][k][e], a);
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[0][h][warp] = a;
        }
        __syncthreads();
        if (tid < H)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) v += red[0][tid][w];
            dots[((size_t) r * (M + 1) + M) * H + tid] = v;
        }
        return;
    }

    if constexpr (MODE == GR2_FOLD_W || MODE == GR2_FOLD_APPLY)
    {
        #pragma unroll
        for (int h = 0; h < H; ++h)
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                int4 wk = ((const int4*) (wn + (size_t) h * D))[tid + k * NT];
                half2* w2 = (half2*) &wk;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float2 wf = __half22float2(w2[i]);
                    s[h][k][2 * i] = __fmul_rn(s[h][k][2 * i], wf.x);
                    s[h][k][2 * i + 1] = __fmul_rn(s[h][k][2 * i + 1], wf.y);
                }
            }
    }

    #pragma unroll
    for (int rr = 0; rr < RB; ++rr)
    {
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                half2* w2 = (half2*) &f[rr][h][k];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float2 wf = __half22float2(w2[i]);
                    a = fmaf(s[h][k][2 * i], wf.x, a);
                    a = fmaf(s[h][k][2 * i + 1], wf.y, a);
                }
            }
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[rr][h][warp] = a;
        }
    }
    __syncthreads();
    if (tid < RB * H)
    {
        const int rr = tid / H;
        const int h = tid % H;
        const int j = j0 + rr;
        if (j < M)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) v += red[rr][h][w];
            dots[((size_t) r * (M + 1) + j) * H + h] = v;
        }
    }
}

#define GR2_THREADS_C 128

template <int H, int LR, bool HALF_OUT, bool APPLY, typename Y_T>
__global__ __launch_bounds__(GR2_THREADS_C)
void gr_finalize2_kernel
(
    float* __restrict__ streams,         // (R, H, D) raw; updated in place when APPLY
    const float* __restrict__ dots,      // (R, M + 1, H)
    const half* __restrict__ upt,        // (H, D / 4, LR, 4)
    const half* __restrict__ w,          // (H * D)
    const Y_T* __restrict__ py,          // (R, D) (APPLY)
    const float* __restrict__ ppost,     // (R, H) (APPLY)
    float* __restrict__ post,            // (R, H) or nullptr
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const float rms_eps
)
{
    constexpr int NIT = LR / 64;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int D4 = D / 4;
    const int c = blockIdx.x * (GR2_THREADS_C / 32) + warp;     // this warp's column quad
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Up loads first (independent of the dots): ranks 2 lane + 64 it and the next, 16 bytes each
    int4 u[NIT][H];
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
        #pragma unroll
        for (int h = 0; h < H; ++h)
            u[it][h] = *(const int4*) (upt + ((((size_t) h * D4 + c) * LR + 2 * lane + 64 * it) * 4));

    // This quad's stream values, norm weight and (folded) apply inputs
    float4 xv[H];
    half4 wq[H];
    float* xr = streams + (size_t) r * H * D;
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        xv[h] = ((const float4*) (xr + (size_t) h * D))[c];
        wq[h] = *(const half4*) (w + (size_t) h * D + 4 * c);
    }
    if constexpr (APPLY)
    {
        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half4 yh = *(const half4*) (py + (size_t) r * D + 4 * c);
            yv = make_float4(LOW_TO_FLOAT(yh.x), HIGH_TO_FLOAT(yh.x), LOW_TO_FLOAT(yh.y), HIGH_TO_FLOAT(yh.y));
        }
        else
            yv = ((const float4*) (py + (size_t) r * D))[c];
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            const float p = ppost[(size_t) r * H + h];
            xv[h].x = fmaf(p, yv.x, xv[h].x);
            xv[h].y = fmaf(p, yv.y, xv[h].y);
            xv[h].z = fmaf(p, yv.z, xv[h].z);
            xv[h].w = fmaf(p, yv.w, xv[h].w);
        }
    }

    __shared__ float rmr_s[H];
    __shared__ float t_s[LR];
    if (tid < H)
        rmr_s[tid] = rsqrtf(dr[(size_t) M * H + tid] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = tid; i < LR; i += GR2_THREADS_C)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && tid < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + tid) * H + h], v);
        post[(size_t) r * H + tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    float g[H][4];
    #pragma unroll
    for (int h = 0; h < H; ++h) g[h][0] = g[h][1] = g[h][2] = g[h][3] = 0.0f;
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const float t0 = t_s[2 * lane + 64 * it];
        const float t1 = t_s[2 * lane + 64 * it + 1];
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            half2* u2 = (half2*) &u[it][h];
            float2 a01 = __half22float2(u2[0]), a23 = __half22float2(u2[1]);
            float2 b01 = __half22float2(u2[2]), b23 = __half22float2(u2[3]);
            g[h][0] = fmaf(t1, b01.x, fmaf(t0, a01.x, g[h][0]));
            g[h][1] = fmaf(t1, b01.y, fmaf(t0, a01.y, g[h][1]));
            g[h][2] = fmaf(t1, b23.x, fmaf(t0, a23.x, g[h][2]));
            g[h][3] = fmaf(t1, b23.y, fmaf(t0, a23.y, g[h][3]));
        }
    }
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            float v = g[h][e];
            v += hc_xor<16>(v);
            v += hc_xor<8>(v);
            v += hc_xor<4>(v);
            v += hc_xor<2>(v);
            v += hc_xor<1>(v);
            g[h][e] = v;
        }
    if (lane != 0) return;

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        const float4 sv = xv[h];
        const float coef = rmr_s[h] * inv_h;
        o.x = fmaf(sigmoidf_(g[h][0]) * coef * LOW_TO_FLOAT(wq[h].x),  sv.x, o.x);
        o.y = fmaf(sigmoidf_(g[h][1]) * coef * HIGH_TO_FLOAT(wq[h].x), sv.y, o.y);
        o.z = fmaf(sigmoidf_(g[h][2]) * coef * LOW_TO_FLOAT(wq[h].y),  sv.z, o.z);
        o.w = fmaf(sigmoidf_(g[h][3]) * coef * HIGH_TO_FLOAT(wq[h].y), sv.w, o.w);
        if constexpr (APPLY) ((float4*) (xr + (size_t) h * D))[c] = sv;
    }
    if (HALF_OUT)
    {
        half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
        out2[c * 2] = __floats2half2_rn(o.x, o.y);
        out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
    }
    else
        ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
}

// ---------------------------------------------------------------------------------------------
// One persistent kernel per decode site (R = 1): the previous site's apply, the dots and the finalize in a single
// launch. Phase 0 issues the block's share of the stream stack (L2) and then its proj dot rows and up column
// quads (activation-independent, the whole 13 MB table in flight at once) and builds the updated, weighted stack
// in LDS plus the per-stream sums of squares (every block, same order). Phase 1: one warp per (proj row j,
// stream h) pair against the LDS stack. A grid barrier (a monotonic 64-bit counter in a zeroed 2-word buffer, never
// reset, so graph replays work) separates it from phase 2: every block derives rmr and t from the dots
// in a fixed order, one warp per column quad finishes mixed and writes the updated streams back. The grid is
// one block per CU (co-resident: 512 threads, ~190 VGPRs, 43 KB LDS). No float atomics: run-to-run
// bit-reproducible; dots differ from the pair only in summation order.
#define GRS_NT 512
#define GRS_NW (GRS_NT / 32)

__device__ __forceinline__ float grs_xor_sum32(float v)
{
    v += hc_permlanex16(v);
    v += hc_xor16<8>(v);
    v += hc_xor16<4>(v);
    v += hc_xor16<2>(v);
    v += hc_xor16<1>(v);
    return v;
}

// Grid barrier on a monotonic 64-bit arrival counter (the 8 zeroed bytes of bar, never reset): each block adds 1
// and waits until the count reaches the end of its own round of nb arrivals. A stale read of the counter can only
// be lower than the truth, so it delays and never releases early. (A counter + generation scheme released early
// about once per 250 launches on gfx1100: the plain glc load of the generation can hit a stale GL1 line.)
__device__ __forceinline__ void grs_grid_barrier(unsigned* bar, unsigned nb)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long* cnt = (unsigned long long*) bar;
        const unsigned long long old = __hip_atomic_fetch_add(cnt, 1ull, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned long long target = (old / nb + 1) * nb;
        while (__hip_atomic_load(cnt, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// D: hidden, LR: low rank, POST: site form (M = LR + H proj rows) vs final mixer (M = LR)
// APPLY: fold the previous site's apply (x += pp * y); y is Y_T
template <int H, int D, int LR, bool POST, bool APPLY, typename Y_T, bool I8>
__global__ __launch_bounds__(GRS_NT) __attribute__((amdgpu_waves_per_eu(8, 8)))
void gr_site_kernel
(
    float* __restrict__ x,               // (H, D) raw streams, updated in place when APPLY
    const Y_T* __restrict__ y,           // (D)
    const float* __restrict__ pp,        // (H)
    const void* __restrict__ fn,         // (M, H * D) half, or int8 (I8)
    const half* __restrict__ fn_s,       // (M, H * D / 128) group scales (I8)
    const void* __restrict__ upt,        // (H, D / 4, LR, 4) half, or int8 (I8)
    const half* __restrict__ upt_s,      // (H, D / 4, LR / 64, 4) group scales (I8)
    const half* __restrict__ w,          // (H * D)
    float* __restrict__ dots,            // (M * H) workspace
    float* __restrict__ post,            // (H) out (POST)
    half* __restrict__ mixed,            // (D) out
    unsigned* __restrict__ bar,          // (2) zero-initialised once
    const float rms_eps
)
{
    constexpr int M = LR + (POST ? H : 0);
    constexpr int NPAIR = M * H;
    constexpr int D4 = D / 4;
    constexpr int D8 = D / 8;
    constexpr int KC = D8 / 32;          // 8-column chunks per lane per dot row
    constexpr int NIT = LR / 64;
    static_assert(D8 % 32 == 0 && LR % 64 == 0, "shape");

    const int nb = gridDim.x;
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    __shared__ float4 sw4[H * D4];       // updated, weighted streams
    __shared__ float t_s[LR];
    __shared__ float red[H][GRS_NW];
    __shared__ float rmr_s[H];

    // Work split: pairs [p0, p1), quads [q0, q1); warp w takes pairs p0 + w, p0 + w + GRS_NW and quad q0 + w
    const int p0 = (int) ((long) b * NPAIR / nb), p1 = (int) ((long) (b + 1) * NPAIR / nb);
    const int q0 = (int) ((long) b * D4 / nb), q1 = (int) ((long) (b + 1) * D4 / nb);
    const int pa = p0 + warp;
    const bool has_a = pa < p1;
    const int qc = q0 + warp;
    const bool has_q = warp < 8 && qc < q1;

    float ppr[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) ppr[h] = APPLY ? pp[h] : 0.0f;

    // Phase 0a: this thread's share of the stream stack (L2-resident), issued first so the stack can be built
    // while the weight loads below are still in flight (vmcnt retires in issue order)
    constexpr int SK = H * D4 / GRS_NT;
    static_assert(H * D4 % GRS_NT == 0, "stack split");
    float4 xs[SK], ys[SK];
    half2 ws[SK][2];
    #pragma unroll
    for (int k = 0; k < SK; ++k)
    {
        const int i = tid + GRS_NT * k, h = i / D4, c = i % D4;
        xs[k] = ((const float4*) x)[i];
        const half2* w2 = (const half2*) (w + 4 * (size_t) i);
        ws[k][0] = w2[0]; ws[k][1] = w2[1];
        if constexpr (APPLY)
        {
            if constexpr (std::is_same_v<Y_T, half>)
            {
                const half2* y2 = (const half2*) (y + 4 * c);
                float2 lo = __half22float2(y2[0]), hi = __half22float2(y2[1]);
                ys[k] = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else ys[k] = ((const float4*) y)[c];
        }
        (void) h;
    }

    // Phase 0b: weight loads (activation-independent)
    // fp16: 8 columns per int4 (KC per lane); int8: 16 columns per int4 (KQ per lane), one scale per 128
    constexpr int KQ = D / 16 / 32;
    int4 fa[I8 ? KQ : KC];
    half fsc[I8 ? KQ : 1];
    {
        const int pc = min(pa, NPAIR - 1);
        if constexpr (I8)
        {
            const int4* ra = (const int4*) ((const int8_t*) fn + (size_t) pc * D);
            #pragma unroll
            for (int k = 0; k < KQ; ++k) fa[k] = ra[lane + 32 * k];
            #pragma unroll
            for (int k = 0; k < KQ; ++k) fsc[k] = fn_s[(size_t) pc * (D / 128) + (lane + 32 * k) / 8];
        }
        else
        {
            const int4* ra = (const int4*) ((const half*) fn + (size_t) pc * D);
            #pragma unroll
            for (int k = 0; k < KC; ++k) fa[k] = ra[lane + 32 * k];
        }
    }
    // up: lane ranks 2 lane + 64 it and the next, 4 columns: 16 bytes fp16 or 8 bytes int8 (scale per 64 ranks:
    // one per (it, h, column), uniform across the warp)
    int4 u[I8 ? 1 : NIT][H];
    int2 u8[I8 ? NIT : 1][H];
    __shared__ int2 usc_s[I8 ? 8 * H * NIT : 1];        // this block's up scales (<= 8 quads), 4 halves each
    if constexpr (I8)
    {
        const int nq = q1 - q0;
        if (tid < nq * H * NIT)
        {
            const int qi = tid / (H * NIT), r = tid % (H * NIT), h = r / NIT, it = r % NIT;
            usc_s[tid] = *(const int2*) (upt_s + (((size_t) h * D4 + q0 + qi) * NIT + it) * 4);
        }
    }
    const int qcl = min(qc, D4 - 1);
    if (warp < 8)
    {
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const size_t e = (((size_t) h * D4 + qcl) * LR + 2 * lane + 64 * it) * 4;
                if constexpr (I8)
                {
                    u8[it][h] = *(const int2*) ((const int8_t*) upt + e);
                }
                else u[it][h] = *(const int4*) ((const half*) upt + e);
            }
    }

    // Build the updated weighted stack + per-thread sums of squares
    float ss[H] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int k = 0; k < SK; ++k)
    {
        const int i = tid + GRS_NT * k, h = i / D4;
        float4 xv = xs[k];
        float p = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) p = hh == h ? ppr[hh] : p;
        if constexpr (APPLY)
        {
            xv.x = fmaf(p, ys[k].x, xv.x); xv.y = fmaf(p, ys[k].y, xv.y);
            xv.z = fmaf(p, ys[k].z, xv.z); xv.w = fmaf(p, ys[k].w, xv.w);
        }
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
            if (hh == h) ss[hh] = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, fmaf(xv.w, xv.w, ss[hh]))));
        float2 w01 = __half22float2(ws[k][0]), w23 = __half22float2(ws[k][1]);
        sw4[i] = make_float4(__fmul_rn(xv.x, w01.x), __fmul_rn(xv.y, w01.y),
                             __fmul_rn(xv.z, w23.x), __fmul_rn(xv.w, w23.y));
    }
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float v = hc_warp_sum_lane0(ss[h]);
        if (lane == 0) red[h][warp] = v;
    }
    __syncthreads();
    if (tid < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < GRS_NW; ++i) v += red[tid][i];
        rmr_s[tid] = rsqrtf(v / (float) D + rms_eps);
    }

    // Phase 1: dots for this warp's pairs (pair p = j * H + h: fn row j, stream h)
    {
        auto dot = [&](int p, int4* f) -> float
        {
            const int h = p % H;
            const float4* s = sw4 + h * D4;
            float a = 0.0f;
            if constexpr (I8)
            {
                #pragma unroll
                for (int k = 0; k < KQ; ++k)
                {
                    const int c16 = lane + 32 * k;
                    const int* q4 = (const int*) &f[k];
                    float pk = 0.0f;
                    #pragma unroll
                    for (int v = 0; v < 4; ++v)
                    {
                        const float4 sv = s[4 * c16 + v];
                        const int q = q4[v];
                        pk = fmaf(sv.x, (float) (int8_t) (q), pk);
                        pk = fmaf(sv.y, (float) (int8_t) (q >> 8), pk);
                        pk = fmaf(sv.z, (float) (int8_t) (q >> 16), pk);
                        pk = fmaf(sv.w, (float) (int8_t) (q >> 24), pk);
                    }
                    a = fmaf(__half2float(fsc[k]), pk, a);
                }
                return hc_warp_sum_lane0(a);
            }
            #pragma unroll
            for (int k = 0; k < KC; ++k)
            {
                const int c8 = lane + 32 * k;
                const float4 s0 = s[2 * c8], s1 = s[2 * c8 + 1];
                const half2* f2 = (const half2*) &f[k];
                float2 a01 = __half22float2(f2[0]), a23 = __half22float2(f2[1]);
                float2 b01 = __half22float2(f2[2]), b23 = __half22float2(f2[3]);
                a = fmaf(s0.x, a01.x, a); a = fmaf(s0.y, a01.y, a);
                a = fmaf(s0.z, a23.x, a); a = fmaf(s0.w, a23.y, a);
                a = fmaf(s1.x, b01.x, a); a = fmaf(s1.y, b01.y, a);
                a = fmaf(s1.z, b23.x, a); a = fmaf(s1.w, b23.y, a);
            }
            return hc_warp_sum_lane0(a);
        };
        __syncthreads();
        float da = dot(min(pa, NPAIR - 1), fa);
        if (has_a && lane == 0) __hip_atomic_store(dots + pa, da, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    // Quad epilogue inputs (independent of the dots)
    float4 xq[H];
    half2 wq[H][2];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        xq[h] = ((const float4*) (x + (size_t) h * D))[qcl];
        const half2* w2 = (const half2*) (w + (size_t) h * D + 4 * qcl);
        wq[h][0] = w2[0]; wq[h][1] = w2[1];
    }
    if constexpr (APPLY)
    {
        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            const half2* y2 = (const half2*) (y + 4 * qcl);
            float2 lo = __half22float2(y2[0]), hi = __half22float2(y2[1]);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else yv = ((const float4*) y)[qcl];
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            xq[h].x = fmaf(ppr[h], yv.x, xq[h].x); xq[h].y = fmaf(ppr[h], yv.y, xq[h].y);
            xq[h].z = fmaf(ppr[h], yv.z, xq[h].z); xq[h].w = fmaf(ppr[h], yv.w, xq[h].w);
        }
    }

    grs_grid_barrier(bar, nb);

    // Phase 2: t from the dots (fixed order, every block the same)
    const float inv_h = 1.0f / (float) H;
    for (int i = tid; i < LR; i += GRS_NT)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[h], __hip_atomic_load(dots + i * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (POST && b == 0 && tid < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[h], __hip_atomic_load(dots + (LR + tid) * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), v);
        post[tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();
    if (!has_q) return;

    float g[H][4];
    #pragma unroll
    for (int h = 0; h < H; ++h) g[h][0] = g[h][1] = g[h][2] = g[h][3] = 0.0f;
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const float t0 = t_s[2 * lane + 64 * it];
        const float t1 = t_s[2 * lane + 64 * it + 1];
        if constexpr (I8)
        {
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const int qa = u8[it][h].x, qb = u8[it][h].y;      // rank i: 4 columns, rank i + 1: 4 columns
                const int2 usv = usc_s[(warp * H + h) * NIT + it];
                const half2* us2 = (const half2*) &usv;
                const float2 s01 = __half22float2(us2[0]), s23 = __half22float2(us2[1]);
                const float sc[4] = { s01.x, s01.y, s23.x, s23.y };
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const float pe = fmaf(t1, (float) (int8_t) (qb >> (8 * e)), t0 * (float) (int8_t) (qa >> (8 * e)));
                    g[h][e] = fmaf(sc[e], pe, g[h][e]);
                }
            }
            continue;
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            const half2* u2 = (const half2*) &u[it][h];
            float2 a01 = __half22float2(u2[0]), a23 = __half22float2(u2[1]);
            float2 b01 = __half22float2(u2[2]), b23 = __half22float2(u2[3]);
            g[h][0] = fmaf(t1, b01.x, fmaf(t0, a01.x, g[h][0]));
            g[h][1] = fmaf(t1, b01.y, fmaf(t0, a01.y, g[h][1]));
            g[h][2] = fmaf(t1, b23.x, fmaf(t0, a23.x, g[h][2]));
            g[h][3] = fmaf(t1, b23.y, fmaf(t0, a23.y, g[h][3]));
        }
    }
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int e = 0; e < 4; ++e) g[h][e] = grs_xor_sum32(g[h][e]);
    if (lane != 0) return;

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        const float4 sv = xq[h];
        const float coef = rmr_s[h] * inv_h;
        float2 w01 = __half22float2(wq[h][0]), w23 = __half22float2(wq[h][1]);
        o.x = fmaf(sigmoidf_(g[h][0]) * coef * w01.x, sv.x, o.x);
        o.y = fmaf(sigmoidf_(g[h][1]) * coef * w01.y, sv.y, o.y);
        o.z = fmaf(sigmoidf_(g[h][2]) * coef * w23.x, sv.z, o.z);
        o.w = fmaf(sigmoidf_(g[h][3]) * coef * w23.y, sv.w, o.w);
        if constexpr (APPLY) ((float4*) (x + (size_t) h * D))[qc] = sv;
    }
    half2* out2 = (half2*) mixed;
    out2[qc * 2] = __floats2half2_rn(o.x, o.y);
    out2[qc * 2 + 1] = __floats2half2_rn(o.z, o.w);
}

// Launch a site kernel on a grid of co-resident blocks: as many as the device holds at once (occupancy x
// multiprocessors; gfx11 reports WGPs), capped at 96. The work split needs >= 81 blocks; false otherwise.
template <typename K, typename Y_T>
static bool gr_site_launch(K kernel, float* x, const Y_T* y, const float* pp, const void* fn, const half* fn_s,
                           const void* upt, const half* upt_s, const half* w, float* dots, float* post, half* mixed,
                           unsigned* bar, float eps, cudaStream_t stream)
{
    int dev; cudaGetDevice(&dev);
    static int nbs[16][8] = { { 0 } };
    static int slot_next = 0;
    static const void* slots[8] = { nullptr };
    int slot = -1;
    for (int i = 0; i < slot_next; ++i) if (slots[i] == (const void*) kernel) slot = i;
    if (slot < 0) { slot = slot_next++; slots[slot] = (const void*) kernel; }
    if (!nbs[dev][slot])
    {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        // HIP's occupancy query does not model gfx11 WGP mode (it reports WGPs as multiprocessors but sizes
        // LDS and VGPRs against one CU) and returns 0 for this kernel. A WGP holds two of its blocks when each
        // fits one CU: 16 wave32 waves over 2 SIMDs at <= 192 VGPRs (1536 per SIMD lane on gfx1100) and <= 64 KB
        // of LDS. Measured: 96 blocks run co-resident on gfx1100's 48 WGPs (the barrier completes)
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*) kernel);
        int per = 0;
        if (strncmp(p.gcnArchName, "gfx11", 5) == 0 && fa.numRegs <= 192 && fa.sharedSizeBytes <= 65536) per = 2;
        else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, GRS_NT, 0);
        nbs[dev][slot] = std::min(96, per * p.multiProcessorCount);
        if (nbs[dev][slot] < 81) nbs[dev][slot] = -1;
    }
    const int nb = nbs[dev][slot];
    if (nb < 0) return false;
    kernel<<<nb, GRS_NT, 0, stream>>>(x, y, pp, fn, fn_s, upt, upt_s, w, dots, post, mixed, bar, eps);
    cuda_check(cudaPeekAtLastError());
    return true;
}

// The pipelined pair for the shapes it is instantiated for; false (nothing launched) otherwise. y_half: the
// pending apply's y dtype. RB: fn rows per dots block (EXL3_GR2_RB, default 4).
static bool gr_mix_rdna_try
(
    float* s_p, const float* ws_p, const half* fn_p, const half* upt_p, const half* w_p,
    const void* py, bool y_half, const float* ppost,
    float* dots_p, float* post_p, void* mixed_p, bool half_out,
    const int M, const int LR, const int D, const int R, const float rms_eps, cudaStream_t stream,
    unsigned* bar = nullptr
)
{
    static const int env_off = [] { const char* e = getenv("EXL3_GR2"); return e && e[0] == '0'; }();
    static const int RBc = [] { const char* e = getenv("EXL3_GR2_RB"); return e ? atoi(e) : 4; }();
    if (env_off) return false;

    // One persistent launch per site at R = 1 (needs the barrier words and a grid of >= 81 co-resident blocks)
    static const bool site_off = [] { const char* e = getenv("EXL3_GR_SITE"); return e && e[0] == '0'; }();
    if (!site_off && bar && R == 1 && half_out && D == 2560 && LR == 320)
    {
        const bool apply = py != nullptr;
        bool done = false;
        #define SITE(POST_, AP, Y_T) done = gr_site_launch(gr_site_kernel<4, 2560, 320, POST_, AP, Y_T, false>, \
            s_p, (const Y_T*) py, ppost, fn_p, nullptr, upt_p, nullptr, w_p, dots_p, post_p, (half*) mixed_p, bar, rms_eps, stream)
        #define SITE_P(POST_) if (!apply) SITE(POST_, false, float); else if (y_half) SITE(POST_, true, half); else SITE(POST_, true, float);
        if (post_p) { SITE_P(true) } else { SITE_P(false) }
        #undef SITE_P
        #undef SITE
        if (done) return true;
    }
    const int D8 = D / 8;
    if (D % 16 || (LR != 320 && LR != 512)) return false;
    int nt = 0, ni = 0;
    if (D8 == 320) { nt = 160; ni = 2; }
    else if (D8 == 512) { nt = 256; ni = 2; }
    else if (D8 == 128) { nt = 128; ni = 1; }
    else return false;
    const int RB = RBc == 2 ? 2 : 4;
    const bool apply = py != nullptr;
    const int mode = apply ? GR2_FOLD_APPLY : (ws_p ? GR2_FOLD_WTD : GR2_FOLD_W);

    dim3 grid_a((M + RB - 1) / RB + 1, R);
    #define DOTS2(NT_, RB_, NI_, MODE_, Y_T) gr_dots_rows2_kernel<4, NT_, RB_, NI_, MODE_, Y_T><<<grid_a, NT_, 0, stream>>> \
        (s_p, ws_p, fn_p, w_p, (const Y_T*) py, ppost, dots_p, M, D)
    #define DOTS2_M(NT_, RB_, NI_) \
        if (mode == GR2_FOLD_APPLY) { if (y_half) DOTS2(NT_, RB_, NI_, GR2_FOLD_APPLY, half); else DOTS2(NT_, RB_, NI_, GR2_FOLD_APPLY, float); } \
        else if (mode == GR2_FOLD_WTD) DOTS2(NT_, RB_, NI_, GR2_FOLD_WTD, float); \
        else DOTS2(NT_, RB_, NI_, GR2_FOLD_W, float);
    #define DOTS2_RB(NT_, NI_) if (RB == 2) { DOTS2_M(NT_, 2, NI_) } else { DOTS2_M(NT_, 4, NI_) }
    if (nt == 160) { DOTS2_RB(160, 2) }
    else if (nt == 256) { DOTS2_RB(256, 2) }
    else { DOTS2_RB(128, 1) }
    #undef DOTS2_RB
    #undef DOTS2_M
    #undef DOTS2
    cuda_check(cudaPeekAtLastError());

    dim3 grid_c(D / 4 / (GR2_THREADS_C / 32), R);
    #define FIN2(LR_, HO, AP, Y_T) gr_finalize2_kernel<4, LR_, HO, AP, Y_T><<<grid_c, GR2_THREADS_C, 0, stream>>> \
        (s_p, dots_p, upt_p, w_p, (const Y_T*) py, ppost, post_p, mixed_p, D, rms_eps)
    #define FIN2_A(LR_, HO) \
        if (!apply) FIN2(LR_, HO, false, float); else if (y_half) FIN2(LR_, HO, true, half); else FIN2(LR_, HO, true, float);
    #define FIN2_H(LR_) if (half_out) { FIN2_A(LR_, true) } else { FIN2_A(LR_, false) }
    if (LR == 320) { FIN2_H(320) } else { FIN2_H(512) }
    #undef FIN2_H
    #undef FIN2_A
    #undef FIN2
    cuda_check(cudaPeekAtLastError());
    return true;
}


// Site kernel with int8 tables (symmetric absmax, one fp16 scale per 128 proj columns along H * D and per 64 up
// ranks per output channel; GatedResidual quantizes them at load under EXL3_GR_I8). R = 1, the Qwen3.8 shape
// only; false (nothing launched) otherwise, and the caller takes the fp16 path.
static bool gr_site_i8_try
(
    float* s_p, const void* py, bool y_half, const float* ppost,
    const int8_t* fn_q, const half* fn_s, const int8_t* upt_q, const half* upt_s, const half* w_p,
    float* dots_p, float* post_p, half* mixed_p, unsigned* bar,
    const int LR, const int D, const int R, const float rms_eps, cudaStream_t stream
)
{
    if (R != 1 || D != 2560 || LR != 320 || !bar) return false;
    const bool apply = py != nullptr;
    bool done = false;
    #define SITE(POST_, AP, Y_T) done = gr_site_launch(gr_site_kernel<4, 2560, 320, POST_, AP, Y_T, true>, \
        s_p, (const Y_T*) py, ppost, fn_q, fn_s, upt_q, upt_s, w_p, dots_p, post_p, mixed_p, bar, rms_eps, stream)
    #define SITE_P(POST_) if (!apply) SITE(POST_, false, float); else if (y_half) SITE(POST_, true, half); else SITE(POST_, true, float);
    if (post_p) { SITE_P(true) } else { SITE_P(false) }
    #undef SITE_P
    #undef SITE
    return done;
}
