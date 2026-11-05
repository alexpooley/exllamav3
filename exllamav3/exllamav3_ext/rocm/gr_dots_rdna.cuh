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
// One persistent kernel per decode site for R = 1..4 rows (decode and the MTP verify): the previous site's apply,
// the dots and the finalize in a single launch, the tables read once for all rows. Block b owns stream h = b % 4,
// 1 / 24 of the proj rows (one warp each) and up to 8 column quads. Phase 0 issues the proj rows and up quads
// (activation-independent, the whole table in flight at once), stages the quads' stream values in LDS and builds
// the updated, weighted stack of stream h for every row in LDS with its sums of squares. Phase 1: R dots per proj
// row against the stack. A grid barrier (a monotonic 64-bit counter in a zeroed 2-word buffer, so graph
// replays work) separates it from phase 2: every block derives rmr and t from the dots in a fixed order; one warp
// per column quad accumulates the up gates for a pair of rows at a time, reduce-scatters them so each lane owns
// one (row, stream, column) value, and finishes mixed and the updated streams. 96 blocks of 512 threads,
// co-resident (<= 192 VGPRs, <= 50 KB LDS). No float atomics: run-to-run bit-reproducible; results differ from
// the pair only in summation order. I8: int8 tables with fp16 group scales (EXL3_GR_I8).
#define GRS_NT 512
#define GRS_NW (GRS_NT / 32)
#define GRS_NB 96             // grid: block b owns stream b % 4 and 1 / 24 of the proj rows

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
// Opaque register touch: stops the compiler from hoisting the per-row unpacking of loop-invariant packed
// weights out of the row loop (it would keep every converted fp32 value live: hundreds of VGPRs)
__device__ __forceinline__ void grs_opaque(int4& v) { asm volatile("" : "+v"(v.x), "+v"(v.y), "+v"(v.z), "+v"(v.w)); }
__device__ __forceinline__ void grs_opaque(int2& v) { asm volatile("" : "+v"(v.x), "+v"(v.y)); }

template <typename Y_T>
__device__ __forceinline__ float4 grs_load_y4(const Y_T* y, int c)
{
    if constexpr (std::is_same_v<Y_T, half>)
    {
        const half2* y2 = (const half2*) (y + 4 * c);
        float2 lo = __half22float2(y2[0]), hi = __half22float2(y2[1]);
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    else return ((const float4*) y)[c];
}

__device__ __forceinline__ float4 grs_fma4(float p, float4 y, float4 x)
{
    return make_float4(fmaf(p, y.x, x.x), fmaf(p, y.y, x.y), fmaf(p, y.z, x.z), fmaf(p, y.w, x.w));
}

template <int H, int D, int LR, int R, bool POST, bool APPLY, typename Y_T, bool I8>
__global__ __launch_bounds__(GRS_NT) __attribute__((amdgpu_waves_per_eu(8, 8)))
void gr_site_kernel
(
    float* __restrict__ x,               // (R, H, D) raw streams, updated in place when APPLY
    const Y_T* __restrict__ y,           // (R, D)
    const float* __restrict__ pp,        // (R, H)
    const void* __restrict__ fn,         // (M, H * D) half, or int8 (I8)
    const half* __restrict__ fn_s,       // (M, H * D / 128) (I8)
    const void* __restrict__ upt,        // (H, D / 4, LR, 4) half, or int8 (I8)
    const half* __restrict__ upt_s,      // (H, D / 4, LR / 64, 4) (I8)
    const half* __restrict__ w,          // (H * D)
    float* __restrict__ dots,            // (R, M + 1, H): rows j < M dots, row M sums of squares
    float* __restrict__ post,            // (R, H) (POST)
    half* __restrict__ mixed,            // (R, D)
    unsigned* __restrict__ bar,
    const float rms_eps
)
{
    // Drop this CU's L0 / GL1 lines before the first read: the streams are rewritten in place by every site (and the
    // pending y by the sublayer), and a line another kernel left in the non-coherent caches would be stale
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    constexpr int M = LR + (POST ? H : 0);
    constexpr int D4 = D / 4;
    constexpr int KC = D / 8 / 32;       // fp16: 8 columns per int4
    constexpr int KQ = D / 16 / 32;      // int8: 16 columns per int4
    constexpr int NIT = LR / 64;
    constexpr int NBH = GRS_NB / H;        // blocks per stream
    constexpr int MQ = 8;                // quads per block, at most
    static_assert(D / 8 % 32 == 0 && LR % 64 == 0, "shape");

    const int nb = gridDim.x;            // == GRS_NB
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int hb = b % H, bi = b / H;

    __shared__ float4 sw4[R][D4];                  // updated, weighted stream hb, all rows
    __shared__ float t_s[R][LR];
    __shared__ float4 xq_s[MQ][R][H];              // the quads' raw streams
    __shared__ float4 yq_s[MQ][R];
    __shared__ int2 usc_s[I8 ? MQ * H * NIT : 1];
    __shared__ float red[R][GRS_NW];
    __shared__ float rmr_s[R][H];

    const int j0 = bi * M / NBH, j1 = (bi + 1) * M / NBH;
    const int ja = j0 + warp;
    const bool has_j = ja < j1;
    const int jc = min(ja, M - 1);
    const int q0 = b * D4 / nb, q1 = (b + 1) * D4 / nb;
    const int qc = q0 + warp;
    const bool has_q = warp < MQ && qc < q1;
    const int qcl = min(qc, D4 - 1);

    // Phase 0a: weights first (activation-independent): proj row jc of stream hb, up quads (warps < MQ)
    int4 fa[I8 ? KQ : KC];
    half fsc[I8 ? KQ : 1];
    if constexpr (I8)
    {
        const int4* ra = (const int4*) ((const int8_t*) fn + ((size_t) jc * H + hb) * D);
        #pragma unroll
        for (int k = 0; k < KQ; ++k) fa[k] = ra[lane + 32 * k];
        #pragma unroll
        for (int k = 0; k < KQ; ++k) fsc[k] = fn_s[((size_t) jc * H + hb) * (D / 128) + (lane + 32 * k) / 8];
    }
    else
    {
        const int4* ra = (const int4*) ((const half*) fn + ((size_t) jc * H + hb) * D);
        #pragma unroll
        for (int k = 0; k < KC; ++k) fa[k] = ra[lane + 32 * k];
    }
    int4 u[I8 ? 1 : NIT][H];
    int2 u8[I8 ? NIT : 1][H];
    if (warp < MQ)
    {
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const size_t e = (((size_t) h * D4 + qcl) * LR + 2 * lane + 64 * it) * 4;
                if constexpr (I8) u8[it][h] = *(const int2*) ((const int8_t*) upt + e);
                else u[it][h] = *(const int4*) ((const half*) upt + e);
            }
    }

    // Phase 0b: quad inputs and up scales into LDS
    {
        const int nq = q1 - q0;
        for (int i = tid; i < nq * R * (H + 1); i += GRS_NT)
        {
            const int qi = i / (R * (H + 1)), rem = i % (R * (H + 1)), r = rem / (H + 1), h = rem % (H + 1);
            if (h < H) xq_s[qi][r][h] = ((const float4*) (x + ((size_t) r * H + h) * D))[q0 + qi];
            else if constexpr (APPLY) yq_s[qi][r] = grs_load_y4<Y_T>(y + (size_t) r * D, q0 + qi);
        }
        if constexpr (I8)
            for (int i = tid; i < nq * H * NIT; i += GRS_NT)
            {
                const int qi = i / (H * NIT), rem = i % (H * NIT), h = rem / NIT, it = rem % NIT;
                usc_s[i] = *(const int2*) (upt_s + (((size_t) h * D4 + q0 + qi) * NIT + it) * 4);
            }
    }

    // Phase 0c: updated weighted stack of stream hb (all rows) + per-row sums of squares. Flat over (row,
    // column quad) with every load issued before use (one L2 round trip, not one per row)
    {
        constexpr int K0 = (R * D4 + GRS_NT - 1) / GRS_NT;
        float4 xv[K0], yv[K0];
        #pragma unroll
        for (int k = 0; k < K0; ++k)
        {
            const int i = min(tid + GRS_NT * k, R * D4 - 1), r = i / D4, c = i % D4;
            xv[k] = ((const float4*) (x + ((size_t) r * H + hb) * D))[c];
            if constexpr (APPLY) yv[k] = grs_load_y4<Y_T>(y + (size_t) r * D, c);
        }
        float ss[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) ss[r] = 0.0f;
        #pragma unroll
        for (int k = 0; k < K0; ++k)
        {
            const int i = tid + GRS_NT * k;
            if (i >= R * D4) break;
            const int r = i / D4, c = i % D4;
            float4 v = xv[k];
            if constexpr (APPLY) v = grs_fma4(pp[r * H + hb], yv[k], v);
            const float sq = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w)));
            #pragma unroll
            for (int rr = 0; rr < R; ++rr) if (rr == r) ss[rr] += sq;
            const half2* w2 = (const half2*) (w + (size_t) hb * D + 4 * c);
            float2 w01 = __half22float2(w2[0]), w23 = __half22float2(w2[1]);
            sw4[r][c] = make_float4(__fmul_rn(v.x, w01.x), __fmul_rn(v.y, w01.y),
                                    __fmul_rn(v.z, w23.x), __fmul_rn(v.w, w23.y));
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const float v = hc_warp_sum_lane0(ss[r]);
            if (lane == 0) red[r][warp] = v;
        }
    }
    __syncthreads();
    if (bi == 0 && tid < R)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < GRS_NW; ++i) v += red[tid][i];
        __hip_atomic_store(dots + ((size_t) tid * (M + 1) + M) * H + hb, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    // Phase 1: R dots for proj row ja (stream hb)
    {
        #pragma unroll 1
        for (int r = 0; r < R; ++r)
        {
            #pragma unroll
            for (int k = 0; k < (I8 ? KQ : KC); ++k) grs_opaque(fa[k]);
            const float4* s = sw4[r];
            float a = 0.0f;
            if constexpr (I8)
            {
                #pragma unroll
                for (int k = 0; k < KQ; ++k)
                {
                    const int c16 = lane + 32 * k;
                    const int* q4 = (const int*) &fa[k];
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
            }
            else
            {
                #pragma unroll
                for (int k = 0; k < KC; ++k)
                {
                    const int c8 = lane + 32 * k;
                    const float4 s0 = s[2 * c8], s1 = s[2 * c8 + 1];
                    const half2* f2 = (const half2*) &fa[k];
                    float2 a01 = __half22float2(f2[0]), a23 = __half22float2(f2[1]);
                    float2 b01 = __half22float2(f2[2]), b23 = __half22float2(f2[3]);
                    a = fmaf(s0.x, a01.x, a); a = fmaf(s0.y, a01.y, a);
                    a = fmaf(s0.z, a23.x, a); a = fmaf(s0.w, a23.y, a);
                    a = fmaf(s1.x, b01.x, a); a = fmaf(s1.y, b01.y, a);
                    a = fmaf(s1.z, b23.x, a); a = fmaf(s1.w, b23.y, a);
                }
            }
            a = hc_warp_sum_lane0(a);
            if (has_j && lane == 0) __hip_atomic_store(dots + ((size_t) r * (M + 1) + ja) * H + hb, a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    grs_grid_barrier(bar, nb);

    // Phase 2: rmr and t for every row (fixed order, every block the same); each thread's dots rows are
    // loaded before the rmr wait
    constexpr int KT = (R * LR + GRS_NT - 1) / GRS_NT;
    float4 dv[KT];
    #pragma unroll
    for (int k = 0; k < KT; ++k)
    {
        const int i = min(tid + GRS_NT * k, R * LR - 1), r = i / LR, l = i % LR;
        const float* dp = dots + ((size_t) r * (M + 1) + l) * H;
        dv[k] = make_float4(__hip_atomic_load(dp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT),
                            __hip_atomic_load(dp + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT),
                            __hip_atomic_load(dp + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT),
                            __hip_atomic_load(dp + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[r][h] = rsqrtf(__hip_atomic_load(dots + ((size_t) r * (M + 1) + M) * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    #pragma unroll
    for (int k = 0; k < KT; ++k)
    {
        const int i = tid + GRS_NT * k;
        if (i >= R * LR) break;
        const int r = i / LR, l = i % LR;
        float v = fmaf(rmr_s[r][3], dv[k].w, fmaf(rmr_s[r][2], dv[k].z, fmaf(rmr_s[r][1], dv[k].y, rmr_s[r][0] * dv[k].x)));
        v *= inv_h;
        t_s[r][l] = v * sigmoidf_(v);
    }
    if (POST && b == 0 && tid < R * H)
    {
        const int r = tid / H, hh = tid % H;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r][h], __hip_atomic_load(dr + (LR + hh) * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), v);
        post[r * H + hh] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();
    if (!has_q) return;

    // One warp per column quad, rows in pairs: g[rl][h][e] for the pair (up unpacked once per pair), then a
    // reduce-scatter over the 32 lanes leaves lane l with the full sum of value l = (rl, h, e); every lane finishes
    // its own value (sigmoid, gate, weight), the stream sum runs over lane bits 2-3 (fixed order) and lanes with
    // h = 0 write mixed; each lane writes its updated stream value back
    const int qi = warp;
    const int le = lane & 3, lh = (lane >> 2) & 3, lrl = lane >> 4;
    const float wl = __half2float(w[(size_t) lh * D + 4 * qc + le]);
    #pragma unroll 1
    for (int r0 = 0; r0 < R; r0 += 2)
    {
        const int nr = min(2, R - r0);
        float g[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) g[i] = 0.0f;
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                if constexpr (I8) grs_opaque(u8[it][h]);
                else grs_opaque(u[it][h]);
            }
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            float t0[2], t1[2];
            #pragma unroll
            for (int rl = 0; rl < 2; ++rl)
            {
                const int r = min(r0 + rl, R - 1);
                t0[rl] = t_s[r][2 * lane + 64 * it];
                t1[rl] = t_s[r][2 * lane + 64 * it + 1];
            }
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float ua[4], ub[4];
                if constexpr (I8)
                {
                    const int qa = u8[it][h].x, qb = u8[it][h].y;
                    const int2 usv = usc_s[(qi * H + h) * NIT + it];
                    const half2* us2 = (const half2*) &usv;
                    const float2 s01 = __half22float2(us2[0]), s23 = __half22float2(us2[1]);
                    const float sc[4] = { s01.x, s01.y, s23.x, s23.y };
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        ua[e] = sc[e] * (float) (int8_t) (qa >> (8 * e));
                        ub[e] = sc[e] * (float) (int8_t) (qb >> (8 * e));
                    }
                }
                else
                {
                    const half2* u2 = (const half2*) &u[it][h];
                    float2 a01 = __half22float2(u2[0]), a23 = __half22float2(u2[1]);
                    float2 b01 = __half22float2(u2[2]), b23 = __half22float2(u2[3]);
                    ua[0] = a01.x; ua[1] = a01.y; ua[2] = a23.x; ua[3] = a23.y;
                    ub[0] = b01.x; ub[1] = b01.y; ub[2] = b23.x; ub[3] = b23.y;
                }
                #pragma unroll
                for (int rl = 0; rl < 2; ++rl)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        float& gg = g[(rl * H + h) * 4 + e];
                        gg = fmaf(t1[rl], ub[e], fmaf(t0[rl], ua[e], gg));
                    }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        // reduce-scatter: after offset o, lanes with bit o set keep the upper half of the remaining values
        #define RS_STEP(O, N) \
            { const bool up = (lane & O) != 0; \
              _Pragma("unroll") for (int i = 0; i < N / 2; ++i) { \
                const float keep = up ? g[i + N / 2] : g[i]; const float send = up ? g[i] : g[i + N / 2]; \
                g[i] = keep + ((O) == 16 ? hc_permlanex16(send) : hc_xor16<(O) == 16 ? 1 : (O)>(send)); } }
        RS_STEP(16, 32)
        RS_STEP(8, 16)
        RS_STEP(4, 8)
        RS_STEP(2, 4)
        RS_STEP(1, 2)
        #undef RS_STEP
        // lane l now holds the full g of value l = (lrl, lh, le)
        const int r = r0 + lrl;
        const bool live = lrl < nr;
        const int rc = min(r, R - 1);
        float sv = ((const float*) &xq_s[qi][rc][lh])[le];
        if constexpr (APPLY) sv = fmaf(pp[rc * H + lh], ((const float*) &yq_s[qi][rc])[le], sv);
        float term = sigmoidf_(g[0]) * (rmr_s[rc][lh] * inv_h) * wl * sv;
        term += hc_xor16<4>(term);
        term += hc_xor16<8>(term);
        if (live)
        {
            if constexpr (APPLY) x[((size_t) r * H + lh) * D + 4 * qc + le] = sv;
            if (lh == 0) mixed[(size_t) r * D + 4 * qc + le] = __float2half_rn(term);
        }
    }
}


// Site kernel for 5..16 rows (batched / longer verify forwards; fp16 tables), the row count at run time. The stack of stream hb is built and
// consumed in groups of up to 4 rows (40 KB of LDS); t for all rows stays in LDS (20 KB). Phase 2 reads each lane's
// own stream / y values from global (only this warp writes them back).
template <int H, int D, int LR, bool POST, bool APPLY, typename Y_T>
__global__ __launch_bounds__(GRS_NT) __attribute__((amdgpu_waves_per_eu(8, 8)))
void gr_site16_kernel
(
    float* __restrict__ x, const Y_T* __restrict__ y, const float* __restrict__ pp, const half* __restrict__ fn,
    const half* __restrict__ upt, const half* __restrict__ w, float* __restrict__ dots, float* __restrict__ post,
    half* __restrict__ mixed, unsigned* __restrict__ bar, const float rms_eps, const int R
)
{
    // Drop this CU's L0 / GL1 lines before the first read: the streams are rewritten in place by every site (and the
    // pending y by the sublayer), and a line another kernel left in the non-coherent caches would be stale
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    constexpr int RMAX = 16, G = 4;
    constexpr int M = LR + (POST ? H : 0);
    constexpr int D4 = D / 4;
    constexpr int KC = D / 8 / 32;
    constexpr int NIT = LR / 64;
    constexpr int NBH = GRS_NB / H;
    constexpr int MQ = 8;
    constexpr int K0 = (G * D4 + GRS_NT - 1) / GRS_NT;
    constexpr int KT = (RMAX * LR + GRS_NT - 1) / GRS_NT;

    const int nb = gridDim.x;
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int hb = b % H, bi = b / H;

    __shared__ float4 sw4[G][D4];
    __shared__ float t_s[RMAX][LR];
    __shared__ float red[RMAX][GRS_NW];
    __shared__ float rmr_s[RMAX][H];

    const int j0 = bi * M / NBH, j1 = (bi + 1) * M / NBH;
    const int ja = j0 + warp;
    const bool has_j = ja < j1;
    const int jc = min(ja, M - 1);
    const int q0 = b * D4 / nb, q1 = (b + 1) * D4 / nb;
    const int qc = q0 + warp;
    const bool has_q = warp < MQ && qc < q1;
    const int qcl = min(qc, D4 - 1);

    // Phase 0a: weights first
    int4 fa[KC];
    {
        const int4* ra = (const int4*) (fn + ((size_t) jc * H + hb) * D);
        #pragma unroll
        for (int k = 0; k < KC; ++k) fa[k] = ra[lane + 32 * k];
    }
    int4 u[NIT][H];
    if (warp < MQ)
    {
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                u[it][h] = *(const int4*) (upt + (((size_t) h * D4 + qcl) * LR + 2 * lane + 64 * it) * 4);
    }

    // Phases 0c + 1 per group of up to G rows: stack (all loads first), sums of squares, dots
    #pragma unroll 1
    for (int g0 = 0; g0 < R; g0 += G)
    {
        const int gn = min(G, R - g0);
        float4 xv[K0], yv[K0];
        #pragma unroll
        for (int k = 0; k < K0; ++k)
        {
            const int i = min(tid + GRS_NT * k, gn * D4 - 1), r = g0 + i / D4, c = i % D4;
            xv[k] = ((const float4*) (x + ((size_t) r * H + hb) * D))[c];
            if constexpr (APPLY) yv[k] = grs_load_y4<Y_T>(y + (size_t) r * D, c);
        }
        float ss[G];
        #pragma unroll
        for (int r = 0; r < G; ++r) ss[r] = 0.0f;
        #pragma unroll
        for (int k = 0; k < K0; ++k)
        {
            const int i = tid + GRS_NT * k;
            if (i >= gn * D4) break;
            const int rl = i / D4, c = i % D4;
            float4 v = xv[k];
            if constexpr (APPLY) v = grs_fma4(pp[(g0 + rl) * H + hb], yv[k], v);
            const float sq = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w)));
            #pragma unroll
            for (int rr = 0; rr < G; ++rr) if (rr == rl) ss[rr] += sq;
            const half2* w2 = (const half2*) (w + (size_t) hb * D + 4 * c);
            float2 w01 = __half22float2(w2[0]), w23 = __half22float2(w2[1]);
            sw4[rl][c] = make_float4(__fmul_rn(v.x, w01.x), __fmul_rn(v.y, w01.y),
                                     __fmul_rn(v.z, w23.x), __fmul_rn(v.w, w23.y));
        }
        #pragma unroll
        for (int rl = 0; rl < G; ++rl)
        {
            const float v = hc_warp_sum_lane0(ss[rl]);
            if (lane == 0 && rl < gn) red[g0 + rl][warp] = v;
        }
        __syncthreads();
        #pragma unroll 1
        for (int rl = 0; rl < gn; ++rl)
        {
            #pragma unroll
            for (int k = 0; k < KC; ++k) grs_opaque(fa[k]);
            const float4* s = sw4[rl];
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < KC; ++k)
            {
                const int c8 = lane + 32 * k;
                const float4 s0 = s[2 * c8], s1 = s[2 * c8 + 1];
                const half2* f2 = (const half2*) &fa[k];
                float2 a01 = __half22float2(f2[0]), a23 = __half22float2(f2[1]);
                float2 b01 = __half22float2(f2[2]), b23 = __half22float2(f2[3]);
                a = fmaf(s0.x, a01.x, a); a = fmaf(s0.y, a01.y, a);
                a = fmaf(s0.z, a23.x, a); a = fmaf(s0.w, a23.y, a);
                a = fmaf(s1.x, b01.x, a); a = fmaf(s1.y, b01.y, a);
                a = fmaf(s1.z, b23.x, a); a = fmaf(s1.w, b23.y, a);
            }
            a = hc_warp_sum_lane0(a);
            if (has_j && lane == 0)
                __hip_atomic_store(dots + ((size_t) (g0 + rl) * (M + 1) + ja) * H + hb, a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();                                   // the next group overwrites the stack
    }
    if (bi == 0 && tid < R)
    {
        float v = 0.0f;
        #pragma unroll
        for (int i = 0; i < GRS_NW; ++i) v += red[tid][i];
        __hip_atomic_store(dots + ((size_t) tid * (M + 1) + M) * H + hb, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    grs_grid_barrier(bar, nb);

    // Phase 2: rmr and t for every row
    float4 dv[KT];
    #pragma unroll
    for (int k = 0; k < KT; ++k)
    {
        const int i = tid + GRS_NT * k, r = i / LR, l = i % LR;
        if (i >= R * LR) break;
        const float* dp = dots + ((size_t) r * (M + 1) + l) * H;
        dv[k] = *(const float4*) dp;     // after the barrier's acquire fence (L0 / GL1 invalidated); the stores
                                         // are agent-scope atomics. 16-byte loads: R x 320 x 4 floats per block
    }
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[r][h] = rsqrtf(__hip_atomic_load(dots + ((size_t) r * (M + 1) + M) * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    #pragma unroll
    for (int k = 0; k < KT; ++k)
    {
        const int i = tid + GRS_NT * k;
        if (i >= R * LR) break;
        const int r = i / LR, l = i % LR;
        float v = fmaf(rmr_s[r][3], dv[k].w, fmaf(rmr_s[r][2], dv[k].z, fmaf(rmr_s[r][1], dv[k].y, rmr_s[r][0] * dv[k].x)));
        v *= inv_h;
        t_s[r][l] = v * sigmoidf_(v);
    }
    if (POST && b == 0 && tid < R * H)
    {
        const int r = tid / H, hh = tid % H;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r][h], __hip_atomic_load(dr + (LR + hh) * H + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), v);
        post[r * H + hh] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();
    if (!has_q) return;

    const int le = lane & 3, lh = (lane >> 2) & 3, lrl = lane >> 4;
    const float wl = __half2float(w[(size_t) lh * D + 4 * qc + le]);
    auto lane_vals = [&](int rr, float& xs_, float& ys_)
    {
        xs_ = x[((size_t) rr * H + lh) * D + 4 * qc + le];
        ys_ = 0.0f;
        if constexpr (APPLY)
        {
            if constexpr (std::is_same_v<Y_T, half>) ys_ = __half2float(y[(size_t) rr * D + 4 * qc + le]);
            else ys_ = y[(size_t) rr * D + 4 * qc + le];
        }
    };
    float nx, ny;
    lane_vals(min(lrl, R - 1), nx, ny);
    #pragma unroll 1
    for (int r0 = 0; r0 < R; r0 += 2)
    {
        const int nr = min(2, R - r0);
        const int r = r0 + lrl;
        const bool live = lrl < nr;
        const int rc = min(r, R - 1);
        float sv = nx, yl = ny;
        lane_vals(min(r0 + 2 + lrl, R - 1), nx, ny);      // next pair's, in flight during this pair's math
        float g[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) g[i] = 0.0f;
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
            #pragma unroll
            for (int h = 0; h < H; ++h) grs_opaque(u[it][h]);
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            float t0[2], t1[2];
            #pragma unroll
            for (int rl = 0; rl < 2; ++rl)
            {
                const int rr = min(r0 + rl, R - 1);
                t0[rl] = t_s[rr][2 * lane + 64 * it];
                t1[rl] = t_s[rr][2 * lane + 64 * it + 1];
            }
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const half2* u2 = (const half2*) &u[it][h];
                float2 a01 = __half22float2(u2[0]), a23 = __half22float2(u2[1]);
                float2 b01 = __half22float2(u2[2]), b23 = __half22float2(u2[3]);
                const float ua[4] = { a01.x, a01.y, a23.x, a23.y };
                const float ub[4] = { b01.x, b01.y, b23.x, b23.y };
                #pragma unroll
                for (int rl = 0; rl < 2; ++rl)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        float& gg = g[(rl * H + h) * 4 + e];
                        gg = fmaf(t1[rl], ub[e], fmaf(t0[rl], ua[e], gg));
                    }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        #define RS_STEP(O, N) \
            { const bool up = (lane & O) != 0; \
              _Pragma("unroll") for (int i = 0; i < N / 2; ++i) { \
                const float keep = up ? g[i + N / 2] : g[i]; const float send = up ? g[i] : g[i + N / 2]; \
                g[i] = keep + ((O) == 16 ? hc_permlanex16(send) : hc_xor16<(O) == 16 ? 1 : (O)>(send)); } }
        RS_STEP(16, 32)
        RS_STEP(8, 16)
        RS_STEP(4, 8)
        RS_STEP(2, 4)
        RS_STEP(1, 2)
        #undef RS_STEP
        if constexpr (APPLY) sv = fmaf(pp[rc * H + lh], yl, sv);
        float term = sigmoidf_(g[0]) * (rmr_s[rc][lh] * inv_h) * wl * sv;
        term += hc_xor16<4>(term);
        term += hc_xor16<8>(term);
        if (live)
        {
            if constexpr (APPLY) x[((size_t) r * H + lh) * D + 4 * qc + le] = sv;
            if (lh == 0) mixed[(size_t) r * D + 4 * qc + le] = __float2half_rn(term);
        }
    }
}

// Launch a site kernel on its fixed grid of GRS_NB blocks, which must all be co-resident (the grid barrier):
// false (nothing launched) when the device holds fewer
// Whether GRS_NB blocks of a site kernel are co-resident on this device (the grid barrier needs all of them)
static bool gr_site_coresident(const void* kernel)
{
    int dev; cudaGetDevice(&dev);
    static int caps[16][64] = { { 0 } };
    static int slot_next = 0;
    static const void* slots[64] = { nullptr };
    int slot = -1;
    for (int i = 0; i < slot_next; ++i) if (slots[i] == kernel) slot = i;
    if (slot < 0) { if (slot_next == 64) return false; slot = slot_next++; slots[slot] = kernel; }
    if (!caps[dev][slot])
    {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        // HIP's occupancy query does not model gfx11 WGP mode (it reports WGPs as multiprocessors and sizes
        // LDS and VGPRs against one CU). A WGP holds two of these blocks when each fits one CU: 16 wave32 waves
        // over 2 SIMDs at <= 192 VGPRs (1536 per SIMD lane on gfx1100) and <= 64 KB of LDS. Measured: 96 blocks
        // run co-resident on gfx1100's 48 WGPs (the barrier completes)
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kernel);
        int per = 0;
        if (strncmp(p.gcnArchName, "gfx11", 5) == 0 && fa.numRegs <= 192 && fa.sharedSizeBytes <= 65536) per = 2;
        else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, GRS_NT, 0);
        caps[dev][slot] = per * p.multiProcessorCount >= GRS_NB ? 1 : -1;
    }
    return caps[dev][slot] > 0;
}

// Launch a 1..4-row site kernel on its fixed grid of GRS_NB blocks; false (nothing launched) when the device
// cannot hold them all at once
template <typename K, typename Y_T>
static bool gr_site_launch(K kernel, float* x, const Y_T* y, const float* pp, const void* fn, const half* fn_s,
                           const void* upt, const half* upt_s, const half* w, float* dots, float* post, half* mixed,
                           unsigned* bar, float eps, cudaStream_t stream)
{
    if (!gr_site_coresident((const void*) kernel)) return false;
    kernel<<<GRS_NB, GRS_NT, 0, stream>>>(x, y, pp, fn, fn_s, upt, upt_s, w, dots, post, mixed, bar, eps);
    cuda_check(cudaPeekAtLastError());
    return true;
}

// The 5..16-row site kernel (fp16 tables)
static bool gr_site16_dispatch
(
    int R, float* s_p, const void* py, bool y_half, const float* ppost, const half* fn, const half* upt,
    const half* w_p, float* dots_p, float* post_p, half* mixed_p, unsigned* bar, float eps, cudaStream_t stream
)
{
    const bool apply = py != nullptr;
    bool done = false;
    #define SITE16(POST_, AP, Y_T) \
        { auto kk = gr_site16_kernel<4, 2560, 320, POST_, AP, Y_T>; \
          if (gr_site_coresident((const void*) kk)) { \
              kk<<<GRS_NB, GRS_NT, 0, stream>>>(s_p, (const Y_T*) py, ppost, fn, upt, w_p, dots_p, post_p, mixed_p, bar, eps, R); \
              cuda_check(cudaPeekAtLastError()); done = true; } }
    #define SITE16_P(POST_) if (!apply) SITE16(POST_, false, float) else if (y_half) SITE16(POST_, true, half) else SITE16(POST_, true, float)
    if (post_p) { SITE16_P(true) } else { SITE16_P(false) }
    #undef SITE16_P
    #undef SITE16
    return done;
}

// Site kernel dispatch over rows (1..4), form, apply and y dtype
template <bool I8>
static bool gr_site_dispatch
(
    int R, float* s_p, const void* py, bool y_half, const float* ppost, const void* fn, const half* fn_s,
    const void* upt, const half* upt_s, const half* w_p, float* dots_p, float* post_p, half* mixed_p,
    unsigned* bar, float eps, cudaStream_t stream
)
{
    const bool apply = py != nullptr;
    bool done = false;
    #define SITE(R_, POST_, AP, Y_T) done = gr_site_launch(gr_site_kernel<4, 2560, 320, R_, POST_, AP, Y_T, I8>, \
        s_p, (const Y_T*) py, ppost, fn, fn_s, upt, upt_s, w_p, dots_p, post_p, mixed_p, bar, eps, stream)
    #define SITE_P(R_, POST_) if (!apply) SITE(R_, POST_, false, float); else if (y_half) SITE(R_, POST_, true, half); else SITE(R_, POST_, true, float);
    #define SITE_R(R_) if (post_p) { SITE_P(R_, true) } else { SITE_P(R_, false) }
    switch (R)
    {
        case 1: SITE_R(1) break;
        case 2: SITE_R(2) break;
        case 3: SITE_R(3) break;
        case 4: SITE_R(4) break;
        default: break;
    }
    #undef SITE_R
    #undef SITE_P
    #undef SITE
    return done;
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
    if (!site_off && bar && R >= 1 && R <= 4 && half_out && D == 2560 && LR == 320 &&
        gr_site_dispatch<false>(R, s_p, py, y_half, ppost, fn_p, nullptr, upt_p, nullptr, w_p, dots_p, post_p,
                                (half*) mixed_p, bar, rms_eps, stream))
        return true;
    // 5..16 rows: off unless EXL3_GR_SITE16=1 (TP-ts run-to-run determinism still open, see hyperconnections.py)
    static const bool site16_on = [] { const char* e = getenv("EXL3_GR_SITE16"); return e && e[0] == '1'; }();
    if (!site_off && site16_on && bar && R >= 5 && R <= 16 && half_out && D == 2560 && LR == 320 &&
        gr_site16_dispatch(R, s_p, py, y_half, ppost, fn_p, upt_p, w_p, dots_p, post_p, (half*) mixed_p, bar,
                           rms_eps, stream))
        return true;
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
// ranks per output channel; GatedResidual quantizes them at load under EXL3_GR_I8). R <= 2, the Qwen3.8 shape
// only; false (nothing launched) otherwise, and the caller takes the fp16 path.
static bool gr_site_i8_try
(
    float* s_p, const void* py, bool y_half, const float* ppost,
    const int8_t* fn_q, const half* fn_s, const int8_t* upt_q, const half* upt_s, const half* w_p,
    float* dots_p, float* post_p, half* mixed_p, unsigned* bar,
    const int LR, const int D, const int R, const float rms_eps, cudaStream_t stream
)
{
    // R <= 2 only: at 3-4 rows the int8 unpacking costs more than the bytes it saves (fp16 31.9 vs int8 34.2 us
    // at R = 4 in the micro-benchmark)
    if (R < 1 || R > 2 || D != 2560 || LR != 320 || !bar) return false;
    return gr_site_dispatch<true>(R, s_p, py, y_half, ppost, fn_q, fn_s, upt_q, upt_s, w_p, dots_p, post_p, mixed_p,
                                  bar, rms_eps, stream);
}
