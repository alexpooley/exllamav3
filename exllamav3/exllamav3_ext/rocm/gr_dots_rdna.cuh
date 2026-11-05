#pragma once

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

// The pipelined pair for the shapes it is instantiated for; false (nothing launched) otherwise. y_half: the
// pending apply's y dtype. RB: fn rows per dots block (EXL3_GR2_RB, default 4).
static bool gr_mix_rdna_try
(
    float* s_p, const float* ws_p, const half* fn_p, const half* upt_p, const half* w_p,
    const void* py, bool y_half, const float* ppost,
    float* dots_p, float* post_p, void* mixed_p, bool half_out,
    const int M, const int LR, const int D, const int R, const float rms_eps, cudaStream_t stream
)
{
    static const int env_off = [] { const char* e = getenv("EXL3_GR2"); return e && e[0] == '0'; }();
    static const int RBc = [] { const char* e = getenv("EXL3_GR2_RB"); return e ? atoi(e) : 4; }();
    if (env_off) return false;
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
