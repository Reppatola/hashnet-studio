// Все CUDA-ядра HashNet4D. Компилируются один раз через NVRTC (см. gpu.rs).
// Везде blockDim.x == 256 (степень двойки — нужно для редукций).
#define BM 64
#define BN 64
#define BK 16

__device__ __forceinline__ float bsum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float bmax(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

// ───────────── SGEMM: C = alpha * op(A) * op(B) + beta * C, строки-мажорный порядок ─────────────
// A хранится как [M,K] (ta=0) или [K,M] (ta=1); B как [K,N] (tb=0) или [N,K] (tb=1).
// Пакетный режим: индекс z = blockIdx.z раскладывается в (z/nh, z%nh);
// смещение = (z/nh)*s?1 + (z%nh)*s?2  — так адресуются головы внимания без gather/scatter.
extern "C" __global__ void sgemm(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int lda, int ldb, int ldc, int ta, int tb,
    int nh, long long sa1, long long sa2, long long sb1, long long sb2, long long sc1, long long sc2,
    float alpha, float beta)
{
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    int z = blockIdx.z;
    long long z1 = z / nh, z2 = z % nh;
    A += z1 * sa1 + z2 * sa2;
    B += z1 * sb1 + z2 * sb2;
    C += z1 * sc1 + z2 * sc2;

    int tid = threadIdx.x;
    int tx = tid & 15, ty = tid >> 4;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            int m, k;
            if (ta) { m = i % BM; k = i / BM; } else { k = i % BK; m = i / BK; }
            int gm = row0 + m, gk = k0 + k;
            float v = 0.f;
            if (gm < M && gk < K) v = ta ? A[(long long)gk * lda + gm] : A[(long long)gm * lda + gk];
            As[k][m] = v;
        }
        for (int i = tid; i < BK * BN; i += 256) {
            int n, k;
            if (tb) { k = i % BK; n = i / BK; } else { n = i % BN; k = i / BN; }
            int gn = col0 + n, gk = k0 + k;
            float v = 0.f;
            if (gn < N && gk < K) v = tb ? B[(long long)gn * ldb + gk] : B[(long long)gk * ldb + gn];
            Bs[k][n] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        int gm = row0 + ty + 16 * i;
        if (gm >= M) continue;
        for (int j = 0; j < 4; j++) {
            int gn = col0 + tx + 16 * j;
            if (gn >= N) continue;
            long long idx = (long long)gm * ldc + gn;
            C[idx] = (beta == 0.f) ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[idx];
        }
    }
}

// ───────────── эмбеддинги ─────────────
extern "C" __global__ void embed_fwd(const float* emb, const float* pos, const unsigned char* tok,
                                     float* x, int t, int d)
{
    int r = blockIdx.x;
    int tk = tok[r], p = r % t;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        x[(long long)r * d + j] = emb[tk * d + j] + pos[p * d + j];
}

extern "C" __global__ void embed_bwd(const float* dx, const unsigned char* tok,
                                     float* g_emb, float* g_pos, int t, int d)
{
    int r = blockIdx.x;
    int tk = tok[r], p = r % t;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = dx[(long long)r * d + j];
        atomicAdd(&g_emb[tk * d + j], v);
        atomicAdd(&g_pos[p * d + j], v);
    }
}

// ───────────── LayerNorm ─────────────
extern "C" __global__ void ln_fwd(const float* x, const float* g, const float* b,
                                  float* out, float* xn, float* stds, int d)
{
    __shared__ float sh[256];
    int r = blockIdx.x;
    const float* row = x + (long long)r * d;
    float s = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s += row[j];
    float mean = bsum(s, sh) / d;
    float q = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) { float c = row[j] - mean; q += c * c; }
    float var = bsum(q, sh) / d;
    float sd = sqrtf(var + 1e-5f);
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float n = (row[j] - mean) / sd;
        xn[(long long)r * d + j] = n;
        out[(long long)r * d + j] = n * g[j] + b[j];
    }
    if (threadIdx.x == 0) stds[r] = sd;
}

// dx = (res) + LN-backward(dout). res может отсутствовать (use_res = 0).
extern "C" __global__ void ln_bwd(const float* dout, const float* xn, const float* stds, const float* g,
                                  const float* res, int use_res, float* dx, int d)
{
    __shared__ float sh[256];
    int r = blockIdx.x;
    long long o = (long long)r * d;
    float s1 = 0.f, s2 = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float dxn = dout[o + j] * g[j];
        s1 += dxn;
        s2 += dxn * xn[o + j];
    }
    float m1 = bsum(s1, sh) / d;
    float m2 = bsum(s2, sh) / d;
    float inv = 1.f / stds[r];
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float dxn = dout[o + j] * g[j];
        float v = (dxn - m1 - xn[o + j] * m2) * inv;
        dx[o + j] = use_res ? v + res[o + j] : v;
    }
}

// dg += Σ_r dout*xn ; db += Σ_r dout   (по столбцам, без атомиков)
extern "C" __global__ void colsum2(const float* dout, const float* xn, float* dg, float* db, int n, int d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;
    float a = 0.f, b = 0.f;
    for (int r = 0; r < n; r++) {
        float o = dout[(long long)r * d + j];
        a += o * xn[(long long)r * d + j];
        b += o;
    }
    dg[j] += a;
    db[j] += b;
}

// ───────────── 8-бит absmax fake-quant активаций по строке ─────────────
extern "C" __global__ void aq(const float* x, float* out, int d)
{
    __shared__ float sh[256];
    long long o = (long long)blockIdx.x * d;
    float mx = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) mx = fmaxf(mx, fabsf(x[o + j]));
    mx = bmax(mx, sh);
    if (mx < 1e-8f) {
        for (int j = threadIdx.x; j < d; j += blockDim.x) out[o + j] = 0.f;
        return;
    }
    float s = 127.f / mx;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float q = roundf(x[o + j] * s);
        q = fminf(127.f, fmaxf(-127.f, q));
        out[o + j] = q / s;
    }
}

extern "C" __global__ void copyk(const float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

extern "C" __global__ void addk(const float* a, const float* b, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void zerok(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.f;
}

// ───────────── GELU ─────────────
#define GC 0.7978845608f
extern "C" __global__ void gelu_fwd(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float u = GC * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.f + tanhf(u));
}

extern "C" __global__ void gelu_bwd(const float* dy, const float* x, float* dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float u = GC * (v + 0.044715f * v * v * v);
    float th = tanhf(u);
    float gr = 0.5f * (1.f + th) + 0.5f * v * (1.f - th * th) * GC * (1.f + 3.f * 0.044715f * v * v);
    dx[i] = dy[i] * gr;
}

// ───────────── softmax с каузальной маской (на месте); строка = (b,h,i), i = row % t ─────────────
extern "C" __global__ void softmax_fwd(float* S, int rows, int t)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int lim = (row % t) + 1;
    float* p = S + (long long)row * t;
    float mx = -1e30f;
    for (int j = 0; j < lim; j++) mx = fmaxf(mx, p[j]);
    float sum = 0.f;
    for (int j = 0; j < lim; j++) { float e = expf(p[j] - mx); p[j] = e; sum += e; }
    float inv = 1.f / sum;
    for (int j = 0; j < lim; j++) p[j] *= inv;
    for (int j = lim; j < t; j++) p[j] = 0.f;
}

// dS = P ∘ (dP − Σ_j P·dP)  (на месте в dP)
extern "C" __global__ void softmax_bwd(const float* P, float* dP, int rows, int t)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* p = P + (long long)row * t;
    float* dp = dP + (long long)row * t;
    float s = 0.f;
    for (int j = 0; j < t; j++) s += p[j] * dp[j];
    for (int j = 0; j < t; j++) dp[j] = p[j] * (dp[j] - s);
}

// ───────────── потеря: одна строка на блок, поток = элемент словаря (VOCAB = 256) ─────────────
extern "C" __global__ void loss_k(const float* logits, const unsigned char* tgt, float* dl,
                                  float* nll, float scale, int write_grad)
{
    __shared__ float sh[256];
    int r = blockIdx.x, v = threadIdx.x;
    float x = logits[(long long)r * 256 + v];
    float mx = bmax(x, sh);
    float e = expf(x - mx);
    float sum = bsum(e, sh);
    float lse = mx + logf(sum);
    int tg = tgt[r];
    if (v == tg) atomicAdd(nll, lse - x);
    if (write_grad) {
        float p = expf(x - lse) * scale;
        if (v == tg) p -= scale;
        dl[(long long)r * 256 + v] = p;
    }
}

// ───────────── тернарная квантизация absmean (BitNet b1.58) ─────────────
extern "C" __global__ void absum(const float* w, int len, float* acc, int idx)
{
    __shared__ float sh[256];
    float s = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) s += fabsf(w[i]);
    float r = bsum(s, sh);
    if (threadIdx.x == 0) atomicAdd(&acc[idx], r);
}

extern "C" __global__ void ternarize(const float* w, float* eff, int len, const float* acc, int idx, float* zcnt)
{
    __shared__ float sh[256];
    float g = fmaxf(acc[idx] / (float)len, 1e-8f);
    float inv = 1.f / g;
    float z = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        float q = fminf(1.f, fmaxf(-1.f, roundf(w[i] * inv)));
        if (q == 0.f) z += 1.f;
        eff[i] = q * g;
    }
    float r = bsum(z, sh);
    if (threadIdx.x == 0) atomicAdd(zcnt, r);
}

// ───────────── AdamW ─────────────
extern "C" __global__ void sumsq(const float* g, int len, float* acc)
{
    __shared__ float sh[256];
    float s = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) s += g[i] * g[i];
    float r = bsum(s, sh);
    if (threadIdx.x == 0) atomicAdd(acc, r);
}

extern "C" __global__ void adam(float* w, const float* g, float* m, float* v, int len,
                                float lr, float b1, float b2, float eps, float wd,
                                float coef, float bc1, float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float gj = g[i] * coef;
    float mj = b1 * m[i] + (1.f - b1) * gj;
    float vj = b2 * v[i] + (1.f - b2) * gj * gj;
    m[i] = mj;
    v[i] = vj;
    float mh = mj / bc1, vh = vj / bc2;
    w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
}
