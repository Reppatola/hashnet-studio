// Ядра инференса (один токен за раз, KV-кэш, тернарные веса int8 × активации int8).
// blockDim.x == 256 везде.
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

extern "C" __global__ void embed_tok(const float* emb, const float* pos_emb, float* x, int tok, int pos, int d)
{
    for (int j = threadIdx.x; j < d; j += blockDim.x) x[j] = emb[tok * d + j] + pos_emb[pos * d + j];
}

// добавить строки хэш-таблиц n-грамм (индексы корзин считает хост той же ng_hash)
extern "C" __global__ void add_rows2(const float* h3, const float* h5, float* x, int i3, int i5, int d)
{
    for (int j = threadIdx.x; j < d; j += blockDim.x) x[j] += h3[(long long)i3 * d + j] + h5[(long long)i5 * d + j];
}

// LayerNorm одного вектора (1 блок)
extern "C" __global__ void ln_f(const float* x, const float* g, const float* b, float* out, int d)
{
    __shared__ float sh[256];
    float s = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s += x[j];
    float mean = bsum(s, sh) / d;
    float q = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) { float c = x[j] - mean; q += c * c; }
    float sd = sqrtf(bsum(q, sh) / d + 1e-5f);
    for (int j = threadIdx.x; j < d; j += blockDim.x) out[j] = (x[j] - mean) / sd * g[j] + b[j];
}

// 8-бит absmax-квантизация: a ≈ q * sc[0]  (тот же результат, что fake-quant при обучении)
extern "C" __global__ void q8(const float* x, signed char* q, float* sc, int n)
{
    __shared__ float sh[256];
    float mx = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, fabsf(x[j]));
    mx = bmax(mx, sh);
    if (mx < 1e-8f) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) q[j] = 0;
        if (threadIdx.x == 0) sc[0] = 0.f;
        return;
    }
    float s = 127.f / mx;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = roundf(x[j] * s);
        q[j] = (signed char)fminf(127.f, fmaxf(-127.f, v));
    }
    if (threadIdx.x == 0) sc[0] = 1.f / s;
}

#define GC 0.7978845608f
extern "C" __global__ void gelu_f(const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = 0.5f * v * (1.f + tanhf(GC * (v + 0.044715f * v * v * v)));
}

// out[out_off+j] (+)= gamma * sc * Σ_i W[j,i] * qa[i];  W — int8 {-1,0,+1}, хранится [out,in]. Один варп на выход.
// Требуется in_n % 4 == 0.
extern "C" __global__ void tmatvec(const signed char* W, const signed char* qa, const float* sc, float gamma,
                                   float* out, int in_n, int out_n, int out_off, int add)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    bool valid = warp < out_n;
    int acc = 0;
    if (valid) {
        const int* w4 = (const int*)(W + (long long)warp * in_n);
        const int* a4 = (const int*)qa;
        for (int i = lane; i < (in_n >> 2); i += 32) {
            int w = w4[i], a = a4[i];
            acc += (int)(signed char)(w)       * (int)(signed char)(a);
            acc += (int)(signed char)(w >> 8)  * (int)(signed char)(a >> 8);
            acc += (int)(signed char)(w >> 16) * (int)(signed char)(a >> 16);
            acc += (int)(signed char)(w >> 24) * (int)(signed char)(a >> 24);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (valid && lane == 0) {
        float v = (float)acc * sc[0] * gamma;
        out[out_off + warp] = add ? out[out_off + warp] + v : v;
    }
}

// f32-матвек для выходной проекции: out[j] = Σ_i x[i]*W[i,j], W [in,out]
extern "C" __global__ void matvec_f32(const float* x, const float* W, float* out, int in_n, int out_n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_n) return;
    float a = 0.f;
    for (int i = 0; i < in_n; i++) a += x[i] * W[(long long)i * out_n + j];
    out[j] = a;
}

// внимание одного запроса по кэшу; блок = голова; динамическая shared = (pos+1)*4 байт
extern "C" __global__ void attn_dec(const float* q, const float* Kc, const float* Vc, float* ctx,
                                    int pos, int d, int dh, float scale)
{
    extern __shared__ float s[];
    __shared__ float sh[256];
    int h = blockIdx.x, T = pos + 1;
    const float* qh = q + h * dh;
    float mx = -1e30f;
    for (int p = threadIdx.x; p < T; p += blockDim.x) {
        const float* kp = Kc + (long long)p * d + h * dh;
        float a = 0.f;
        for (int c = 0; c < dh; c++) a += qh[c] * kp[c];
        a *= scale;
        s[p] = a;
        mx = fmaxf(mx, a);
    }
    mx = bmax(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < T; p += blockDim.x) { float e = expf(s[p] - mx); s[p] = e; sum += e; }
    sum = bsum(sum, sh);
    float inv = 1.f / sum;
    for (int c = threadIdx.x; c < dh; c += blockDim.x) {
        float a = 0.f;
        for (int p = 0; p < T; p++) a += s[p] * Vc[(long long)p * d + h * dh + c];
        ctx[h * dh + c] = a * inv;
    }
}
