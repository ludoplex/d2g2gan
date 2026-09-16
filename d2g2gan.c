/* d2g2gan.c — dual-discriminator / dual-generator adversarial network in plain C (C11)
 *
 * Portable build targets (same source):
 *   native : gcc -O2 -std=c11 d2g2gan.c -lopenblas -lm            (OpenBLAS exports sgemm_/ssyev_/sgeqrf_/sorgqr_)
 *   cosmo  : cosmocc -O2 -std=c11 -DCLAPACK_F2C_WRAP d2g2gan.c $C/lapack_COSMO.a $C/blas_COSMO.a $C/F2CLIBS/libf2c.a -lm   (netlib CLAPACK 3.2.1, pure C)
 *
 * Model (MGAN-style, K=2 generators; two discriminators with different jobs):
 *   G_k : z(B,Z) -> lrelu(z W1_k + b1_k) -> x(B,X)                  k in {0,1}
 *   D_r : x(B,X) -> lrelu(x V1 + c1)     -> logit  (real=1 vs fake=0, BCE)
 *   D_c : x(B,X) -> lrelu(x U1 + d1)     -> logit  (from G_1=1 vs G_0=0, BCE)  "which generator" classifier
 *   L_Dr = BCE(D_r(real),1) + 1/2 sum_k BCE(D_r(G_k(z)),0)
 *   L_Dc = 1/2 sum_k BCE(D_c(G_k(z)),k)
 *   L_Gk = BCE(D_r(G_k(z)),1) + lambda * BCE(D_c(G_k(z)),k)           (non-saturating; D params frozen in G step)
 *   Target: 8-Gaussian ring, radius 2, sigma 0.05 — the standard mode-collapse toy.
 *
 * Numerics: every matmul is Fortran-ABI sgemm_ via the row-major transposition identity
 *   C = op(A) op(B)  (row-major)  <=>  C^T = op(B)^T op(A)^T  (column-major)
 * LAPACK: sgeqrf_+sorgqr_ (orthogonal init), ssyev_ (symmetric sqrt for the Frechet/2-Wasserstein metric).
 *
 * Discipline (operator directive 2026-09-14):
 *   - no explicit branches in the AUTHORED elementwise kernels: arithmetic masks, fmaxf/fabsf; no `if` in inner loops
 *     (libm calls — tanhf/expf/log1pf/atan2f — branch internally; the claim is about this source, not the emitted code)
 *   - dynamic arena allocators: chunked bump allocator with mark/reset; persistent arena (params, Adam state),
 *     frame arena reset per training step (activations, grads)
 *   - generous asserts: ASSERT is always on (not tied to NDEBUG); every kernel checks shapes/strides/alignment/
 *     arena invariants at entry; finite-checks run at a fixed cadence outside inner loops.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ----------------------------------------------------------------------------------------------- asserts */
#define ASSERT(c) do { if (!(c)) { fprintf(stderr, "ASSERT FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); fflush(stderr); abort(); } } while (0)
#define ASSERT_MSG(c, ...) do { if (!(c)) { fprintf(stderr, "ASSERT FAILED %s:%d: %s — ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); abort(); } } while (0)
#define ASSERT_ALIGNED(p, a) ASSERT((((uintptr_t)(p)) & ((a) - 1)) == 0)

/* ----------------------------------------------------------------------------------------------- Fortran BLAS/LAPACK ABI */
#ifdef CLAPACK_F2C_WRAP          /* netlib CLAPACK renames BLAS entry points (INCLUDE/blaswrap.h:127: #define sgemm_ f2c_sgemm) */
#define sgemm_ f2c_sgemm
typedef long fint;               /* f2c.h: typedef long int integer;  (8 bytes on LP64 — caught by ASSERT(info==0) on first run) */
#else
typedef int fint;                /* OpenBLAS / LAPACKE default: 32-bit Fortran INTEGER */
#endif
extern void sgemm_(const char *ta, const char *tb, const fint *m, const fint *n, const fint *k, const float *alpha,
                   const float *a, const fint *lda, const float *b, const fint *ldb, const float *beta, float *c, const fint *ldc);
extern void ssyev_(const char *jobz, const char *uplo, const fint *n, float *a, const fint *lda, float *w,
                   float *work, const fint *lwork, fint *info);
extern void sgeqrf_(const fint *m, const fint *n, float *a, const fint *lda, float *tau, float *work, const fint *lwork, fint *info);
extern void sorgqr_(const fint *m, const fint *n, const fint *k, float *a, const fint *lda, const float *tau,
                    float *work, const fint *lwork, fint *info);

/* row-major GEMM: C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C ; op = 'N' or 'T' */
static void gemm(char transA, char transB, int M, int N, int K, float alpha,
                 const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    ASSERT(M > 0 && N > 0 && K > 0);
    ASSERT(transA == 'N' || transA == 'T');
    ASSERT(transB == 'N' || transB == 'T');
    ASSERT(lda >= (transA == 'N' ? K : M));
    ASSERT(ldb >= (transB == 'N' ? N : K));
    ASSERT(ldc >= N);
    ASSERT(A && B && C);
    /* row-major C = op(A) op(B)  ==  col-major C^T = op(B)^T op(A)^T ; a row-major [r,c] buffer IS a col-major [c,r] buffer */
    fint fM = M, fN = N, fK = K, flda = lda, fldb = ldb, fldc = ldc;
    sgemm_(&transB, &transA, &fN, &fM, &fK, &alpha, B, &fldb, A, &flda, &beta, C, &fldc);
}

/* ----------------------------------------------------------------------------------------------- dynamic arena */
#define ARENA_ALIGN 64u
typedef struct ArenaChunk { struct ArenaChunk *next; size_t cap; size_t used; unsigned char *base; } ArenaChunk;
typedef struct Arena { ArenaChunk *head; size_t chunk_bytes; size_t total_reserved; size_t total_used; size_t n_chunks; size_t n_allocs; } Arena;
typedef struct ArenaMark { ArenaChunk *chunk; size_t used; size_t n_chunks; } ArenaMark;

static void *xaligned_alloc(size_t align, size_t bytes)
{
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, align);
#else
    if (posix_memalign(&p, align, bytes) != 0) p = NULL;
#endif
    ASSERT_MSG(p != NULL, "OOM requesting %zu bytes", bytes);
    return p;
}
static ArenaChunk *arena_new_chunk(Arena *a, size_t min_bytes)
{
    size_t cap = a->chunk_bytes;
    while (cap < min_bytes) cap *= 2;                      /* dynamic growth: chunk sizes double as needed */
    ArenaChunk *c = (ArenaChunk *)malloc(sizeof *c);
    ASSERT(c);
    c->base = (unsigned char *)xaligned_alloc(ARENA_ALIGN, cap);
    c->cap = cap; c->used = 0; c->next = a->head;
    a->head = c; a->total_reserved += cap; a->n_chunks++;
    return c;
}
static void arena_init(Arena *a, size_t chunk_bytes)
{
    ASSERT(chunk_bytes >= 4096 && (chunk_bytes & (chunk_bytes - 1)) == 0);
    memset(a, 0, sizeof *a); a->chunk_bytes = chunk_bytes; arena_new_chunk(a, chunk_bytes);
}
static void *arena_alloc(Arena *a, size_t bytes)
{
    ASSERT(a && a->head);
    ASSERT(bytes > 0 && bytes < ((size_t)1 << 40));
    size_t need = (bytes + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1);
    ArenaChunk *c = a->head;
    if (c->cap - c->used < need) c = arena_new_chunk(a, need);
    ASSERT(c->cap - c->used >= need);
    void *p = c->base + c->used;
    c->used += need; a->total_used += need; a->n_allocs++;
    ASSERT_ALIGNED(p, ARENA_ALIGN);
    ASSERT(c->used <= c->cap);
    return p;
}
static float *arena_floats(Arena *a, size_t n) { float *p = (float *)arena_alloc(a, n * sizeof(float)); return p; }
static float *arena_zeros(Arena *a, size_t n) { float *p = arena_floats(a, n); memset(p, 0, n * sizeof(float)); return p; }
static ArenaMark arena_mark(const Arena *a) { ArenaMark m = { a->head, a->head->used, a->n_chunks }; return m; }
static void arena_reset_to(Arena *a, ArenaMark m)
{
    /* release chunks allocated after the mark, then rewind the marked chunk */
    while (a->head != m.chunk) {
        ArenaChunk *c = a->head; ASSERT(c && c->next);
        a->head = c->next; a->total_reserved -= c->cap; a->total_used -= c->used; a->n_chunks--;
#if defined(_WIN32)
        _aligned_free(c->base);
#else
        free(c->base);
#endif
        free(c);
    }
    ASSERT(a->head == m.chunk && a->n_chunks == m.n_chunks);
    ASSERT(a->head->used >= m.used);
    a->total_used -= (a->head->used - m.used);
    a->head->used = m.used;
}

/* ----------------------------------------------------------------------------------------------- RNG (branchless) */
typedef struct { uint64_t s; } Rng;
static inline uint64_t rng_u64(Rng *r) { uint64_t x = r->s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; r->s = x; return x * 0x2545F4914F6CDD1DULL; }
static inline float rng_uniform(Rng *r) { return (float)((rng_u64(r) >> 40) + 1) * (1.0f / 16777217.0f); } /* (0,1) */
static inline void rng_gauss2(Rng *r, float *g0, float *g1)                                                /* Box–Muller, no branch */
{
    float u1 = rng_uniform(r), u2 = rng_uniform(r);
    float rad = sqrtf(-2.0f * logf(u1)), th = 6.283185307179586f * u2;
    *g0 = rad * cosf(th); *g1 = rad * sinf(th);
}
static void fill_gauss(Rng *r, float *p, size_t n, float scale)
{
    ASSERT(p && n > 0);
    size_t i = 0;
    for (; i + 1 < n; i += 2) { float a, b; rng_gauss2(r, &a, &b); p[i] = a * scale; p[i + 1] = b * scale; }
    for (; i < n; i++) { float a, b; rng_gauss2(r, &a, &b); p[i] = a * scale; }
}

/* ----------------------------------------------------------------------------------------------- branchless kernels */
#define LRELU_ALPHA 0.2f
static inline float lrelu_f(float x)   { float m = (float)(x > 0.0f); return x * (m + LRELU_ALPHA * (1.0f - m)); }
static inline float lrelu_df(float x)  { float m = (float)(x > 0.0f); return m + LRELU_ALPHA * (1.0f - m); }
static inline float sigmoid_f(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }                   /* stable, branchless */
static inline float softplus_f(float x){ return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))); }       /* log(1+e^x), branchless */
/* BCE with logits: loss = softplus(l) - t*l ; dloss/dl = sigmoid(l) - t */

static void k_lrelu_fwd(const float *restrict pre, float *restrict out, size_t n)
{ ASSERT(pre && out && n); for (size_t i = 0; i < n; i++) out[i] = lrelu_f(pre[i]); }
static void k_lrelu_bwd(const float *restrict pre, float *restrict dout_inplace, size_t n)
{ ASSERT(pre && dout_inplace && n); for (size_t i = 0; i < n; i++) dout_inplace[i] *= lrelu_df(pre[i]); }
static void k_add_bias(float *restrict y, const float *restrict b, int rows, int cols)
{ ASSERT(y && b && rows > 0 && cols > 0); for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) y[(size_t)r * cols + c] += b[c]; }
static void k_bias_grad(const float *restrict dy, float *restrict db, int rows, int cols)
{ ASSERT(dy && db && rows > 0 && cols > 0); for (int c = 0; c < cols; c++) db[c] = 0.0f;
  for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) db[c] += dy[(size_t)r * cols + c]; }
/* BCE-with-logits on a (B,1) logit column with scalar target t; returns mean loss, writes dlogit = (sig - t)/B */
static float k_bce_logits(const float *restrict logit, float *restrict dlogit, int B, float t, float grad_scale)
{
    ASSERT(logit && dlogit && B > 0);
    ASSERT(t == 0.0f || t == 1.0f);
    float invB = 1.0f / (float)B, loss = 0.0f;
    for (int i = 0; i < B; i++) {
        float l = logit[i];
        loss += softplus_f(l) - t * l;
        dlogit[i] = (sigmoid_f(l) - t) * invB * grad_scale;
    }
    return loss * invB;
}
static int k_all_finite(const float *p, size_t n)                       /* branchless accumulate, one branch at return */
{ ASSERT(p && n); uint32_t bad = 0; for (size_t i = 0; i < n; i++) bad |= (uint32_t)!isfinite(p[i]); return bad == 0; }
static void vadd(float *restrict dst, const float *restrict src, size_t n) { ASSERT(dst && src && n); for (size_t i = 0; i < n; i++) dst[i] += src[i]; }

/* ----------------------------------------------------------------------------------------------- Adam (branchless) */
typedef struct { float *p, *g, *m, *v; size_t n; } Param;
typedef struct { float lr, b1, b2, eps; float b1t, b2t; long t; } Adam;
static void adam_step(Adam *o, Param *P, int np)
{
    ASSERT(o && P && np > 0);
    o->t++; o->b1t *= o->b1; o->b2t *= o->b2;
    float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
    for (int k = 0; k < np; k++) {
        float *restrict p = P[k].p, *restrict g = P[k].g, *restrict m = P[k].m, *restrict v = P[k].v; size_t n = P[k].n;
        ASSERT(p && g && m && v && n);
        for (size_t i = 0; i < n; i++) {
            float gi = g[i];
            m[i] = o->b1 * m[i] + (1.0f - o->b1) * gi;
            v[i] = o->b2 * v[i] + (1.0f - o->b2) * gi * gi;
            p[i] -= o->lr * (m[i] * c1) / (sqrtf(v[i] * c2) + o->eps);
        }
    }
}

/* ----------------------------------------------------------------------------------------------- 2-layer MLP */
typedef struct { int in, hid, out; Param W1, b1, W2, b2; } Mlp;      /* y = lrelu(x W1 + b1) W2 + b2 */
typedef struct { float *pre, *h, *y; float *dpre, *dx; int B; } Act; /* per-step activations (frame arena) */

static Param param_new(Arena *a, size_t n) { Param P; P.n = n; P.p = arena_zeros(a, n); P.g = arena_zeros(a, n); P.m = arena_zeros(a, n); P.v = arena_zeros(a, n); return P; }

/* orthogonal init of a row-major (rows,cols) matrix via LAPACK QR: Q from sgeqrf/sorgqr on a Gaussian matrix */
static void orth_init(Arena *frame, Rng *r, float *W, int rows, int cols, float gain)
{
    ASSERT(W && rows > 0 && cols > 0);
    ArenaMark mk = arena_mark(frame);
    fint m = rows > cols ? rows : cols, n = rows < cols ? rows : cols;  /* tall column-major m x n */
    float *A = arena_floats(frame, (size_t)m * n); fill_gauss(r, A, (size_t)m * n, 1.0f);
    float *tau = arena_floats(frame, (size_t)n);
    fint lwork = -1, info = 0; float wq = 0.0f, wq2 = 0.0f;
    sgeqrf_(&m, &n, A, &m, tau, &wq, &lwork, &info); ASSERT(info == 0);
    sorgqr_(&m, &n, &n, A, &m, tau, &wq2, &lwork, &info); ASSERT(info == 0);
    lwork = (fint)fmaxf(wq, wq2); ASSERT(lwork >= n);
    float *work = arena_floats(frame, (size_t)lwork);
    sgeqrf_(&m, &n, A, &m, tau, work, &lwork, &info); ASSERT_MSG(info == 0, "sgeqrf info=%ld", (long)info);
    sorgqr_(&m, &n, &n, A, &m, tau, work, &lwork, &info); ASSERT_MSG(info == 0, "sorgqr info=%ld", (long)info);
    /* A is col-major (m x n), orthonormal columns, A(i,j) = A[j*m + i].  Cold path: one branch per call, not per element. */
    if (rows >= cols) { for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) W[(size_t)i * cols + j] = gain * A[(size_t)j * m + i]; }
    else              { for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) W[(size_t)i * cols + j] = gain * A[(size_t)i * m + j]; } /* W = A^T */
    ASSERT(k_all_finite(W, (size_t)rows * cols));
    arena_reset_to(frame, mk);
}
static Mlp mlp_new(Arena *persist, Arena *frame, Rng *r, int in, int hid, int out, float gain)
{
    ASSERT(in > 0 && hid > 0 && out > 0);
    Mlp M; M.in = in; M.hid = hid; M.out = out;
    M.W1 = param_new(persist, (size_t)in * hid); M.b1 = param_new(persist, (size_t)hid);
    M.W2 = param_new(persist, (size_t)hid * out); M.b2 = param_new(persist, (size_t)out);
    orth_init(frame, r, M.W1.p, in, hid, gain); orth_init(frame, r, M.W2.p, hid, out, gain);
    return M;
}
static Act mlp_fwd(Arena *frame, const Mlp *M, const float *x, int B)
{
    ASSERT(M && x && B > 0);
    Act A; A.B = B;
    A.pre = arena_floats(frame, (size_t)B * M->hid); A.h = arena_floats(frame, (size_t)B * M->hid);
    A.y = arena_floats(frame, (size_t)B * M->out); A.dpre = NULL; A.dx = NULL;
    gemm('N', 'N', B, M->hid, M->in, 1.0f, x, M->in, M->W1.p, M->hid, 0.0f, A.pre, M->hid);
    k_add_bias(A.pre, M->b1.p, B, M->hid);
    k_lrelu_fwd(A.pre, A.h, (size_t)B * M->hid);
    gemm('N', 'N', B, M->out, M->hid, 1.0f, A.h, M->hid, M->W2.p, M->out, 0.0f, A.y, M->out);
    k_add_bias(A.y, M->b2.p, B, M->out);
    return A;
}
/* backward: given dy (B,out). accumulate_params: write param grads (1) or leave them (0). Always produces A->dx (B,in). */
static void mlp_bwd(Arena *frame, Mlp *M, const float *x, Act *A, const float *dy, int accumulate_params)
{
    ASSERT(M && x && A && dy && A->B > 0);
    int B = A->B;
    A->dpre = arena_floats(frame, (size_t)B * M->hid); A->dx = arena_floats(frame, (size_t)B * M->in);
    if (accumulate_params) {                                                   /* cold path: per-step decision, not per-element */
        gemm('T', 'N', M->hid, M->out, B, 1.0f, A->h, M->hid, dy, M->out, 0.0f, M->W2.g, M->out);
        k_bias_grad(dy, M->b2.g, B, M->out);
    }
    gemm('N', 'T', B, M->hid, M->out, 1.0f, dy, M->out, M->W2.p, M->out, 0.0f, A->dpre, M->hid);
    k_lrelu_bwd(A->pre, A->dpre, (size_t)B * M->hid);
    if (accumulate_params) {
        gemm('T', 'N', M->in, M->hid, B, 1.0f, x, M->in, A->dpre, M->hid, 0.0f, M->W1.g, M->hid);
        k_bias_grad(A->dpre, M->b1.g, B, M->hid);
    }
    gemm('N', 'T', B, M->in, M->hid, 1.0f, A->dpre, M->hid, M->W1.p, M->hid, 0.0f, A->dx, M->in);
}
static void mlp_params(Mlp *M, Param out[4]) { out[0] = M->W1; out[1] = M->b1; out[2] = M->W2; out[3] = M->b2; }

/* ----------------------------------------------------------------------------------------------- data: 8-Gaussian ring */
#define N_MODES 8
#define RING_R 2.0f
#define RING_SIGMA 0.05f
static void sample_ring(Rng *r, float *x, int B)                        /* branchless mode selection */
{
    ASSERT(x && B > 0);
    for (int i = 0; i < B; i++) {
        unsigned k = (unsigned)(rng_u64(r) >> 61);                            /* 0..7 */
        float th = (float)k * (6.283185307179586f / N_MODES);
        float g0, g1; rng_gauss2(r, &g0, &g1);
        x[2 * i] = RING_R * cosf(th) + RING_SIGMA * g0;
        x[2 * i + 1] = RING_R * sinf(th) + RING_SIGMA * g1;
    }
}

/* ----------------------------------------------------------------------------------------------- metrics (LAPACK) */
/* mean/cov of (n,2) samples */
static void moments2(const float *x, int n, float mu[2], float S[4])
{
    ASSERT(x && n > 1);
    double m0 = 0, m1 = 0; for (int i = 0; i < n; i++) { m0 += x[2 * i]; m1 += x[2 * i + 1]; }
    mu[0] = (float)(m0 / n); mu[1] = (float)(m1 / n);
    double s00 = 0, s01 = 0, s11 = 0;
    for (int i = 0; i < n; i++) { double a = x[2 * i] - mu[0], b = x[2 * i + 1] - mu[1]; s00 += a * a; s01 += a * b; s11 += b * b; }
    S[0] = (float)(s00 / (n - 1)); S[1] = S[2] = (float)(s01 / (n - 1)); S[3] = (float)(s11 / (n - 1));
}
/* symmetric sqrt of 2x2 SPD via ssyev; A col-major in/out */
static void sym_sqrt2(float A[4], float R[4])
{
    fint n = 2, lda = 2, lwork = 64, info = 0; float w[2], work[64], V[4];
    memcpy(V, A, sizeof V);
    ssyev_("V", "U", &n, V, &lda, w, work, &lwork, &info); ASSERT_MSG(info == 0, "ssyev info=%ld", (long)info);
    w[0] = sqrtf(fmaxf(w[0], 0.0f)); w[1] = sqrtf(fmaxf(w[1], 0.0f));
    /* R = V diag(sqrt w) V^T */
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        float s = 0; for (int k = 0; k < 2; k++) s += V[k * 2 + i] * w[k] * V[k * 2 + j]; R[j * 2 + i] = s;
    }
}
/* Frechet distance between Gaussians fit to two sample sets: |mu1-mu2|^2 + tr(S1 + S2 - 2 sqrt(sqrt(S1) S2 sqrt(S1))) */
static float frechet2(const float *a, int na, const float *b, int nb)
{
    float mua[2], Sa[4], mub[2], Sb[4], Ra[4], M[4], T[4], Rm[4];
    moments2(a, na, mua, Sa); moments2(b, nb, mub, Sb);
    sym_sqrt2(Sa, Ra);
    /* M = Ra Sb Ra (all symmetric 2x2, col-major == row-major for symmetric) */
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) { float s = 0; for (int k = 0; k < 2; k++) s += Ra[i * 2 + k] * Sb[k * 2 + j]; T[i * 2 + j] = s; }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) { float s = 0; for (int k = 0; k < 2; k++) s += T[i * 2 + k] * Ra[k * 2 + j]; M[i * 2 + j] = s; }
    M[1] = M[2] = 0.5f * (M[1] + M[2]);                                        /* symmetrize fp noise */
    sym_sqrt2(M, Rm);
    float d0 = mua[0] - mub[0], d1 = mua[1] - mub[1];
    float fd = d0 * d0 + d1 * d1 + (Sa[0] + Sa[3]) + (Sb[0] + Sb[3]) - 2.0f * (Rm[0] + Rm[3]);
    ASSERT(isfinite(fd));
    return fd;
}
/* mode coverage: count modes that receive >= 1% of samples within 3 sigma of the center ("high quality") */
static void coverage(const float *x, int n, int *modes_covered, float *hq_frac)
{
    ASSERT(x && n > 0 && modes_covered && hq_frac);
    int cnt[N_MODES] = { 0 }; int hq = 0;
    for (int i = 0; i < n; i++) {
        float px = x[2 * i], py = x[2 * i + 1];
        float ang = atan2f(py, px); ang += 6.283185307179586f * (float)(ang < 0.0f);           /* branchless wrap */
        int k = (int)(ang * (N_MODES / 6.283185307179586f) + 0.5f) & (N_MODES - 1);
        float cx = RING_R * cosf((float)k * (6.283185307179586f / N_MODES)), cy = RING_R * sinf((float)k * (6.283185307179586f / N_MODES));
        float dx = px - cx, dy = py - cy;
        int good = (int)((dx * dx + dy * dy) <= (9.0f * RING_SIGMA * RING_SIGMA));
        cnt[k] += good; hq += good;
    }
    int cov = 0; for (int k = 0; k < N_MODES; k++) cov += (int)(cnt[k] * 100 >= n);
    *modes_covered = cov; *hq_frac = (float)hq / (float)n;
}

/* ----------------------------------------------------------------------------------------------- training */
typedef struct { int B, Z, H, X, steps, eval_every; float lr, lambda; uint64_t seed; } Cfg;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

int main(int argc, char **argv)
{
    Cfg c = { 256, 8, 64, 2, 6000, 500, 2e-4f, 0.5f, 0x9E3779B97F4A7C15ULL };
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--steps")) c.steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--batch")) c.B = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--hidden")) c.H = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lr")) c.lr = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--lambda")) { char *e; c.lambda = strtof(argv[i + 1], &e); if (*e) { fprintf(stderr, "--lambda: not a number: %s\n", argv[i + 1]); return 2; } }
        else if (!strcmp(argv[i], "--seed")) c.seed = strtoull(argv[i + 1], NULL, 0);
        else if (!strcmp(argv[i], "--eval-every")) c.eval_every = atoi(argv[i + 1]);
        else { fprintf(stderr, "usage: %s [--steps N] [--batch B] [--hidden H] [--lr f] [--lambda f] [--seed n] [--eval-every N]\n", argv[0]); return 2; }
    }
    if ((argc - 1) & 1) { fprintf(stderr, "%s: flag %s has no value\n", argv[0], argv[argc - 1]); return 2; }
    ASSERT(c.B >= 8 && c.Z > 0 && c.H > 0 && c.X == 2 && c.steps > 0 && c.eval_every > 0 && c.lr > 0 && c.lambda >= 0);

    Arena persist, frame; arena_init(&persist, 1u << 20); arena_init(&frame, 1u << 20);
    Rng rng = { c.seed | 1ULL };

    Mlp G[2], Dr, Dc;
    G[0] = mlp_new(&persist, &frame, &rng, c.Z, c.H, c.X, 1.0f);
    G[1] = mlp_new(&persist, &frame, &rng, c.Z, c.H, c.X, 1.0f);
    Dr = mlp_new(&persist, &frame, &rng, c.X, c.H, 1, 1.0f);
    Dc = mlp_new(&persist, &frame, &rng, c.X, c.H, 1, 1.0f);
    Adam optG[2], optDr, optDc; Adam a0 = { c.lr, 0.5f, 0.999f, 1e-8f, 1.0f, 1.0f, 0 };
    optG[0] = optG[1] = optDr = optDc = a0;
    Param PG[2][4], PDr[4], PDc[4]; mlp_params(&G[0], PG[0]); mlp_params(&G[1], PG[1]); mlp_params(&Dr, PDr); mlp_params(&Dc, PDc);

    printf("d2g2gan  B=%d Z=%d H=%d steps=%d lr=%g lambda=%g seed=0x%llx\n", c.B, c.Z, c.H, c.steps, c.lr, c.lambda, (unsigned long long)c.seed);
    printf("persist arena: %zu B reserved / %zu B used / %zu allocs\n", persist.total_reserved, persist.total_used, persist.n_allocs);
    double t0 = now_s(); size_t peak_frame = 0;

    for (int step = 1; step <= c.steps; step++) {
        ArenaMark fm = arena_mark(&frame);
        int B = c.B;
        float *xr = arena_floats(&frame, (size_t)B * c.X); sample_ring(&rng, xr, B);
        float *z0 = arena_floats(&frame, (size_t)B * c.Z); fill_gauss(&rng, z0, (size_t)B * c.Z, 1.0f);
        float *z1 = arena_floats(&frame, (size_t)B * c.Z); fill_gauss(&rng, z1, (size_t)B * c.Z, 1.0f);

        /* ---- generators forward (fakes are detached for the D steps) */
        Act ag0 = mlp_fwd(&frame, &G[0], z0, B), ag1 = mlp_fwd(&frame, &G[1], z1, B);
        ASSERT(step % c.eval_every != 0 || (k_all_finite(ag0.y, (size_t)B * c.X) && k_all_finite(ag1.y, (size_t)B * c.X)));

        /* ---- D_r step: real->1, fake0->0, fake1->0 (fake terms weighted 1/2) */
        float *dl = arena_floats(&frame, (size_t)B);
        Act ar = mlp_fwd(&frame, &Dr, xr, B);  float lDr = k_bce_logits(ar.y, dl, B, 1.0f, 1.0f);   mlp_bwd(&frame, &Dr, xr, &ar, dl, 1);
        /* accumulate grads across the three sub-batches: copy W grads after each bwd (cold path, per step) */
        float *gW1 = arena_floats(&frame, Dr.W1.n), *gb1 = arena_floats(&frame, Dr.b1.n), *gW2 = arena_floats(&frame, Dr.W2.n), *gb2 = arena_floats(&frame, Dr.b2.n);
        memcpy(gW1, Dr.W1.g, Dr.W1.n * sizeof(float)); memcpy(gb1, Dr.b1.g, Dr.b1.n * sizeof(float)); memcpy(gW2, Dr.W2.g, Dr.W2.n * sizeof(float)); memcpy(gb2, Dr.b2.g, Dr.b2.n * sizeof(float));
        Act af0 = mlp_fwd(&frame, &Dr, ag0.y, B); lDr += k_bce_logits(af0.y, dl, B, 0.0f, 0.5f) * 0.5f; mlp_bwd(&frame, &Dr, ag0.y, &af0, dl, 1);
        vadd(gW1, Dr.W1.g, Dr.W1.n); vadd(gb1, Dr.b1.g, Dr.b1.n);
        vadd(gW2, Dr.W2.g, Dr.W2.n); vadd(gb2, Dr.b2.g, Dr.b2.n);
        Act af1 = mlp_fwd(&frame, &Dr, ag1.y, B); lDr += k_bce_logits(af1.y, dl, B, 0.0f, 0.5f) * 0.5f; mlp_bwd(&frame, &Dr, ag1.y, &af1, dl, 1);
        vadd(Dr.W1.g, gW1, Dr.W1.n); vadd(Dr.b1.g, gb1, Dr.b1.n);
        vadd(Dr.W2.g, gW2, Dr.W2.n); vadd(Dr.b2.g, gb2, Dr.b2.n);
        adam_step(&optDr, PDr, 4);

        /* ---- D_c step: fake0->0, fake1->1 */
        Act ac0 = mlp_fwd(&frame, &Dc, ag0.y, B); float lDc = k_bce_logits(ac0.y, dl, B, 0.0f, 0.5f) * 0.5f; mlp_bwd(&frame, &Dc, ag0.y, &ac0, dl, 1);
        float *hW1 = arena_floats(&frame, Dc.W1.n), *hb1 = arena_floats(&frame, Dc.b1.n), *hW2 = arena_floats(&frame, Dc.W2.n), *hb2 = arena_floats(&frame, Dc.b2.n);
        memcpy(hW1, Dc.W1.g, Dc.W1.n * sizeof(float)); memcpy(hb1, Dc.b1.g, Dc.b1.n * sizeof(float)); memcpy(hW2, Dc.W2.g, Dc.W2.n * sizeof(float)); memcpy(hb2, Dc.b2.g, Dc.b2.n * sizeof(float));
        Act ac1 = mlp_fwd(&frame, &Dc, ag1.y, B); lDc += k_bce_logits(ac1.y, dl, B, 1.0f, 0.5f) * 0.5f; mlp_bwd(&frame, &Dc, ag1.y, &ac1, dl, 1);
        vadd(Dc.W1.g, hW1, Dc.W1.n); vadd(Dc.b1.g, hb1, Dc.b1.n);
        vadd(Dc.W2.g, hW2, Dc.W2.n); vadd(Dc.b2.g, hb2, Dc.b2.n);
        adam_step(&optDc, PDc, 4);

        /* ---- G_k steps: fresh z, D params frozen (accumulate_params=0), dx flows into G */
        float lG[2] = { 0, 0 };
        for (int k = 0; k < 2; k++) {
            float *z = arena_floats(&frame, (size_t)B * c.Z); fill_gauss(&rng, z, (size_t)B * c.Z, 1.0f);
            Act ag = mlp_fwd(&frame, &G[k], z, B);
            Act dr = mlp_fwd(&frame, &Dr, ag.y, B); float *dlr = arena_floats(&frame, (size_t)B);
            float lr_ = k_bce_logits(dr.y, dlr, B, 1.0f, 1.0f); mlp_bwd(&frame, &Dr, ag.y, &dr, dlr, 0);
            Act dc = mlp_fwd(&frame, &Dc, ag.y, B); float *dlc = arena_floats(&frame, (size_t)B);
            float lc_ = k_bce_logits(dc.y, dlc, B, (float)k, c.lambda); mlp_bwd(&frame, &Dc, ag.y, &dc, dlc, 0);
            float *dx = arena_floats(&frame, (size_t)B * c.X);
            for (size_t i = 0; i < (size_t)B * c.X; i++) dx[i] = dr.dx[i] + dc.dx[i];
            mlp_bwd(&frame, &G[k], z, &ag, dx, 1);
            adam_step(&optG[k], PG[k], 4);
            lG[k] = lr_ + c.lambda * lc_;
        }

        if (frame.total_used > peak_frame) peak_frame = frame.total_used;
        if (step % c.eval_every == 0 || step == c.steps) {
            int n = 4096;
            float *ze = arena_floats(&frame, (size_t)n * c.Z); float *xe = arena_floats(&frame, (size_t)2 * n * c.X);
            float *xreal = arena_floats(&frame, (size_t)2 * n * c.X); sample_ring(&rng, xreal, 2 * n);
            fill_gauss(&rng, ze, (size_t)n * c.Z, 1.0f); Act e0 = mlp_fwd(&frame, &G[0], ze, n); memcpy(xe, e0.y, (size_t)n * c.X * sizeof(float));
            fill_gauss(&rng, ze, (size_t)n * c.Z, 1.0f); Act e1 = mlp_fwd(&frame, &G[1], ze, n); memcpy(xe + (size_t)n * c.X, e1.y, (size_t)n * c.X * sizeof(float));
            ASSERT(k_all_finite(xe, (size_t)2 * n * c.X));
            for (int k = 0; k < 4; k++) { ASSERT(k_all_finite(PG[0][k].p, PG[0][k].n) && k_all_finite(PG[1][k].p, PG[1][k].n) && k_all_finite(PDr[k].p, PDr[k].n) && k_all_finite(PDc[k].p, PDc[k].n)); }
            int cov0, cov1, covU; float hq0, hq1, hqU;
            coverage(xe, n, &cov0, &hq0); coverage(xe + (size_t)n * c.X, n, &cov1, &hq1); coverage(xe, 2 * n, &covU, &hqU);
            float fdU = frechet2(xreal, 2 * n, xe, 2 * n), fd0 = frechet2(xreal, 2 * n, xe, n), fd1 = frechet2(xreal, 2 * n, xe + (size_t)n * c.X, n);
            printf("step %5d  L_Dr=%.3f L_Dc=%.3f L_G0=%.3f L_G1=%.3f | modes G0=%d G1=%d union=%d/8  hq G0=%.2f G1=%.2f union=%.2f | FD G0=%.3f G1=%.3f union=%.3f | %.1fs\n",
                   step, lDr, lDc, lG[0], lG[1], cov0, cov1, covU, hq0, hq1, hqU, fd0, fd1, fdU, now_s() - t0);
            fflush(stdout);
        }
        arena_reset_to(&frame, fm);
        ASSERT(frame.total_used == 0 && frame.n_chunks >= 1);
    }
    /* final sample dump for external plotting / second-method verification */
    {
        int n = 2000; float *ze = arena_floats(&frame, (size_t)n * c.Z);
        FILE *f = fopen("samples.tsv", "w"); ASSERT(f);
        for (int k = 0; k < 2; k++) { fill_gauss(&rng, ze, (size_t)n * c.Z, 1.0f); Act e = mlp_fwd(&frame, &G[k], ze, n);
            for (int i = 0; i < n; i++) fprintf(f, "%d\t%.5f\t%.5f\n", k, e.y[2 * i], e.y[2 * i + 1]); }
        fclose(f);
    }
    printf("frame arena peak %zu B (%zu chunks reserved %zu B); persist %zu B; elapsed %.2fs; %.1f steps/s\n",
           peak_frame, frame.n_chunks, frame.total_reserved, persist.total_used, now_s() - t0, c.steps / (now_s() - t0));
    return 0;
}
