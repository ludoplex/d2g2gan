# d2g2gan

A dual-discriminator / dual-generator adversarial network in one C file, built two ways from the
same source:

| target | toolchain | BLAS / LAPACK | output |
|---|---|---|---|
| `make native` | gcc | OpenBLAS (`sgemm_`, `ssyev_`, `sgeqrf_`, `sorgqr_`) | `build/d2g2gan.native` |
| `make cosmo` | [cosmocc](https://github.com/jart/cosmopolitan) | netlib CLAPACK 3.2.1 (f2c'd reference BLAS + LAPACK, pure C, static) | `build/d2g2gan.com` — one Actually Portable Executable |

Splash page with the results: https://ludoplex.github.io/d2g2gan/

## The model

Two generators and two discriminators with *different jobs* (MGAN-style, K = 2):

```
G_k : z(B,8) -> lrelu(z W1_k + b1_k) -> x(B,2)                  k in {0,1}
D_r : x -> logit    real = 1  vs  fake = 0            (BCE, fakes weighted 1/2 each)
D_c : x -> logit    from G_1 = 1  vs  from G_0 = 0    (BCE) — "which generator made this?"

L_Gk = BCE(D_r(G_k z), 1) + lambda * BCE(D_c(G_k z), k)
```

`D_r` keeps samples on the data manifold; `D_c` pays each generator to be *identifiable*, which is
only possible if the two generators cover different modes. Target is the 8-Gaussian ring
(radius 2, sigma 0.05) — the standard mode-collapse toy — so the result is checkable by counting.

Result (seed `0x9e3779b97f4a7c15`, `--hidden 128 --lr 1e-3 --steps 6000`):

| build | modes G0 / G1 / union | high-quality (within 3σ) | Fréchet dist. union | speed |
|---|---|---|---|---|
| APE, CLAPACK reference BLAS | 3 / 4 / **7 of 8** | 0.64 | 0.057 | 65.8 steps/s |
| native, OpenBLAS (1 thread) | 4 / 4 / **8 of 8** | 0.64 | 0.051 | 158.6 steps/s |

`L_Dc -> 0.000` on both: the generators become perfectly separable, i.e. they split the ring.

## What the code insists on

- **Every matmul is Fortran-ABI `sgemm_`** through the row-major identity
  `C = op(A) op(B)  <=>  C^T = op(B)^T op(A)^T`, so the same call works against OpenBLAS and
  against f2c'd netlib BLAS.
- **LAPACK does real work**: `sgeqrf_` + `sorgqr_` give orthogonal weight init; `ssyev_` gives the
  symmetric matrix square root inside the Fréchet (2-Wasserstein Gaussian) metric.
- **Branchless hot paths**: leaky-ReLU forward/backward via `(float)(x > 0)` masks,
  `sigmoid = 0.5*tanh(0.5x)+0.5`, `softplus = fmaxf(x,0) + log1pf(expf(-fabsf(x)))`, Adam, the
  ring sampler and the coverage binning — no `if` inside an elementwise loop.
- **Dynamic arenas**: a 64-byte-aligned chunked bump allocator with mark/reset and chunk doubling.
  A persistent arena holds parameters + Adam state (62 KB at H=128); a frame arena is reset every
  step (peak 4.95 MB at H=128) and a postcondition asserts it is empty.
- **Asserts stay on** (independent of `NDEBUG`): shapes, leading dimensions, alignment and arena
  invariants at every kernel entry, finite-checks on parameters and samples at eval cadence.
  The very first APE run aborted inside `sgeqrf_` — that is how the 8-byte-integer ABI mismatch
  below was found.

## Cosmopolitan notes (each one was hit for real)

1. OpenBLAS is a dynamic glibc library — unusable from a static APE. Netlib CLAPACK 3.2.1
   (https://www.netlib.org/clapack/clapack.tgz) is f2c-translated C and builds with any C compiler.
2. cosmocc is GCC 14, which hard-errors on implicit declarations that 2009-era f2c output relies
   on: build CLAPACK with `-std=gnu89 -Wno-implicit-function-declaration -include stdio.h -include string.h -include stdlib.h`.
3. `F2CLIBS/libf2c/Makefile` hardcodes GNU `ar`, so the archive lacks the `.aarch64/` fat-object
   twin and the link fails with *missing concomitant ../F2CLIBS/.aarch64/libf2c.a*. Re-archive with
   `cosmoar`.
4. `cosmocc.zip` ships `cosmoranlib` without the execute bit.
5. CLAPACK's `lapacklib` target runs its `INSTALL/` test binaries, so APE files must be executable
   on the build host (register `binfmt_misc` or rely on `/bin/sh`'s ENOEXEC fallback).
6. Link the archives by path — they are not `lib*.a`.
7. `INCLUDE/blaswrap.h:127` renames BLAS entry points (`sgemm_` → `f2c_sgemm`); LAPACK symbols keep
   the underscore. `-DCLAPACK_F2C_WRAP` maps it in the source.
8. `f2c.h` declares `typedef long int integer` — 8 bytes on LP64 — while OpenBLAS uses 32-bit
   `int`. All BLAS/LAPACK integer arguments go through `fint`, selected by the same flag.

## Build

```sh
make native            # needs libopenblas-dev
make cosmo             # needs cosmocc on PATH (or COSMOCC=/opt/cosmocc/bin/cosmocc); fetches + builds CLAPACK
make test STEPS=300    # smoke both, asserts on
./build/d2g2gan.com --hidden 128 --lr 1e-3 --steps 6000 --eval-every 1000   # writes samples.tsv
```

Options: `--steps --batch --hidden --lr --lambda --seed --eval-every`. Default H=64 / lr=2e-4 is
too weak for `D_r` on this target (it sits at chance, 2·ln 2); use H ≥ 128 with lr = 1e-3.

`results/` holds the sample dumps behind the numbers above. License: MIT.
