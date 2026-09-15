# d2g2gan — dual-discriminator / dual-generator GAN in C, two build targets from one source.
#
#   make native      gcc + system OpenBLAS (Ubuntu: apt install libopenblas-dev)     -> build/d2g2gan.native
#   make clapack     fetch + build netlib CLAPACK 3.2.1 with cosmocc (pure C, static) -> third_party/CLAPACK-3.2.1/*.a
#   make cosmo       cosmocc + CLAPACK                                                -> build/d2g2gan.com (APE)
#   make test        300-step smoke run of whichever targets are built
#   make clean
#
# Override: COSMOCC=/opt/cosmocc/bin/cosmocc  CLAPACK_URL=...  STEPS=300

CC        ?= gcc
COSMOCC   ?= cosmocc
COSMOBIN  := $(dir $(shell command -v $(COSMOCC) 2>/dev/null))
COSMOAR   ?= $(COSMOBIN)cosmoar
COSMORANLIB ?= $(COSMOBIN)cosmoranlib
CFLAGS    ?= -O2 -std=c11 -Wall -Wextra
STEPS     ?= 300

CLAPACK_URL ?= https://www.netlib.org/clapack/clapack.tgz
CLAPACK_DIR := third_party/CLAPACK-3.2.1
CLAPACK_LIBS := $(CLAPACK_DIR)/lapack_COSMO.a $(CLAPACK_DIR)/blas_COSMO.a $(CLAPACK_DIR)/F2CLIBS/libf2c.a

.PHONY: all native cosmo clapack test clean

all: native

build:
	mkdir -p build

native: build
	$(CC) $(CFLAGS) -o build/d2g2gan.native d2g2gan.c -lopenblas -lm

# --- CLAPACK under cosmocc -------------------------------------------------------------------------
# Every line below is a gotcha that was hit for real (2026-09-14, see README "Cosmopolitan notes"):
#   * cosmocc is GCC 14: f2c output needs gnu89 + pre-included headers or implicit decls hard-error
#   * F2CLIBS/libf2c/Makefile hardcodes GNU ar -> re-archive with cosmoar so the .aarch64 twin exists
#   * cosmocc.zip ships cosmoranlib without the exec bit
#   * CLAPACK's lapacklib target RUNS its INSTALL test binaries -> APE must be executable here
#     (either register binfmt_misc ':APE:M::MZqFpD::<ape loader>:' or let /bin/sh's ENOEXEC fallback run the stub)
$(CLAPACK_DIR)/make.inc:
	mkdir -p third_party
	cd third_party && curl -fsSLO $(CLAPACK_URL) && tar xzf clapack.tgz && rm -f clapack.tgz
	sed -e 's|^PLAT *=.*|PLAT = _COSMO|' \
	    -e 's|^CC *=.*|CC = $(COSMOCC)|' \
	    -e 's|^LOADER *=.*|LOADER = $(COSMOCC)|' \
	    -e 's|^CFLAGS *=.*|CFLAGS = -O2 -std=gnu89 -Wno-implicit-function-declaration -include stdio.h -include string.h -include stdlib.h -I$$(TOPDIR)/INCLUDE|' \
	    -e 's|^NOOPT *=.*|NOOPT = -O0 -std=gnu89 -Wno-implicit-function-declaration -include stdio.h -include string.h -include stdlib.h -I$$(TOPDIR)/INCLUDE|' \
	    -e 's|^ARCH *=.*|ARCH = $(COSMOAR)|' \
	    -e 's|^RANLIB *=.*|RANLIB = $(COSMORANLIB)|' \
	    $(CLAPACK_DIR)/make.inc.example > $@

clapack: $(CLAPACK_DIR)/make.inc
	chmod +x $(COSMOBIN)* 2>/dev/null || true
	$(MAKE) -C $(CLAPACK_DIR) f2clib
	cd $(CLAPACK_DIR)/F2CLIBS/libf2c && rm -f ../libf2c.a && $(COSMOAR) cr ../libf2c.a *.o && $(COSMORANLIB) ../libf2c.a
	$(MAKE) -C $(CLAPACK_DIR) blaslib
	$(MAKE) -C $(CLAPACK_DIR) lapacklib
	$(COSMORANLIB) $(CLAPACK_DIR)/lapack_COSMO.a $(CLAPACK_DIR)/blas_COSMO.a
	@ls -la $(CLAPACK_LIBS)

$(CLAPACK_DIR)/lapack_COSMO.a:
	$(MAKE) clapack

cosmo: build $(CLAPACK_DIR)/lapack_COSMO.a
	$(COSMOCC) $(CFLAGS) -DCLAPACK_F2C_WRAP -o build/d2g2gan.com d2g2gan.c $(CLAPACK_LIBS) -lm

test:
	@set -e; ran=0; \
	for b in build/d2g2gan.native build/d2g2gan.com; do \
	  if [ -x $$b ]; then echo "== $$b"; OPENBLAS_NUM_THREADS=1 $$b --hidden 128 --lr 1e-3 --steps $(STEPS) --eval-every $(STEPS) | tail -3; ran=1; fi; \
	done; \
	[ $$ran -eq 1 ] || { echo "nothing built: run make native and/or make cosmo"; exit 1; }

clean:
	rm -rf build samples.tsv
