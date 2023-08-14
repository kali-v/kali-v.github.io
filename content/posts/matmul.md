---
url: /posts/matmul
title: Not So Slow Matrix Multiplication
date: 2022-07-10
desc: [
    "<b>date</b>: 2022-07-15",
    "<b>code</b>: <a href=\"https://github.com/kali-v/lightflow/blob/master/extra/matmul.cc\" > here </a>"
]
---

- [Naive Matmul](#naive-matmul)
- [Tiling](#tiling)
- [AVX and FMA](#avx-and-fma)
- [AVX and FMA tiling](#avx-and-fma-tiling)
- [Aligning Data](#aligning-data)
- [Cheap Tricks With Compiler](#cheap-tricks-with-compiler)
  - [-Ofast and -funroll(-all)-loops](#-ofast-and--funroll-all-loops)
  - [FMA Without Using FMA Intrinsics](#fma-without-using-fma-intrinsics)
- [Comparison with numpy and cblas_sgemm](#comparison-with-numpy-and-cblas_sgemm)

Everybody likes fast matmul, so let's rewrite it from scratch, rulez in short:
- matrices (row-major order) with arbitrary size 
- cpp, no 3rd party libs
- cpu and single thread only
- single precision floats


Execution time is measure using monotonic clock. Matrices are filled with random numbers drawn from standard normal distribution.

```cpp
void fill_matrix(float* mat, int s) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};

    for (int i = 0; i < s; i++) {
        mat[i] = d(gen);
    }
}

...
int main() {
    float* a = new float[AH * AW];
    float* b = new float[AW * BW];
    float* c = new float[AH * BW];

    fill_matrix(a, AH * AW);
    fill_matrix(b, AW * BW);
    
    // ...

    auto st = std::chrono::steady_clock::now();
    matmul(a, b, c);
    auto et = std::chrono::steady_clock::now();

    // ...
}

```


# Naive Matmul

The easiest way to think of matrix multiplication C=A@B is:
 - `row` = n-th row from matrix A
 - `col` = m-th column from matrix B
 - `C[n, m]` = dot product of `row` and `col`

Let's start easy with the most basic implementation, the ijk cycle. It's straightforward and slow, but if you wait long enough it will give you the correct result.

```cpp
// g++ -O2 matmul.cc -o matmul
// AH, AW - height, width of matrix A; BW - width of B
// a[AH * AW]; b[AW * BW]
// c = a @ b

void naive_matmul(float* a, float* b, float* c) {
    for (int i = 0; i < AH; i++) {
        for (int j = 0; j < BW; j++) {
            for (int k = 0; k < AW; k++) {
                c[i * BW + j] += a[i * AW + k] * b[k * BW + j];
            }
        }
    }
}
```

Multiplying two 2048x2048 matrices took me over 25s, and because there are `AH * BW * AW * 2` floating-point operations (3 loops and mul + add), we achieved ~0.68 GFLOP/s.

The main problem here is the cache miss rate. If we look at cachegrind, we can see that the D1 miss rate is over 30%.

```asm
==61448== D   refs:      26,230,132,749  (17,502,568,259 rd   + 8,727,564,490 wr)
==61448== D1  misses:     8,599,903,549  ( 8,599,376,753 rd   +       526,796 wr)
==61448== LLd misses:         1,059,288  (       533,444 rd   +       525,844 wr)
==61448== D1  miss rate:           32.8% (          49.1%     +           0.0%  )
```

A partial solution to this is to switch the order of the j-k loops, so we access `c`, and `b` sequentially. Additionally we can load value from `a` outside the j-loop to reduce number of reads.

```cpp
void matmul_baseline(float* a, float* b, float* c) {
    for (int i = 0; i < AH; i++) {
        for (int k = 0; k < AW; k++) {
            float av = a[i * AW + k];
            for (int j = 0; j < BW; j++) {
                c[i * BW + j] += av * b[k * BW + j];
            }
        }
    }
}
```

2750 ms, 6.2 GFLOP/s - 9x speedup, with only 2.1% D1 miss rate. Not bad, but if we try to think how the cache is used, it's still not ideal - mostly with accesses to `b` matrix.
The matmul goes like this:
- fetch part of the first row from all matrices
- end of j cycle; `k++`; fetch part of the next row from B
- end of j cycle; `k++`; fetch part of the next row from B
- and again until the end of the k cycle
- `i++; k=0; j=0;` now we would like the first row from `b`, but because we already loaded 2047 previous rows (in our 2048x2048 matrices example) the first row won't be in cache, so it needs to be fetched again, and again for every other row of B.
    - If these rows would fit into a cache, it would be a nice speedup (tiling addresses this issue)


# Tiling

Tiling is a technique that splits the matrices into smaller blocks - compute matmul with these blocks and accumulate results to the final matrix.

```cpp
void matmul_tiling(float* a, float* b, float* c) {
    const int BSA = 32;
    const int BSB = 512;

    for (int ba = 0; ba < AW; ba += BSA) {
        for (int i = 0; i < AH; i++) {
            for (int bb = 0; bb < BW; bb += BSB) {
                for (int k = ba; k < std::min(ba + BSA, AW); k++) {
                    float av = a[i * AW + k];
                    for (int j = bb; j < std::min(bb + BSB, BW); j++) {
                        c[i * BW + j] += av * b[k * BW + j];
                    }
                }
            }
        }
    }
}
```

2260 ms; 7.6 GFLOP/s.  

There are two constants `BSA` and `BSB`, that need to be adjusted to hardware for best performance.
Higher `BSB` allows us to cheaply accumulate more results. Higher `BSA` would process more `b` rows "at once," which might cause the same problem as in the 
previous implementation. The best compromise depends on the architecture. I received the best results with a 32 and 512, which is bigger than what fits
into an D1 cache, so the cache misses might not necessarily decrease, however this more localized approach allows the CPU more coherent access to the data and a decent speedup.


# AVX and FMA

The next step is vectorization. So far, we have just multiplied (and added) a single float at a time. Using AVX intrinsics and FMA instructions, we employ a vector processor to process multiple floats simultaneously.

![avx-matmul](/images/matmul/avx-matmul.png)

In the image, the lengths of vectors are only 64 bits (2 floats) for the best clarity, but the real AVX registers that we are going to use are 256 bits (8 floats). Moreover, the extension AVX-512 uses even bigger registers called `zmm` that are 512 bits long (16 floats)

Alongside the number of loaded floats, another thing needs to be fixed - this algorithm itself cannot multiply arbitrary large matrices.
When multiplying matrices that don't have sizes of multiplies of 8, the load will overflow, and if the program doesn't segfault, we will have trash in the loaded vector, and thus wrong results. The best way around this is to simply stop before overflowing and compute the rest one by one.


```cpp
// g++ -mavx -mfma -O2 matmul.cc -o matmul
void matmul_avx(float* a, float* b, float* res) {
    for (int i = 0; i < AH; i++) {
        for (int j = 0; j < AW; j++) {
            __m256 vec_a = _mm256_set1_ps(a[i * AW + j]);
            int k;
            for (k = 0; k <= BW - 8; k += 8) {
                _mm256_storeu_ps(&res[i * BW + k], _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&b[j * BW + k]),
                                                                   _mm256_loadu_ps(&res[i * BW + k])));
            }
            // compute exceding elements
            for (int q = 0; q + k < BW; q++) {
                res[i * BW + k + q] += a[i * AW + j] * b[j * BW + k + q];
            }
        }
    }
}
```

16 GFLOP/s, which is more than twice the speedup on the 2048x2048 matrix. Of course, this is an ideal scenario where there are no exceeding elements. However, in the vast majority of cases, the leftover computation is negligible.


Unfortunately, I don't have a way to test AVX-512 instructions. If your CPU supports it, replace all `_mm256` intrinsics with `_mm512` and change the k-cycle to load every 16th instead of every 8th position.


# AVX and FMA Tiling

Now let's implement tiling into our AVX-FMA approach.


```cpp
void matmul_avx_tiling(float* a, float* b, float* res) {
    const int BSB = 64;
    const int BSA = 4;

    for (int bb = 0; bb < AH; bb += BSB) {
        float bbm = std::min(bb + BSB, AH);
        for (int ba = 0; ba < AW; ba += BSA) {
            float bam = std::min(ba + BSA, AW);
            for (int i = bb; i < bbm; i++) {
                for (int j = ba; j < bam; j++) {
                    __m256 vec_a = _mm256_set1_ps(a[i * AW + j]);

                    int k;
                    for (k = 0; k <= BW - 8; k += 8) {
                        _mm256_storeu_ps(&res[i * BW + k], _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&b[j * BW + k]),
                                                                           _mm256_loadu_ps(&res[i * BW + k])));
                    }

                    // compute exceding elements
                    for (int q = 0; q + k < BW; q++) {
                        res[i * BW + k + q] += a[i * AW + j] * b[j * BW + k + q];
                    }
                }
            }
        }
    }
}
```

The average time of multiplying two 2048x2048 matrices is about 500ms with around 34 GFLOP/s. That's a big improvement from the initial 25 seconds, but we can do better...

# Aligning Data

Another speed-up might be achieved by aligning the data. Each time we call `_mm256_loadu_ps`, it loads 8 floats (32 bytes) into a register, but because our data might stretch over 2 cache lines (that are usually 64 bytes long), it will need to load 2 cache lines, which unnecessarily slows down the computation. By using the `alignas` specifier, like this:

```cpp
float* a = new alignas(32) float[AH * AW];
float* b = new alignas(32) float[AW * BW];
float* c = new alignas(32) float[AH * BW];
```

we obtain over 40 GFLOP/s, which is a pretty decent speedup compared to the previous version without even changing the matmul function itself.

Note that using intrinsics for aligned data, such as `_mm256_load_ps` instead of `_mm256_loadu_ps`, and `_mm256_store_ps` instead of `_mm256_storeu_ps`, leads to negligible performance gain, but they can be used to assert the alignment of data at runtime because they crash on an unaligned chunk of data.


# Cheap Tricks With Compiler

## -Ofast and -funroll(-all)-loops

So far, we have only used the `-O2` switch and `-mavx -mfma` if needed. `-O2` is usually considered a fair tradeoff between compile time, size of binary, and execution time. However, if you want to push the limits, there are some ways.


|                    |  -O0  |  -O2  | -Ofast | -Ofast -funroll-all-loops |
| :----------------: | :---: | :---: | :----: | :-----------------------: |
|  **matmul_baseline**   |  1.4  |  6.2  |  15.5  |           17.4            |
|   **matmul_tiling**    |  0.9  |  7.6  |  21.5  |           26.5            |
|     **matmul_avx**     |  5.5  | 16.1  |  16.6  |           20.0            |
| **matmul_avx_tiling**  |  6.4  | 33.4  |  34.8  |           34.1            |
| **matmul_avx_aligned** |  6.8  | 40.1  |  50.8  |           56.6            |

&ast; naive_matmul has been omitted, as it's slow af anyway. All the results are in GLOP/s on 2048x2048 matrices

A few more notes on this:
- `-funroll-loops` was overall slightly worse than `-funroll-all-loops`, and both of them are gambles, not certainty of better performance.
- Performance with `-Ofast` was slightly better(~1 GFLOP/s) than with `-O3`.
- `-mavx -mfma` had pretty much the same performance as targeting specific architecture using `-march` and `-mtune` 


## FMA Without Using FMA Intrinsics

There is one more thing about compiler optimizations. Nowadays compilers are really smart and can do a lot of things for you.

Let's take a look on the mul and add part of the asm `matmul_baseline` code compiled with `-O2`.
```asm
16a8:	0f 59 c1             	mulps  %xmm1,%xmm0
16ab:	0f 58 c2             	addps  %xmm2,%xmm0
```

Nothing special, huh? But, if you compile the code with `-O2 -mfma`. These instructions will be fused into single fma instruction:
```asm
159c:	c4 e2 71 a9 00       	vfmadd213ss (%rax),%xmm1,%xmm0
```

Now, this `matmul_baseline` code can achieve over 7 GLOP/s.

However, if we compile it with `-O3 -mfma`, the compiler will optimize even more aggressively and will use packed `vfmadd213ps` instruction instead of the scalar one and will utilize wider `ymm` registers - that are twice as big as `xmm`. (`xmm` is lower half of 256-bits long `ymm`.)
```asm
17e6:	c4 e2 75 a8 04 02    	vfmadd213ps (%rdx,%rax,1),%ymm1,%ymm0
``` 

21 GFLOP/s as result.


# Comparison with numpy and cblas_sgemm

Before we spoil the fun, let's look at a comparison with numpy that doesn't have super quick matmul.

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time

SIZE = 2048

a = np.random.rand(SIZE, SIZE)
b = np.random.rand(SIZE, SIZE)

for i in range(10):
    st = time.monotonic()
    c = a @ b
    et = time.monotonic()

    print((2 * SIZE**3) / ((et - st) * 1e9))
```

63 GFLOP/s, which means that our best approach has about 90% performance of numpy - not great, not terrible. 

Now, on to the different beast:  

```cpp
void matmul_cblas(float* a, float* b, float* c) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, AH, BW, AW, 1.0f, a, AW, b, BW, 1.0f, c, BW);
}
```
`cblas_sgemm` is optimized to the bone resulting in a whooping 130 GFLOP/s i.e., ~2x quicker than numpy. I benchmarked everything on 12th Gen Intel(R) Core(TM) i5-12600H which is one of the newer CPUs, so the cblas was able to really exploit all the benefits and quirks of the Alder Lake architecture.



