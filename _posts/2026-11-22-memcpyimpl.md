---
title: "Implementing memcpy from Scratch"
categories: [Tech]
tags: []
math: true
description: "Speeding up memcpy on an M1 Mac one step at a time, using measurements and assembly to find out what matters"
---

This post is part of [my series on High Performance Computing]({% post_url 2026-10-12-planning %}).

# What this post covers
This time I'll implement `memcpy` myself, from scratch. I'll start with the simplest possible version, which copies one byte at a time, and improve it step by step until it's about as fast as the library `memcpy`.

What I want to get across isn't a set of tricks that make code fast. It's the process: **make one change at a time, measure every time, and if the result isn't what you expected, go read the assembly**. Honestly, the compiler does most of what this post covers if you just pass `-O3` (I'll check that at the end). Even so, in real optimization work the compiler sometimes doesn't produce the code you expect and you end up reading assembly, and you can't know what helps until you measure. That's why I add changes one at a time and isolate each effect.

All the code is in [Materials](/assets/1122/memcpyimpl.zip).

# Hardware specs

|Item|Value|
|---|---|
|Machine|MacBook Air (M1, 2020)|
|CPU|Apple M1 (4 Performance cores + 4 Efficiency cores)|
|Memory|16 GB LPDDR4X-4266|
|Memory Bus|128 bit|
|Memory Bandwidth|68.25 GB/s (calculated, see below)|
|Compiler|Apple clang 17.0.0 (`gcc` is an alias for clang)|

# 1. Theoretical speed and the measurement setup
## Theoretical memory bandwidth
`memcpy` does almost no computation. It just reads from memory and writes to memory. So the limit isn't the FLOPS we dealt with in [the post before last]({% post_url 2026-10-19-howtoseecatalogspec %}) and [the previous post]({% post_url 2026-10-31-hpcformulas %}). It's **memory bandwidth**.

The M1 connects LPDDR4X-4266 memory over a 128-bit (16-byte) bus [[1]](#ref-m1-wiki). 4266 is the number of transfers per second (MT/s), so the theoretical bandwidth is

$$
4266 \times 10^6\,[\text{transfer/s}] \times 16\,[\text{byte/transfer}] \approx 68.25\,\text{GB/s}
$$

Just as the previous post broke peak performance down into "SIMD width × instructions × frequency × cores", memory bandwidth breaks down into "transfer rate × bus width". This 68.25 GB/s is the ceiling for this post.

## Formula for bandwidth
I copy an array of $N = 2^{30}$ bytes (1 GiB). `memcpy` reads $N$ bytes from `src` and writes $N$ bytes to `dst`, so $2N$ bytes move between the CPU and memory in total. If one copy takes time $t$, the achieved bandwidth is

$$
\text{Bandwidth} = \frac{2N}{t}
$$

Strictly speaking, CPUs often read the destination cache line before writing to it (write-allocate), so the actual memory traffic can be $3N$. But here I count "how many bytes `memcpy` moved" and compare against the library `memcpy` with the same formula, so I use $2N$.

## Measurement code
For timing I use a small header called `timeutils.h`.

```c
#ifndef UTIL_H_
#define UTIL_H_

#include <stdlib.h>
#include <time.h>

typedef struct timespec Time;
Time* timeEvent(){
    Time* t = (Time*)malloc(1 * sizeof(Time));
    clock_gettime(CLOCK_MONOTONIC, t);
    return t;
}

double elapsedTime(Time* start, Time* end){
    return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1e9;
}

long long sum(char* a, long long N){
    long long s = 0;
    for (long long i = 0; i < N; ++i){
        s += a[i];
    }
    return s;
}

#endif
```

- `timeEvent()` gets the current time with `clock_gettime(CLOCK_MONOTONIC, ...)`. `CLOCK_MONOTONIC` never jumps backward (for example when the system clock is adjusted), so it's suited to measuring elapsed time.
- `elapsedTime()` returns the difference between two times in seconds.
- `sum()` is for checking the copy. If `src` is filled with 1s and the copy is correct, the sum of `dst` is $N$.

And `memcpy_sample.c` measures the library `memcpy`.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "timeutils.h"

int main(){
    long long N = 1 << 30; // 1G elements
    long long iter = 10;
    char* src = (char*)malloc(sizeof(char) * N);
    char* dst = (char*)malloc(sizeof(char) * N);
    memset((void *)src, 1, N);
    memset((void *)dst, 0, N); // init and alloc physical memory

    Time* start = timeEvent();
    for (long long i = 0; i < iter; ++i) memcpy((void *)dst, (void *)src, N);
    Time* end = timeEvent();
    // checksum
    assert(sum(dst, N) == N);
    double elapsed = elapsedTime(start, end) / iter;
    printf("Elapsed: %lf\n", elapsed);
    printf("Achieved Bandwidth %lf GB/s\n", 2 * N / elapsed / 1e9);
}
```

The key point is that `memset` touches both `src` and `dst` once. Right after `malloc`, physical memory may not be assigned yet (especially early in the process), so the first touch causes page faults that assign it. If that doesn't happen outside the timed region, the page-fault time gets counted in the bandwidth. After that, I take the average over 10 copies.

Also, in this experiment **I assume everything is aligned**. A 1 GiB `malloc` returns a page-aligned address, and $N$ is a multiple of 128, so the "leftover" loops that show up later never actually run. A real `memcpy` also handles misaligned heads and tails, but I don't deal with that here.

## Makefile

```makefile
CC=gcc
CFLAGS=-O0
PROGRAM=memcpy_sample memcpy_naive memcpy_8byte memcpy_loopunroll memcpy_simd memcpy_simd_loopunroll memcpy_simd_inlineasm
ASM=memcpy_sample.s memcpy_naive.s memcpy_8byte.s memcpy_loopunroll.s memcpy_simd.s memcpy_simd_loopunroll.s memcpy_simd_inlineasm.s

all: $(PROGRAM) $(ASM)

%: %.c 
	$(CC) $(CFLAGS) -o $@ $<

%.s: %.c
	$(CC) $(CFLAGS) -S -o $@ $<

clean:
	rm -f *.o *~ $(PROGRAM) $(ASM)

.PHONY: all clean
```

`make` builds every executable, plus the assembly (`.s`) via `-S`. The important part is that **everything is compiled with `-O0` (no optimization)**. At `-O0`, the compiler translates the C code into machine code almost line for line. I want to see, one step at a time, what happens when I change how the C code is written, so I don't want the compiler optimizing things on its own. The library `memcpy`, though, is an already-optimized binary, so it stays fast even at `-O0`. In fact, `memcpy_sample.s` just calls the library function with `bl ___memcpy_chk`.

## How I measured
The first run was often slow, so I ran each program 5 times and took the **maximum**. For example, `memcpy_sample` gave these 5 results:

```
31.40 56.93 55.82 56.10 55.10 (GB/s)
```

Only the first run is slow, possibly because the clock frequency hasn't ramped up yet right after starting, among other factors.

## The library `memcpy`
The result is **56.93 GB/s**, **83%** of the theoretical 68.25 GB/s. When reads and writes are mixed there is overhead such as switching the memory between reading and writing, so it's natural not to reach 100%. This is the target for this post.

# 2. Naive implementation
First, the byte-by-byte copy anyone can write.

```c
void *memcpy_naive(void* dst, const void* src, size_t n){
    const char* s = (const char*) src;
    char* d = (char*) dst;
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

`main` is `memcpy_sample.c` with `memcpy` replaced by `memcpy_naive` (the same goes for every version below).

The result is **3.01 GB/s**. That's about 1/19 of the library `memcpy` and 4.4% of the theoretical value. Slow.

To see why, let's look at the assembly generated at `-O0` (`memcpy_naive.s`).

```nasm
LBB3_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #24]
	subs	x9, x8, #1
	str	x9, [sp, #24]
	cbz	x8, LBB3_3
	b	LBB3_2
LBB3_2:                                 ;   in Loop: Header=BB3_1 Depth=1
	ldr	x8, [sp, #16]
	add	x9, x8, #1
	str	x9, [sp, #16]
	ldrb	w8, [x8]
	ldr	x9, [sp, #8]
	add	x10, x9, #1
	str	x10, [sp, #8]
	strb	w8, [x9]
	b	LBB3_1
```

You can see the C code translated almost as is. `n`, `s` and `d` live on the stack (`[sp, #24]` and so on), not in registers, and every iteration loads them from the stack, adds 1, and stores them back. All we really want is two instructions, `ldrb` (load 1 byte) and `strb` (store 1 byte), yet **copying 1 byte takes 14 instructions**. At this point the speed of processing instructions is the bottleneck, and we're nowhere near using up the memory bandwidth.

# 3. Copying 8 bytes at a time
If one instruction moves only 1 byte, it seems we should increase the number of bytes per instruction. This is a 64-bit CPU, so one general-purpose register can carry 8 bytes.

```c
void *memcpy_8byte(void* dst, const void* src, size_t n){
    const char* s = (const char*) src;
    char* d = (char*) dst;

    while (n >= 8){
        *(int64_t*)d = *(const int64_t*)s;
        n -= 8;
        d += 8;
        s += 8;
    }
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

Leftover bytes (fewer than 8) are copied one at a time at the end (this never runs here, since $N$ is a multiple of 8).

The result is **15.34 GB/s**, about 5.1x the naive version.

# 4. The assembly shows the loop bookkeeping is heavy
We're doing 8x the work per iteration but only got 5x faster. Let's look at the assembly (`memcpy_8byte.s`).

```nasm
LBB3_1:                                 ; =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #24]
	subs	x8, x8, #8
	b.lo	LBB3_3
	b	LBB3_2
LBB3_2:                                 ;   in Loop: Header=BB3_1 Depth=1
	ldr	x8, [sp, #16]
	ldr	x8, [x8]
	ldr	x9, [sp, #8]
	str	x8, [x9]
	ldr	x8, [sp, #24]
	subs	x8, x8, #8
	str	x8, [sp, #24]
	ldr	x8, [sp, #8]
	add	x8, x8, #8
	str	x8, [sp, #8]
	ldr	x8, [sp, #16]
	add	x8, x8, #8
	str	x8, [sp, #16]
	b	LBB3_1
```

One iteration is 18 instructions, and only two of them, `ldr x8, [x8]` and `str x8, [x9]`, do the actual copy. Most of the rest is the loop condition and updating `n`, `d` and `s`, in other words **bookkeeping to keep the loop running**. The cost of the loop itself is too large compared with the work done per iteration (8 bytes).

So we should copy more per iteration and run the loop fewer times. That's **loop unrolling**.

# 5. Loop unrolling
Each iteration does sixteen 8-byte copies, 128 bytes in total.

```c
void *memcpy_loopunroll(void* dst, const void* src, size_t n){
    const char* s = (const char*) src;
    char* d = (char*) dst;

    while (n >= 128){
        *(int64_t*)d = *(const int64_t*)s;
        *(int64_t*)(d + 8) = *(const int64_t*)(s + 8);
        *(int64_t*)(d + 16) = *(const int64_t*)(s + 16);
        *(int64_t*)(d + 24) = *(const int64_t*)(s + 24);
        *(int64_t*)(d + 32) = *(const int64_t*)(s + 32);
        *(int64_t*)(d + 40) = *(const int64_t*)(s + 40);
        *(int64_t*)(d + 48) = *(const int64_t*)(s + 48);
        *(int64_t*)(d + 56) = *(const int64_t*)(s + 56);
        *(int64_t*)(d + 64) = *(const int64_t*)(s + 64);
        *(int64_t*)(d + 72) = *(const int64_t*)(s + 72);
        *(int64_t*)(d + 80) = *(const int64_t*)(s + 80);
        *(int64_t*)(d + 88) = *(const int64_t*)(s + 88);
        *(int64_t*)(d + 96) = *(const int64_t*)(s + 96);
        *(int64_t*)(d + 104) = *(const int64_t*)(s + 104);
        *(int64_t*)(d + 112) = *(const int64_t*)(s + 112);
        *(int64_t*)(d + 120) = *(const int64_t*)(s + 120);

        d += 128;
        s += 128;
        n -= 128;
    }
    while (n >= 8){
        *(int64_t*)d = *(const int64_t*)s;
        n -= 8;
        d += 8;
        s += 8;
    }
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

The result is **28.06 GB/s**, another 1.8x over the 8-byte version. That's about half of the library `memcpy`.

In the assembly, the loop body is these 4 instructions repeated.

```nasm
	ldr	x8, [sp, #16]
	ldr	x8, [x8]
	ldr	x9, [sp, #8]
	str	x8, [x9]
```

Since this is `-O0`, it still reloads `s` and `d` from the stack every time, but the loop check and the updates to `n`, `d` and `s` now happen only once per 128 bytes.

A side note: when I first wrote this code, I had a typo that copied `s + 64` into `d + 72`. Because `src` is filled entirely with 1s, copying from the wrong place still gives 1, so the `sum` check passed anyway. A check passing doesn't necessarily mean the code is correct. It's safer to fill the buffer with random values and compare with `memcmp`.

# 6. Using SIMD
Can we move data in units larger than 8 bytes? The M1 (ARMv8) has SIMD instructions called **NEON**, with 32 128-bit (16-byte) registers, `q0` to `q31`. With the intrinsics in `arm_neon.h`, we can read and write 16 bytes at once.

```c
void *memcpy_simd(void* dst, const void* src, size_t n){
    const uint8_t* s = (const uint8_t*) src;
    uint8_t* d = (uint8_t*) dst;

    while (n >= 16){
        uint8x16_t v0 = vld1q_u8(s);
        vst1q_u8(d, v0);

        d += 16;
        s += 16;
        n -= 16;
    }
    while (n >= 8){
        *(int64_t*)d = *(const int64_t*)s;
        n -= 8;
        d += 8;
        s += 8;
    }
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

`vld1q_u8` reads 16 bytes and `vst1q_u8` writes 16 bytes. Each iteration moves twice as many bytes as the 8-byte version, so I'd hope for about twice the 8-byte version's speed (15.34 GB/s).

The result is **14.39 GB/s**. Not as fast as I expected. It's actually slower than the 8-byte version. Let's look at the assembly.

```nasm
LBB3_2:                                 ;   in Loop: Header=BB3_1 Depth=1
	ldr	x8, [sp, #80]
	ldr	q0, [x8]
	str	q0, [sp, #32]
	ldr	q0, [sp, #32]
	str	q0, [sp, #16]
	ldr	q0, [sp, #16]
	str	q0, [sp, #48]
	ldr	q0, [sp, #48]
	str	q0, [sp]
	ldr	x8, [sp, #72]
	ldr	q0, [sp]
	str	q0, [x8]
	...
```

After `ldr q0, [x8]` reads 16 bytes from `src`, it **writes to the stack and reads back** three times, as in `str q0, [sp, #32]` → `ldr q0, [sp, #32]`, before it finally writes to `dst`. At `-O0`, the intrinsic's return value and the variable `v0` are each placed on the stack. As a result, one iteration (16 bytes) takes 26 instructions, and since the stack writes and reloads form a serial chain, each instruction has to wait for the previous one to finish.

# 7. SIMD + loop unrolling
Loop unrolling worked for the 8-byte version, so let's do the same for the SIMD version. Each iteration handles 16 bytes × 8 = 128 bytes.

```c
void *memcpy_simd_unroll(void* restrict dst, const void* restrict src, size_t n){
    const uint8_t* s = (const uint8_t*) src;
    uint8_t* d = (uint8_t*) dst;

    while (n >= 128) {
        uint8x16_t v0 = vld1q_u8(s);
        uint8x16_t v1 = vld1q_u8(s + 16);
        uint8x16_t v2 = vld1q_u8(s + 32);
        uint8x16_t v3 = vld1q_u8(s + 48);
        uint8x16_t v4 = vld1q_u8(s + 64);
        uint8x16_t v5 = vld1q_u8(s + 80);
        uint8x16_t v6 = vld1q_u8(s + 96);
        uint8x16_t v7 = vld1q_u8(s + 112);


        vst1q_u8(d, v0);
        vst1q_u8(d + 16, v1);
        vst1q_u8(d + 32, v2);
        vst1q_u8(d + 48, v3);
        vst1q_u8(d + 64, v4);
        vst1q_u8(d + 80, v5);
        vst1q_u8(d + 96, v6);
        vst1q_u8(d + 112, v7);

        s += 128;
        d += 128;
        n -= 128;
    }
    while (n >= 8){
        *(int64_t*)d = *(const int64_t*)s;
        n -= 8;
        d += 8;
        s += 8;
    }
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

The hope is that issuing all 8 loads first and then all 8 stores lets the load latencies overlap and hide each other.

The result is **20.06 GB/s**. Faster than the plain SIMD version, but it still loses to the 8-byte loop-unrolled version (28.06 GB/s). Let's look at the assembly again.

```nasm
LBB3_2:                                 ;   in Loop: Header=BB3_1 Depth=1
	ldr	x8, [sp, #528]
	ldr	q0, [x8]
	str	q0, [sp, #480]
	ldr	q0, [sp, #480]
	str	q0, [sp, #464]
	ldr	q0, [sp, #464]
	str	q0, [sp, #496]
	ldr	x8, [sp, #528]
	ldr	q0, [x8, #16]
	str	q0, [sp, #432]
	ldr	q0, [sp, #432]
	str	q0, [sp, #416]
	ldr	q0, [sp, #416]
	str	q0, [sp, #448]
	...
	ldr	q0, [sp, #496]
	str	q0, [sp, #112]
	ldr	x8, [sp, #520]
	ldr	q0, [sp, #112]
	str	q0, [x8]
	...
```

Counting how often each `q` register appears in the whole function:

```bash
$ grep -oE "q[0-9]+" memcpy_simd_loopunroll.s | sort | uniq -c
  80 q0
```

**Only `q0` is used.** There are 32 NEON registers, yet `v0` to `v7` all live on the stack, and the single register `q0` is reused to "load and put on the stack" 8 times, then "load from the stack and store" 8 times. This isn't the "load into 8 registers at once, then store at once" I was hoping for, and one iteration (128 bytes) takes 110 instructions.

So why is it slower than the 8-byte loop-unrolled version, which also moves 128 bytes per iteration? Let's compare the number of load (`ldr`) and store (`str`) instructions in the loop body.

|Implementation|ldr|str|Total instructions|
|---|---|---|---|
|loopunroll|51|19|78|
|simd_loopunroll|59|43|110|

In loopunroll, data flows directly from src → register → dst, and the only things read from the stack are the pointers `s` and `d`. There are just 16 stores to `dst` and 3 for the loop variables. In simd_loopunroll, on the other hand, the 16 bytes of data themselves go src → `q0` → stack → `q0` → stack → … → dst, written to the stack and read back 5 times along the way, so there are 43 stores, more than twice as many. The M1's Performance core (Firestorm) has 4 load/store units, but only 2 of them can handle stores [[2]](#ref-firestorm). So it can issue at most about 2 stores per cycle, and more stores alone means more cycles per loop iteration. On top of that, "write then immediately read back" is chained serially for each piece of data, so those latencies add up too. In short, the cost of the extra stores from going through the stack likely outweighed the benefit of widening each instruction from 8 to 16 bytes with SIMD. (This explanation is inferred from instruction counts. I haven't confirmed it with performance counters.)

Since the problem is that the registers aren't fully used, I'll specify which registers to use myself. For that I use **inline assembly**.

# 8. SIMD + inline assembly

```c
void *memcpy_simd_inline(void* dst, const void* src, size_t n){
    const uint8_t* s = (const uint8_t*) src;
    uint8_t* d = (uint8_t*) dst;

    while (n >= 128) {
        __asm__ volatile (
            "ldp q0, q1, [%[s]]\n\t"
            "ldp q2, q3, [%[s], #32]\n\t"
            "ldp q4, q5, [%[s], #64]\n\t"
            "ldp q6, q7, [%[s], #96]\n\t"
            "stp q0, q1, [%[d]]\n\t"
            "stp q2, q3, [%[d], #32]\n\t"
            "stp q4, q5, [%[d], #64]\n\t"
            "stp q6, q7, [%[d], #96]\n\t"
            :
            : [s] "r" (s), [d] "r" (d)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
        );

        s += 128;
        d += 128;
        n -= 128;
    }
    while (n >= 8){
        *(int64_t*)d = *(const int64_t*)s;
        n -= 8;
        d += 8;
        s += 8;
    }
    while (n--){
        *(d++) = *(s++);
    }
    return dst;
}
```

`ldp` (load pair) loads two `q` registers, 32 bytes, in one instruction, and `stp` (store pair) is its store counterpart. It loads 128 bytes into the 8 registers `q0` to `q7` and then writes them all out. `[s] "r" (s)` puts `s` in a register so it can be referred to as `%[s]`, and the last line tells the compiler that the code modifies `v0` to `v7` (the same registers as `q0` to `q7`) and memory.

A note: `ldp`/`stp` aren't SIMD-only instructions. They're ordinary AArch64 load/store instructions that also work with general-purpose registers, as in `ldp x0, x1, [x8]`. Here they target `q` registers (NEON's 128-bit registers), so each register carries 16 bytes. They don't do data-parallel computation, like `add v0.16b, ...` computing 16 elements at once. Since `memcpy` doesn't compute anything, the only benefit we get from SIMD here is moving more bytes per instruction. The `vld1q_u8`/`vst1q_u8` in section 6 also turn into `ldr q0`/`str q0` in the assembly, which are the same kind of instruction.

In the assembly, the instructions I wrote appear exactly as written.

```nasm
LBB3_2:                                 ;   in Loop: Header=BB3_1 Depth=1
	ldr	x8, [sp, #16]
	ldr	x9, [sp, #8]
	; InlineAsm Start
	ldp	q0, q1, [x8]
	ldp	q2, q3, [x8, #32]
	ldp	q4, q5, [x8, #64]
	ldp	q6, q7, [x8, #96]
	stp	q0, q1, [x9]
	stp	q2, q3, [x9, #32]
	stp	q4, q5, [x9, #64]
	stp	q6, q7, [x9, #96]

	; InlineAsm End
	ldr	x8, [sp, #16]
	add	x8, x8, #128
	str	x8, [sp, #16]
	...
```

The code around the copy is still `-O0` code, but the copy itself is 128 bytes in 8 instructions, with no detour through the stack. The data stores are down to just 4 `stp`s, from 43 in simd_loopunroll.

The result is **56.30 GB/s**. That's almost the same as the library `memcpy` (56.93 GB/s), and 82% of the theoretical value!

# Summary of results
Here are all the results side by side. The instruction count is the number of instructions in the main loop per iteration, counted from the assembly, and also converted to instructions per byte.

|Implementation|Bandwidth (GB/s)|% of theoretical|Bytes per iteration|Instructions per iteration|Instructions/byte|
|---|---|---|---|---|---|
|memcpy (library)|56.93|83%|-|-|-|
|naive|3.01|4.4%|1|14|14|
|8byte|15.34|22%|8|18|2.25|
|loopunroll|28.06|41%|128|78|0.61|
|simd|14.39|21%|16|26|1.63|
|simd_loopunroll|20.06|29%|128|110|0.86|
|simd_inlineasm|56.30|82%|128|24|0.19|

Roughly, the fewer instructions per byte, the faster it gets. But the simd version has fewer instructions per byte than the 8byte version and is still slower, so instruction count alone doesn't explain everything. The stack writes and reloads forming a serial chain, which creates waiting time, likely matter too. This is what isolating causes looks like: **predict with one metric, like instruction count, and dig further where the prediction misses**.

# Was `-O3` enough all along?
We've worked hard at `-O0` so far, but what happens if we compile the first naive implementation with `-O3`?

```bash
$ gcc -O3 -S -o - memcpy_naive.c
```

```nasm
LBB3_7:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x8, #-32]
	ldp	q2, q3, [x8], #64
	stp	q0, q1, [x9, #-32]
	stp	q2, q3, [x9], #64
	subs	x10, x10, #64
	b.ne	LBB3_7
```

The compiler rewrote the byte-by-byte loop into a loop using `q` registers and `ldp`/`stp`. It did the loop unrolling, the SIMD conversion and the register allocation all by itself. For something as simple as `memcpy`, `-O3` is enough for everything in this post.

Even so, real optimization work often doesn't go this smoothly. The compiler may give up on SIMD because it's worried about pointer aliasing, or it may not use registers the way you want. When that happens, the only way to find out why your code isn't getting faster is to read the assembly, as we did here. And as we saw when switching to SIMD alone made things slower, **you can't know which change helps until you measure**. So add changes one at a time, measure every time, and when the result differs from what you expected, read the assembly to isolate the cause.

# Summary
- The ceiling for `memcpy` is memory bandwidth. On the M1 it can be estimated as 68.25 GB/s from "transfer rate × bus width". The library `memcpy` reached 56.93 GB/s, 83% of that.
- I calculated bandwidth as $2N/t$ ($N$ bytes read + $N$ bytes written).
- Starting from a byte-by-byte copy (3.01 GB/s), reducing instructions per byte made it faster: 8-byte units (15.34 GB/s), then loop unrolling (28.06 GB/s).
- Just using SIMD intrinsics didn't help (14.39 GB/s), and even with unrolling only `q0` was used (20.06 GB/s).
- Using `q0` to `q7` and `ldp`/`stp` explicitly with inline assembly gave 56.30 GB/s, almost the same speed as the library `memcpy`.
- In practice `-O3` is sometimes enough, but you can't know what helps until you measure. Change one thing at a time, measure, and go down to the assembly when needed to isolate the cause.

# Reference

<span id="ref-m1-wiki"></span>[1] Wikipedia, "Apple M1", <https://en.wikipedia.org/wiki/Apple_M1>

<span id="ref-firestorm"></span>[2] Dougall Johnson, "Apple Microarchitecture Research: Firestorm", <https://dougallj.github.io/applecpu/firestorm.html>
