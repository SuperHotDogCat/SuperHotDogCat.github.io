---
title: "memcpyをゼロから実装してみる"
categories: [技術記事]
tags: []
math: true
description: "M1 Macでmemcpyを1つずつ高速化しながら, 計測とアセンブリで原因を切り分けます"
---

これは[High Performance Computingのシリーズ連載]({% post_url ja/2026-10-12-planning %})の一部です。

# このブログで話すこと
今回は`memcpy`を自分でゼロから実装してみます。1バイトずつコピーする一番素朴な実装から始めて, 少しずつ手を加えながら, 最終的にライブラリの`memcpy`と同じくらいの速度を出すところまで行きます。

ここで伝えたいのは「こう書けば速い」というテクニックそのものよりも, **1つずつ変更を加えて, 毎回計測して, 期待通りでなければアセンブリまで見に行く**という進め方です。実際のところ, 今回の話の大半は`-O3`をつければコンパイラがやってくれます(最後に確認します)。それでも, 最適化の現場ではコンパイラが期待通りのコードを出してくれずにアセンブリを読むことがありますし, 何が効くのかは計測するまでわかりません。だから変更は1つずつ入れて切り分けます。

使ったコードは全て[Materials](/assets/1122/memcpyimpl.zip)にあります。

# ハードウェアのスペック

|項目|値|
|---|---|
|Machine|MacBook Air (M1, 2020)|
|CPU|Apple M1 (Performance 4 cores + Efficiency 4 cores)|
|Memory|16 GB LPDDR4X-4266|
|Memory Bus|128 bit|
|Memory Bandwidth|68.25 GB/s(計算値, 後述)|
|Compiler|Apple clang 17.0.0 (`gcc`はclangのエイリアス)|

# 1. 理論速度と計測の準備
## 理論メモリ帯域
`memcpy`は計算をほとんどせず, メモリから読んでメモリに書くだけの関数です。なので上限を決めるのは[前々回の記事]({% post_url ja/2026-10-19-howtoseecatalogspec %})や[前回の記事]({% post_url ja/2026-10-31-hpcformulas %})で扱ったFLOPSではなく**メモリ帯域**です。

M1はLPDDR4X-4266のメモリを128 bit (16 byte) のバスでつないでいます [[1]](#ref-m1-wiki)。4266は1秒あたりの転送回数(MT/s)なので, 理論帯域は

$$
4266 \times 10^6\,[\text{transfer/s}] \times 16\,[\text{byte/transfer}] \approx 68.25\,\text{GB/s}
$$

です。前回の記事でピーク性能を「SIMD width × 命令数 × 周波数 × コア数」に分解したのと同じで, メモリ帯域も「転送レート × バス幅」に分解できます。今回はこの68.25 GB/sをゴールの上限とします。

## 帯域の計算式
今回は$N = 2^{30}$バイト (1 GiB) の配列をコピーします。`memcpy`は`src`から$N$バイト読み, `dst`に$N$バイト書くので, メモリとの間で合計$2N$バイトが動きます。1回のコピーにかかった時間を$t$とすると, 達成帯域は

$$
\text{Bandwidth} = \frac{2N}{t}
$$

です。厳密には, CPUは書き込み先のキャッシュラインを一度読み込んでから書く(write-allocate)ことが多いので, 実際のメモリトラフィックは$3N$になることもあります。ただ今回は「`memcpy`として何バイト動かしたか」を基準にして, ライブラリの`memcpy`とも同じ式で比べるので$2N$で計算します。

## 計測用のコード
時間計測には`timeutils.h`という小さなヘッダを使います。

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

- `timeEvent()`は`clock_gettime(CLOCK_MONOTONIC, ...)`で現在時刻を取ります。`CLOCK_MONOTONIC`は時刻合わせなどで巻き戻らない時計なので, 経過時間の計測に向いています。
- `elapsedTime()`は2つの時刻の差を秒で返します。
- `sum()`はコピー結果の検算用です。`src`を全て1で埋めておけば, 正しくコピーできていれば`dst`の総和は$N$になります。

そしてライブラリの`memcpy`を測るのが`memcpy_sample.c`です。

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

ポイントは`memset`で`src`と`dst`の両方を一度触っていることです。`malloc`しただけでは物理メモリは割り当てられていない可能性があるため(特にプロセスの初めの方), 最初に触ったときにページフォルトが起きて割り当てられます。これを計測区間の外で済ませておかないと, ページフォルトの時間まで帯域に含まれてしまいます。そのうえで10回コピーした平均を取ります。

また, 今回の実験では**アライメントは揃っているものとします**。1 GiBの`malloc`はページ境界に揃ったアドレスを返しますし, $N$は128の倍数なので, 後で出てくる「端数を処理するループ」は実際には一度も回りません。本物の`memcpy`は先頭や末尾がずれている場合も処理しますが, 今回はその部分は扱いません。

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

`make`で全ての実行ファイルと, `-S`をつけてアセンブリ(`.s`)を生成します。重要なのは**全て`-O0`(最適化なし)でコンパイルしている**ことです。`-O0`ではコンパイラは書いたCのコードをほぼそのまま機械語に翻訳します。今回は「Cのコードの書き方を変えたら何が起きるか」を1つずつ見たいので, コンパイラに勝手に最適化されると困るわけです。ただしライブラリの`memcpy`はすでに最適化済みのバイナリなので, `-O0`でも速いままです。実際, `memcpy_sample.s`を見ると`bl ___memcpy_chk`とライブラリ関数をそのまま呼んでいるだけです。

## 計測方法
最初の1回は遅く出ることが多かったので, 各プログラムを5回実行して**最大値**を採用します。例えば`memcpy_sample`は5回で

```
31.40 56.93 55.82 56.10 55.10 (GB/s)
```

でした。1回目だけ遅いのは, 実行直後はまだ周波数が上がりきっていないなどの要因が考えられます。

## ライブラリの`memcpy`
結果は**56.93 GB/s**で, 理論値68.25 GB/sの**83%**です。読み書きが混ざるとメモリの読み/書きの切り替えなどのオーバーヘッドがあるので, 100%に届かないのは自然です。これを今回の目標とします。

# 2. Naive実装
まずは誰でも書ける1バイトずつのコピーです。

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

`main`は`memcpy_sample.c`の`memcpy`を`memcpy_naive`に置き換えただけです(以降も同様です)。

結果は**3.01 GB/s**。ライブラリの`memcpy`の約1/19, 理論値の4.4%です。遅いですね。

なぜ遅いのか, `-O0`で出てきたアセンブリ(`memcpy_naive.s`)を見てみます。

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

Cのコードがほぼそのまま翻訳されているのがわかります。`n`, `s`, `d`はレジスタではなくスタック(`[sp, #24]`など)に置かれていて, 毎回スタックから読んで, 1を足して, スタックに書き戻しています。本当にやりたいのは`ldrb`(1バイト読む)と`strb`(1バイト書く)の2命令だけなのに, **1バイトのコピーに14命令**かかっています。これでは命令を処理する速さが律速になってしまい, メモリ帯域を使い切るどころではありません。

# 3. 8バイトずつコピーする
1命令で1バイトしか動かせないなら, 1命令で動かすバイト数を増やせば良さそうです。64 bitのCPUなので, 汎用レジスタ1本で8バイト運べます。

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

8バイト未満の端数は最後に1バイトずつコピーします(今回は$N$が8の倍数なので回りません)。

結果は**15.34 GB/s**で, Naiveの約5.1倍になりました。

# 4. アセンブリを見るとループの管理が重い
8倍の仕事をしているのに5倍しか速くなっていません。アセンブリ(`memcpy_8byte.s`)を見てみます。

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

1回のループは18命令で, そのうち実際のコピーは`ldr x8, [x8]`と`str x8, [x9]`の2命令だけです。残りの大半は, ループの条件判定と`n`, `d`, `s`の更新, つまり**ループを回すための管理**です。1回のループでやる仕事(8バイト)に対して, ループそのもののコストが大きすぎます。

それなら, 1回のループでたくさんコピーしてループの回数を減らせばよいはずです。これが**ループアンローリング**です。

# 5. ループアンローリング
1回のループで8バイトのコピーを16回, つまり128バイトを処理します。

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

結果は**28.06 GB/s**で, 8バイト版からさらに1.8倍になりました。ライブラリの`memcpy`の約半分です。

アセンブリを見ると, ループの中身は以下の4命令の繰り返しになっています。

```nasm
	ldr	x8, [sp, #16]
	ldr	x8, [x8]
	ldr	x9, [sp, #8]
	str	x8, [x9]
```

`-O0`なので相変わらず毎回`s`と`d`をスタックから読み直していますが, ループの判定と`n`, `d`, `s`の更新は128バイトに1回で済むようになりました。

余談ですが, このコードを書いたとき最初は`d + 72`に`s + 64`をコピーするタイプミスがありました。`src`を全て1で埋めているので, 間違った場所からコピーしても値は同じ1で, `sum`による検算は通ってしまいます。検算は「通ったから正しい」とは限らないので, 本当はランダムな値で埋めて`memcmp`で比較する方が安全です。

# 6. SIMDを使う
8バイトずつより大きな単位で運べないでしょうか。M1(ARMv8)には**NEON**というSIMD命令があり, 128 bit (16バイト) のレジスタ`q0`〜`q31`が32本あります。`arm_neon.h`のintrinsicsを使うと, 16バイトをまとめて読み書きできます。

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

`vld1q_u8`で16バイト読み, `vst1q_u8`で16バイト書きます。8バイト版と比べると1回のループで倍のバイト数を運べるので, 8バイト版(15.34 GB/s)の倍くらいは出てほしいところです。

結果は**14.39 GB/s**。思ったより速度が出ません。むしろ8バイト版より遅いです。アセンブリを見てみます。

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

`ldr q0, [x8]`で`src`から16バイト読んだ後, `str q0, [sp, #32]`→`ldr q0, [sp, #32]`のように**スタックに書いて読み直す**ことを3回繰り返してから, やっと`dst`に書いています。`-O0`では, intrinsicsの戻り値や変数`v0`もいちいちスタックに置かれるからです。結果として1ループ(16バイト)あたり26命令かかっており, しかもスタックへの書き込みと読み直しが直列につながっているので, 前の命令が終わるのを待たないと次に進めません。

# 7. SIMD + ループアンローリング
8バイト版でもループアンローリングが効いたので, SIMD版にも同じことをしてみます。1ループで16バイト×8 = 128バイトを処理します。

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

8本の読み込みを先にまとめて発行し, その後8本書き込むことで, 読み込みのレイテンシを重ねて隠せることを期待しています。

結果は**20.06 GB/s**。SIMD版よりは速くなりましたが, 8バイトのループアンローリング版(28.06 GB/s)にすら負けています。またアセンブリを見てみます。

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

関数全体で`q`レジスタが何回出てくるか数えると

```bash
$ grep -oE "q[0-9]+" memcpy_simd_loopunroll.s | sort | uniq -c
  80 q0
```

**`q0`しか使われていません**。NEONレジスタは32本もあるのに, `v0`〜`v7`は全てスタックに置かれ, `q0`1本を使い回して「読んでスタックに置く」を8回繰り返し, その後「スタックから読んで書く」を8回繰り返しています。期待していた「8本のレジスタに一気に読み込んでから一気に書く」にはなっておらず, 1ループ(128バイト)に110命令かかっています。

では, 同じ128バイトを運ぶ8バイトのループアンローリング版より遅いのはなぜでしょうか。ループ本体のロード命令(`ldr`)とストア命令(`str`)の数を比べてみます。

|実装|ldr|str|合計命令数|
|---|---|---|---|
|loopunroll|51|19|78|
|simd_loopunroll|59|43|110|

loopunrollでは, データは src → レジスタ → dst と直接流れていて, スタックから読んでいるのはポインタ`s`, `d`だけです。ストアは`dst`への16本とループ変数の更新3本しかありません。一方simd_loopunrollでは, 16バイトのデータそのものが src → `q0` → スタック → `q0` → スタック → … → dst と途中で5回スタックに書かれては読み直されていて, ストアは43本と2倍以上になっています。M1のPerformanceコア(Firestorm)はロード/ストアユニットを4つ持ちますが, そのうちストアを扱えるのは2つだけです [[2]](#ref-firestorm)。つまり1サイクルに発行できるストアは最大2本程度で, ストアが増えるとそれだけでループ1周にかかるサイクル数が増えます。さらに「書いてすぐ読む」が1つのデータについて直列に並ぶので, そのレイテンシも積み重なります。つまり, SIMDで1命令あたりの幅を8バイトから16バイトに広げた利点よりも, スタック経由で増えたストア命令のコストの方が大きかったと考えられます。(この説明は命令数からの推測で, パフォーマンスカウンタで確かめたわけではありません。)

レジスタがフルに活用されていないのが原因なので, どのレジスタを使うかを自分で指定します。そのために**インラインアセンブリ**を使います。

# 8. SIMD + インラインアセンブリ

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

`ldp`(load pair)は1命令で2本の`q`レジスタ, つまり32バイトを読み込む命令で, `stp`(store pair)はその書き込み版です。`q0`〜`q7`の8本に128バイトを読み込んでから, まとめて書き出します。`[s] "r" (s)`は`s`をレジスタに入れて`%[s]`として参照できるようにする指定で, 最後の行は`v0`〜`v7`(`q0`〜`q7`と同じレジスタ)とメモリを書き換えることをコンパイラに伝えています。

アセンブリを見ると, 書いた命令がそのまま入っています。

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

ループの外側は相変わらず`-O0`のコードですが, コピーそのものは8命令で128バイト, スタックを経由することもありません。データのストアは`stp`の4本だけで, simd_loopunrollの43本から大きく減りました。

結果は**56.30 GB/s**。ライブラリの`memcpy`(56.93 GB/s)とほぼ同じ, 理論値の82%に到達しました!

# 結果のまとめ
全ての結果を並べます。命令数はアセンブリの128バイト(あるいは1ループ)あたりのメインループの命令数を数えたもので, 1バイトあたりに直しています。

|実装|帯域 (GB/s)|理論値比|1ループで運ぶバイト数|1ループの命令数|命令/byte|
|---|---|---|---|---|---|
|memcpy (ライブラリ)|56.93|83%|-|-|-|
|naive|3.01|4.4%|1|14|14|
|8byte|15.34|22%|8|18|2.25|
|loopunroll|28.06|41%|128|78|0.61|
|simd|14.39|21%|16|26|1.63|
|simd_loopunroll|20.06|29%|128|110|0.86|
|simd_inlineasm|56.30|82%|128|24|0.19|

大まかには, 1バイトあたりの命令数が減るほど速くなっています。ただしsimd版は8byte版より命令/byteが少ないのに遅いので, 命令数だけでは説明できません。スタックを経由した書き込みと読み直しが直列につながっていて, 待ち時間が発生していることも効いていると考えられます。このように**命令数という1つの指標で予想して, 外れたところをさらに調べる**のが切り分けです。

# 実は`-O3`でよかった?
ここまで`-O0`で頑張ってきましたが, 最初のNaive実装を`-O3`でコンパイルするとどうなるでしょうか。

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

1バイトずつのループが, コンパイラによって`ldp`/`stp`を使ったSIMDのループに書き換えられています。ループアンローリング, SIMD化, レジスタ割り当てを全部やってくれているわけです。単純な`memcpy`程度なら, 今回の作業は`-O3`で事足ります。

それでも, 実際の最適化ではこうはいかないことがよくあります。コンパイラがポインタのエイリアスを気にしてSIMD化を諦めたり, 思ったレジスタの使い方をしてくれなかったりします。そういうときに「どうして速くならないのか」を知るには, 今回のようにアセンブリを見に行くしかありません。そして, 今回SIMDにしただけでは逆に遅くなったように, **どの変更が効くのかは計測するまでわかりません**。だから変更は1つずつ入れて, 毎回計測し, 期待と違えばアセンブリを見て原因を切り分けます。

# まとめ
- `memcpy`の上限はメモリ帯域で決まり, M1では「転送レート × バス幅」から68.25 GB/sと見積もれます。ライブラリの`memcpy`はその83%の56.93 GB/sでした。
- 帯域は$2N/t$(読み$N$バイト + 書き$N$バイト)で計算しました。
- 1バイトずつのコピー(3.01 GB/s)から, 8バイト単位(15.34 GB/s), ループアンローリング(28.06 GB/s)と, 1バイトあたりの命令数を減らすことで速くなりました。
- SIMDのintrinsicsを使っただけでは速くならず(14.39 GB/s), アンローリングしても`q0`しか使われていませんでした(20.06 GB/s)。
- インラインアセンブリで`q0`〜`q7`と`ldp`/`stp`を明示的に使うと56.30 GB/sとなり, ライブラリの`memcpy`とほぼ同じ速度が出ました。
- 実際には`-O3`で十分なこともありますが, 何が効くかは計測するまでわからないので, 1つずつ変更して計測し, 必要ならアセンブリまで見て切り分けましょう。

# Reference

<span id="ref-m1-wiki"></span>[1] Wikipedia, "Apple M1", <https://en.wikipedia.org/wiki/Apple_M1>

<span id="ref-firestorm"></span>[2] Dougall Johnson, "Apple Microarchitecture Research: Firestorm", <https://dougallj.github.io/applecpu/firestorm.html>
