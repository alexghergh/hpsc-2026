# FINAL REPORT - 13 Tensor Core Matmul — run summary

![Speed-up matplotlib plot](./trajectory.png)

First end-to-end run of the cuda_source problem source. Problem: dense FP32 GEMM
`C = A·B`, column-major, `m=10240, k=4096, n=8192`, reference `cublasGemmEx`
with `CUBLAS_COMPUTE_32F_FAST_16F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`. Tolerance:
mean-abs-error < 1.0e-2. The agent ran the harness on one H100 (Hinadori)
system; the final kernel was then re-evaluated independently on a second H100
(Tsubame) - the test host - and that is the result that counts. The agent's only
handle on the problem was the harness MCP server (`workspace_overview`,
`read_workspace_file`, `write_candidate`, `run_candidate`, `profile_ncu`,
`goal_status`, `best_result`, `complete_problem`). Profiler was wired but
produced cuBLAS's stats on every call due to a harness bug (fixed in a later
commit), so the agent drove the entire 56-turn search off end-to-end
`run_candidate` timings — measured on the harness host, not the test host.

**Result on the test host: did NOT beat cuBLAS.** Final candidate 2.085 ms /
329,718 GFLOPS vs cuBLAS 1.926 ms / 356,963 GFLOPS — cuBLAS +8.2%, candidate at
~33% of H100's 989 TFLOPS dense FP16/FP32-accum peak. On the harness host the
same kernel beats cuBLAS by 7.3% (2.443 ms / 281,330 GFLOPS vs 2.623 ms /
262,089 GFLOPS), so the agent's internal feedback loop reported a win. The
margin flips because the test host's cuBLAS dispatches a `wgmma.mma_async`-based
Hopper kernel (sm_90a, warp-group MMA), while the harness host's cuBLAS
dispatches an Ampere-style `mma.sync.m16n8k16` algorithm — the same instruction
the candidate uses. On the harness host both code paths share an instruction
ceiling and the candidate's tiling/swizzle wins; on the test host the candidate
is locked into the slower ISA. The decisive optimization moves were splitting
the fp32→fp16 conversion into a dedicated kernel so the matmul hot loop only
reads fp16, pairing that with explicit `ldmatrix.x4.trans` +
`mma.sync.aligned.m16n8k16` PTX (which beats the `nvcuda::wmma` C++ API),
`BM=128 BN=256 BK=64` tiling, padded conflict-free shared (`APAD=BPAD=8`), and a
`GROUP=8` threadblock swizzle for L2 B-panel reuse. The agent stayed entirely
within Ampere-era intrinsics — it never considered TMA, `wgmma`, thread-block
clusters, DSMEM, `mbarrier`, or stream-K. That headroom is now the entire story:
the next agent run has to reach `wgmma`-class instruction throughput to clear
the test-host cuBLAS, not just out-tile an mma.sync algorithm.

## Run

| | |
|---|---|
| Model | `claude-opus-4-8` (Claude Code) |
| Reasoning effort | `max` (Claude `effortLevel: "max"`) |
| Wall-clock | 82.9 min |
| Cost | $18.10 |
| Turns completed | 56 |
| Input / output tokens | 12.94M / 358k |
| Subagent spawns | 27 |
| `run_candidate` calls | 22 (all correct, zero cheating flags) |
| `profile_ncu` calls | 3 (all returned cuBLAS's stats — harness bug, fixed) |
| `web_search` calls | 0 |
| Harness-host terminal state | `done` / `measured_outcome: beats_baseline` |
| Test-host outcome | cuBLAS +8.2%, candidate did not beat baseline |

## Trajectory (harness host)

All timings below are from the agent's `run_candidate` calls on the Hinadori
host, against that host's cuBLAS (which dispatches the mma.sync code path). The
final harness-host runtime (2.44 ms) does **not** carry over to the test host:
there the same kernel runs in 2.085 ms, but the test-host cuBLAS runs in 1.926
ms.

| id | runtime | GF/s | speedup vs harness cuBLAS | move |
|---:|--------:|------:|--------:|------|
| 0 | 27.7 ms | 24.8k | 0.10× | starter (your `13_tensorcore.cu` WMMA scaffold) |
| 2 | 7.52 ms | 91.4k | 0.35× | **WMMA tiled + coalesced loads** ✓ |
| 4 | 3.73 ms | 184k | 0.70× | **vectorized float4 / half2 loads** ✓ |
| 6 | 3.56 ms | 193k | 0.74× | **explicit `ldmatrix` + `mma.sync.m16n8k16` PTX** ✓ |
| 8 | 89.7 ms | 7.7k | 0.03× | 256×256 tiles — register spill ✗ |
| 11 | 3.37 ms | 204k | 0.78× | `cp.async` deep pipeline — plateau, never recovered ✗ |
| 15 | 3.28 ms | 209k | 0.80× | **padded shared + GROUP=8 swizzle** ✓ |
| 17 | 2.79 ms | 247k | 0.95× | **dedicated `convert_kernel` (fp32→fp16 split)** ✓ ★ |
| **18** | **2.45 ms** | **281k** | **1.07×** | **+ BK=64 (lighter prefetch path) ★ FIRST harness-host BEAT** |
| 21 | 2.44 ms | 281k | 1.07× | confirmation |

What didn't work (measured + rejected on the harness host): TF32 path
(catastrophic shared bank conflicts), `cp.async` pipelines (both
conversion-aware and deep fp16 versions), 256×256 tiles (register spill → 89.7
ms regression), higher occupancy / smaller warp tiles, BK=16, convert-at-load,
interleaved store.

## Cross-host comparison

| host | role | cuBLAS GFLOPS | cuBLAS ms | candidate GFLOPS | candidate ms | margin |
|---|---|---:|---:|---:|---:|---:|
| Hinadori H100 | harness | 262,089 | 2.623 | 281,329 | 2.443 | candidate +7.3% |
| Tsubame H100 | **test** | 356,962 | 1.926 | 329,717 | 2.085 | **cuBLAS +8.2%** |

Candidate gained 17% going harness → test (2.443 → 2.085 ms), consistent with
clock / memory-bandwidth headroom between hosts. cuBLAS gained 36% (2.623 →
1.926 ms), which is too large to be clock scaling alone — that is an algorithm
change. To confirm: `nsys profile ./best_kernel` on both and check the cuBLAS
kernel name (`sm90_xmma_gemm_*` / `cutlass3_sm90_*` on the test host vs
`ampere_*_gemm_*` / `cutlass_80_*` on the harness host). Likely upstream cause
is a CUDA / cuBLAS version delta (12.0 vs 12.3+); clock locking, power caps, and
SKU differences (SXM5 / NVL / PCIe) are secondary.

## Honest caveats

The harness-host beat is on the specific shape `(10240, 8192, 4096)` and the
specific cuBLAS the harness host happens to dispatch (DEFAULT algo, no
`cublasLt` tuning, mma.sync code path). The test host demonstrates exactly the
pre-existing concern: a different cuBLAS config / version on the same hardware
family yields 5–25%+ faster cuBLAS, which shrinks or inverts the candidate's
margin. The kernel only handles shapes where `m % 128 == 0, n % 256 == 0, k % 64
== 0`. It always computes `α=1, β=0` (the reference's actual call); it is not a
general GEMM. All numbers are reproducible: see `attempts/sample_21.json` for
the harness-host trace and the standalone `best_kernel.cu` for the test-host
benchmark.
