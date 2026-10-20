# `broadwell-pcie3`: GLM-5.3-Flash on one RTX 3090 + an old dual-Xeon

This branch is upstream [glm53-flash-offload](https://github.com/sybil-solutions/glm-flash-lite) at commit
`1966a4f` (2026-10-01, before the project was renamed glm-flash-lite) plus patches for older hardware: two Xeon
E5-2699 v4 (Broadwell, AVX2, no AVX-512), 512 GB DDR4-2400, one RTX 3090 on PCIe 3.0 x16. It is the exact code behind
the field note [How does a 320B model run on one RTX 3090?](https://nathankeilar.com/systems/glm53-flash-320b-moe-one-rtx-3090),
which explains the design, the measurements and their limits. It is **not** rebased onto current glm-flash-lite.

On that box: `exact` mode 4.13 tok/s, upstream `fast` settings ~13.1-13.3 tok/s, this branch with the profile below
**15.2 tok/s** decode (single stream, short prose prompt, greedy, thinking off, 131k context configured), and **~16.6 tok/s**
with the N32CTRL decode flags and `GLM53_CT_TZC=1.3` (2026-10-16, 60 s greedy windows; 16.6 on the 512-token essay
benchmark used for E1). With the opt-in **E1 lookup-table
CPU tier** described below: **29 tok/s** at one user and 30 total across four users (2026-10-20, two configuration
fixes on top of E1). Your numbers will differ with your CPU, RAM channels and
PCIe generation.

Every switch added here defaults to off (upstream behaviour). In the table, the bold value is the one in the measured profile.

## Switches added by this branch

| env | values | what it does | measured on the test box |
|---|---|---|---|
| `GLM53_CT_SPLIT1` | 0 · 1 · **2** | CPU tier work units: 1 = gate and up per 128-column block as separate units; 2 = also 64-column halves | +2.8% decode, CPU job −4.5% (bit-identical to stock) |
| `GLM53_CT_P0BLK` | 0 · **1** | block-parallel input prep (needs `SPLIT1=2`, I16 path) | CPU job −2.5% (bit-identical) |
| `GLM53_CT_SOCKQ` | 0 · **1** | per-socket work counters with stealing instead of one shared atomic | ≈ 0 (kept, harmless) |
| `GLM53_CT_CFLAGS` | string | compiler flags for the CPU-tier JIT build; `-DFT_OPT_BSUM_SHUF=1 -DFT_OPT_FLOAT_EXTRACT=1` enable Broadwell decode tweaks | CPU time −3% (bit-identical) |
| `GLM53_CT_CFLAGS` += `-DFT_OPT_EXP_EXTRACT=1 -DFT_OPT_BSUM_MADD_N=32 -DFT_OPT_CTRL_REUSE=1` ("N32CTRL") | **on** in the profile | cheaper trellis decode: exponent-field bit extraction instead of convert-multiply-convert, `vpmaddubsw` byte sums on all 32 groups, shuffle masks kept in registers. Optional, also off: `GLM53_CT_LIGHT_STATIC=1`, `GLM53_CT_SERIAL_SUM=1` (static short phases, serial reduction) | decode loop 188 → ~160 cycles per 256 weights (uiCA, Broadwell); harness on the test box: 7.4-13.9% less time per decode call, projected +3.5% tok/s. **Live on the test box (8 restarts per arm):** CPU job −6.4% (0.182 + 0.255·K → 0.160 + 0.242·K ms), quality-panel output identical with the I16 path forced; tok/s +0.4% with the split cost model unchanged, **+7.9% (15.42 → 16.63, ten alone) together with `GLM53_CT_TZC=1.3`**, which lets the faster CPU take more of each layer's misses. Output bit-identical in the harness (server flags and `-ffp-contract=off`, both layouts); may set the MXCSR inexact status flag (exception stays masked) |
| `GLM53_CT_BUILD` | path | separate JIT build dir per flag set, so variants don't overwrite each other | |
| `GLM53_ZC_THP` | 0 · **1** | pinned expert arena backed by 2 MB pages (mmap + `MADV_HUGEPAGE` before first touch) | prefill +4-5%, decode tie; needs THP `madvise` or `always` |
| `GLM53_LA` | 0 · **1** | decode lookahead (ported from upstream `1f6d74e`); runtime `GET /la?on=0\|1` | +3.3%; greedy-identical in `exact` mode |
| `GLM53_CT_SWIGLU_CLAMP` | 0 · **1** | apply the model's `swiglu_limit` (10) in the CPU path, as the GPU path does | correctness: CPU-path panel top-1 0.9842 → 0.9865 |
| `GLM53_EC_BG` | 0 · **1** | background admission into the VRAM expert cache on a copy stream (`GLM53_EC_BG_*` tune it) | flat on one GPU; +10% with the remote tier |
| `GLM53_CT_FUSE`, `GLM53_CT_FUSE_MAXN`, `GLM53_CT_PF_LEAN` | 0 · **1** | CPU job overhead trims (1-expert fused jobs, lean prefetch) | CPU job −1.5%, < 1% tok/s |
| `GLM53_CT_LUT` | "" or 8 floats | measured CPU cost table for the split (off) | ≤ 1%; left off |
| `GLM53_K_OVL_AFTER` | **0** · 1 | shared expert after the routed path | within noise; off |
| `GLM53_TRACE_ROUTES` | path | append one JSON line per decode MoE step (routing picks, CPU picks, cache hits) | analysis only |
| `GLM53_PROF_DIR` | path (default `traces`) | where `GET /prof?seconds=4&tag=x` writes a torch.profiler chrome trace | analysis only |
| `GLM53_RT`, `GLM53_RT_*`, `GLM53_CT_E4` | | client for an experimental remote expert tier (a second machine's GPUs over TCP). **The server is not in this branch**; leave unset | |

Runtime endpoints added: `GET /ct_policy` (cost-model knobs without a restart), `GET /ec_verify` (exact consistency
check of the expert cache tables), `GET /ct_trace`, `GET /prof`, `GET /la`. Comment tags such as `[bdw-opt]`, `[E3]`,
and `[E4]` name the experiment each change came from.

## E1: lookup-table CPU expert tier (opt-in)

The default CPU tier decodes each cold expert from the 3.05-bit EXL3 trellis copy. On Broadwell that decode, not RAM,
is the bottleneck: it takes longer than reading the bytes. E1 replaces only the CPU's copy of the routed experts.
- **Format:** IQ3_K gate/up and IQ4_KS down, quantised from Unsloth's Q8_0 GGUF with a per-expert importance matrix;
  ~3.7 bits per weight.
- **Decode:** the experts decode with small lookup tables.
- **Layout:** half of every expert sits on each socket, with one worker pool per NUMA node, so both memory systems
  stream at once.

The GPU path, the home copy the GPU reads, attention and the expert cache are unchanged. The trellis path compiles
identically when `GLM53_CT_FORMAT` is unset.

### Measured on the test box (2026-10-19)

Decode, greedy, thinking off, 512-token essays; total tok/s with per-user in brackets:

| Users at once | trellis (profile below) | E1, 8-bit activations, batch cap 32 |
|---|---|---|
| 1 | 16.6 (median of 3) | **25.8** (median; 25.6-26.3) |
| 2 | 17.0 (8.5) | 22.4 (11.2) |
| 4 | 17.1 (4.3) | **18.6 (4.7)** |

- **Prefill, 8.5k prompt:** ~410 tok/s trellis, 442-446 E1.

**Assembly-level pass (2026-10-20).** A same-day A/B of the E1 build before and after the assembly pass described
below. Same scripts, both measured that day:

| Users at once | E1 before | E1 after (current code) |
|---|---|---|
| 1 | 25.3 (median of 3) | 25.3 (median of 5) |
| 2 | 21.6 | 22.0 / 21.5 |
| 4 | 18.3 | **21.85 / 21.83 (+19.6%)** |

- **Prefill:** 435-441 before vs 420-434 after, which is within E1's day-to-day spread. Prefill doesn't use the
  changed code.
- **Day-to-day drift:** the 1-user medians differ from the 2026-10-19 table above (25.3 vs 25.8) between days, on the
  unchanged build too.

**Two configuration fixes (2026-10-20, current).** The same E1 build, plus the server's main thread pinned to a free
core and `GLM53_CT_TZC=3.0` (both explained below). Measured the same day after a restart, with the same scripts:

| Users at once | E1 build as above | **+ pinned main thread + `GLM53_CT_TZC=3.0`** |
|---|---|---|
| 1 | 25.3 | **28.2 / 29.3 / 29.9 / 28.6** (median 29.0) |
| 2 | 22.0 / 21.5 | **27.5 / 28.0** |
| 4 | 21.85 / 21.83 | **30.1 / 30.4** |

- **Prefill:** 438-445.
- **Current-run checks:** 0 CPU-job timeouts and 0 failures.
- **Earlier soak (2026-10-18, before these fixes):** chats of up to 97.5k tokens with tool calls passed at ~21 tok/s.
- **Earlier error check (2026-10-19, before these fixes):** 0 CPU-job timeouts and 0 failures over 111k calls.
- **Earlier accuracy replays (unchanged by the 2026-10-20 fixes, which are bit-exact or scheduling-only):** per-expert output error against the Q8_0 model, on real decode activations:

| Layer | trellis (median / weighted) | E1 (median / weighted) |
|---|---|---|
| 3 (ordinary recipe) | 0.0275 / 0.0271 | 0.0248 / 0.0240 (16-bit), 0.0253 / 0.0245 (8-bit) |
| 11 (upgraded recipe) | 0.0390 / 0.0377 | 0.0082 / 0.0080 (16-bit), 0.0085 / 0.0083 (8-bit) |

- A teacher-forced full-model panel against the trellis model's own answers scores top-1 0.942, KL 0.037. That panel's
  reference was produced by trellis, so it can't rank the two formats; it only guards against breakage.
- **Limits:**
  - Only 2 of 42 layers have the Q8_0 comparison.
  - The 8-bit numbers ran on a newer host kernel than the 16-bit ones.
  - Everything was tuned on one Broadwell box; timings are measured there and will differ elsewhere.

### Running it

E1 needs four artifacts, and startup refuses without each one and its SHA-256 (no silent fallback):
- the 42-layer bank and its `e1-lut-bank/v1` manifest;
- a REAL-PASS quality report from replaying real activations;
- a cost table measured on your own machine;
- a receipt for the precompiled LUT modules. LUT serving never JIT-compiles.

`tools/e1/README.md` is the end-to-end runbook to build them from Unsloth's public Q8_0 release; the bank is
byte-reproducible and checkable against the published per-layer SHAs. The serving profile differs from the trellis
profile below only in the CPU tier:

```bash
# base (as below): GLM53_MODE=fast GLM53_CPU_TIER=1 GLM53_EC=1 GLM53_K_FTSPLIT=1 GLM53_K_HCFUSE=1 GLM53_K_OVL=1 GLM53_LA=1 ...
export GLM53_CT_FORMAT=lut GLM53_CT_LUT_ACT=q8 \
  GLM53_CT_LUT_MANIFEST=/path/e1-lut-bank.json \
  GLM53_CT_LUT_QUALITY=/path/real-i16-bank.json   GLM53_CT_LUT_QUALITY_SHA256=<sha256> \
  GLM53_CT_LUT_Q8_QUALITY=/path/real-q8-bank.json GLM53_CT_LUT_Q8_QUALITY_SHA256=<sha256> \
  GLM53_CT_LUT_COST=/path/lut-cost-q8.json        GLM53_CT_LUT_COST_SHA256=<sha256> \
  GLM53_CT_LUT_PRECOMPILE=/path/lut/reviewed-precompile.json GLM53_CT_LUT_PRECOMPILE_SHA256=<sha256> \
  GLM53_CT_LUT_BATCH_POLICY=bounded GLM53_CT_MAXN=32 \
  GLM53_ZC_HOME_NODE=0 GLM53_ZC_NODE1_ARENAS=10 \
  GLM53_CT_CPUS=2-43 GLM53_CT_THREADS=0 GLM53_CT_COORD_CPU=0 GLM53_CT_MAXBSZ=4 GLM53_CT_SWIGLU_CLAMP=1 \
  GLM53_CT_B=0.31 GLM53_CT_TZC=3.0 GLM53_CT_FORCE_N=-1   # TZC 3.0, not 1.3, with the bounded policy: see "Two configuration fixes"
# not used with E1: GLM53_CT_SWZ, GLM53_CT_SPLIT1, GLM53_CT_SOCKQ, GLM53_CT_P0BLK, GLM53_CT_CFLAGS (trellis-only)
```

RAM: the pinned home copy (~108 GB) plus the bank (134 GiB, half per socket); each NUMA node needs ~80 GB free when the
bank loads. Load takes ~8 min.

Once `/health` answers, pin the server's main thread to a core outside `GLM53_CT_CPUS` and the coordinator CPU. On the
test box that's core 1 (CPUs 1 and 45). The main thread queues every GPU kernel:

```bash
taskset -p -c 1,45 <server pid>     # the main thread's TID is the PID; threads created earlier keep their affinity
```

### E1 switches

| env | values | what it does |
|---|---|---|
| `GLM53_CT_FORMAT` | (unset) · **`lut`** | `lut` loads the bank instead of the trellis CPU copy |
| `GLM53_CT_LUT_ACT` | `i16` · **`q8`** | activations: exact two-part 16-bit dot, or 8-bit (15% cheaper per 3-expert call; needs a bank-bound Q8 report whose binding equals the I16 one) |
| `GLM53_CT_LUT_MANIFEST`, `_QUALITY`, `_Q8_QUALITY`, `_COST`, `_PRECOMPILE` (+ `_SHA256`) | path + sha | the four admission artifacts; any mismatch refuses startup |
| `GLM53_CT_LUT_BATCH_POLICY` | `shared` · **`bounded`** | how many distinct experts per *batch* the split may give the CPU (next section) |
| `GLM53_CT_MAXN` | ≤ 8 (`shared`) · **32** (`bounded`) | that per-batch cap; runtime `GET /ct_policy?maxn=N` |
| `GLM53_ZC_HOME_NODE` | -1 · **0** | pin the GPU's pinned home copy to the GPU's NUMA node (first touch otherwise follows the loader thread) |
| `GLM53_ZC_NODE1_ARENAS` | **10** | the first N layers' home arenas prefer node 1, leaving room on node 0 for its half of the bank |
| `GLM53_TF_DECODE` | 0 · 1 | `/generate` teacher-forced logprobs through decode steps (quality panels that must exercise the CPU tier) |

### The batch policy (why multi-user needs `bounded`)

The split counts K, the distinct CPU-assigned experts across the **whole batch**.
- **`shared` caps K at 8 for the batch.** With 4 users there are up to 32 cold misses per layer, so ~24 of them go over
  PCIe at ~1.3 ms each. The GPU lane becomes the bottleneck while the CPU idles. Measured: 4 users 6.6 tok/s total,
  where trellis, uncapped, did 17.1.
- **`bounded`** lets the cap be `min(MAXN, 8*M)` for M tokens per step. M=1 keeps its measured rows and a cap ≤ 8.
- **Costs beyond K=8** are extrapolated per recipe and M as `C_M(8) + B_M*(K-8)`, with `B_M = max_k C_M(k)/k`. That's a
  conservative estimate, not a measured cost. The live result at 4 users was 6.6 → 17.6 tok/s total with 16-bit
activations (18.6 with 8-bit), with 1 user unchanged.

### What made E1 fast (reusable lessons)

Ordered by contribution. Unless marked, numbers are measured on the test box: ms are per CPU job at K = 3 experts and
1 token, tok/s are live decode. Only three changes have an isolated live tok/s number: the format switch (as a bundle
with everything else E1 needed), the batch cap and 8-bit activations.

| # | Change | Hardware constraint it works around | Measured effect | Reusable for |
|---|---|---|---|---|
| 1 | **Table-lookup expert formats** (IQ3_K/IQ4_KS) instead of decoding the EXL3 trellis on the CPU | On AVX2 the trellis decode is compute-bound (perf: ~0.84 cycles/weight against a 0.56 port bound), so a CPU job moved only ~30 GB/s | Screen: 0.880 → 0.537 ms (16-bit), 0.428 ms (8-bit). Whole server: 16.7 → 21.7 tok/s in the first A/B; 16.6 → 25.8 today | Any AVX2-only CPU expert/GEMV path |
| 2 | **Split every expert across sockets** (TP=2: each node owns half the intermediate rows), one worker pool per NUMA node, memory bound with `MPOL_PREFERRED` | QPI: on this box, one pool spanning both sockets read 49 GB/s vs 95 GB/s for per-socket pools | 0.520 ms vs 0.574 interleaved vs 0.784 on one node (16-bit). Pitfall: an off-by-one in `maxnode` silently made "preferred" act as "local"; check placement with `get_mempolicy`/sampled pages | Any 2-socket host |
| 3 | **Scale the CPU's per-batch expert cap with the batch** (`min(MAXN, 8·M)`, cap 32) | PCIe 3.0: each cold expert the cap pushes to the GPU costs ~1.3 ms of zero-copy reads (estimate), so a cap of 8 per batch starved the CPU with several users | 4 users 6.6 → 17.6 tok/s total, 2 users 15.4 → 22.7 (16-bit) | Any hybrid split that caps CPU work per batch rather than per token |
| 4 | **Spin ~50 ms before sleeping** (was 2 ms), then futex | Napping workers wake late (deep C-states are the likely cause; not measured), and the slowest of 42 workers gates the barrier | 1.10-1.30 ms napping vs 0.81 ms never napping | Any fork-join pool called every few ms |
| 5 | **Software prefetch 16 rows ahead** (gate/up) and 48 (down) | Intel's L2 streamer stops at every 4 KiB page, so cold streams re-ramp | 0.880 → 0.745 ms (−15%); 24/72 rows gave 0.739 | Any cold streaming GEMV on Intel |
| 6 | **Expert-major contiguous rows** | Long sequential streams for the prefetchers | 1.677 → 1.162 ms (integrated harness) | Any weight layout read by experts |
| 7 | **8-bit activations** (Q8_K) | Fewer multiply/widen uops per weight | −15% at 3 experts, −33% at 8; live +2-4% at 1 user, 17.6 → 18.6 at 4 users. Accuracy still beats trellis on the measured layers | General |
| 8 | **Check CPU-tier failures once per token** (in the sampler) instead of a host sync after each of 42 layers | Per-layer syncs stalled the GPU and blocked lookahead | Not measured alone; it is what let lookahead (+3.3% on trellis), HCFUSE and the fast split run with E1 | Any GPU-CPU hybrid that syncs per layer |
| 9 | **Pin the GPU's pinned host copy to the GPU's node**, moving a few layers' copies to node 1 | First touch follows the loader thread: all 114 GB had landed on node 1, so every zero-copy read crossed QPI and node 1 had no room for its half of the bank | Enabling: E1 could start at all. Not measured alone | Any pinned zero-copy arena on a multi-socket host |
| 10 | **Per-expert imatrix normalisation** when quantising from Unsloth's GGUF imatrix | ik's converter uses one count for all 288 experts | Accuracy. Whole recipe vs Unsloth's own quant: layer 3 0.0248 vs 0.0290, layer 11 0.0082 vs 0.0097 | Any MoE requantisation from GGUF-v3 imatrices |

**Measurement lessons that changed conclusions:**
- **Flush caches on the node that owns the memory.** A benchmark fixture where a node-0 thread flushed node 1's lines
  made node-1 workers 1.6× slower: 0.744 → 0.462 ms once fixed, with prefetch held constant. Together with row prefetch,
  it took E1's cost gate from a fail (1.193 ms) to a pass (0.587 ms).
- **Measure at the real call cadence, with enough samples.** Calls were spaced 1.5 ms apart like decode. 31 samples per
  cell, because 7 samples gave non-monotonic curves (K=5 cheaper than K=4).
- **Idle neighbours.** An idle trellis server's polling threads (~4 cores) contaminated CPU benchmarks on shared cores.
- **Rank quantisations against a higher-precision reference.** A quality panel generated by one of the candidates can't
  rank them; per-layer error against Q8_0 can.

**Ruled out, with evidence:**
- **Unsloth's IQ3_S/IQ4_XS as-is:** 0.704/0.856 ms. IQ3_S costs 175.9 cycles per 256 weights vs 44.4 for IQ3_K.
- **Futex wake alone:** no speed change (1.155 vs 1.182 ms), but idle CPU 0.24 → 0.00 cores.
- **2 MB huge pages:** ≤ ~1% DTLB walks, and decode tied. Parked.
- **2-bit cold experts (IQ2_K):** 2.7-3.4× the error for 4-15% fewer bytes. Fails the quality gate.
- **IQ3_KS gate/up and Hadamard rotation:** fail the quality gate (layer 3 0.028-0.031 and 0.042 vs trellis 0.0275).
- **Cross-token weight reuse:** decode is one token per sequence, and the experts of different users barely overlap.
- **Row-interleaved repack and a fused per-expert kernel:** no gain while the batch cap or the formats limited.

### Assembly-level pass (2026-10-20)

The production `ft_tier_host.so` was disassembled and audited at the instruction level, using Ghidra, the uiCA
Broadwell port model and timing on one spare core. The same GCC 13.3 reproduced it instruction for instruction, so
every finding was checked against the real binary. Every change below is bit-exact against the previous binary, gated
on 40,000 activation vectors, 512,000 dot products, millions of raw bit patterns and real-weight output hashes.

| Finding | Fix | Effect |
|---|---|---|
| A finiteness check called a non-inlined function **per element**, and some checks ran twice | one vector compare per call, each check once | part of halving the fixed per-call path (57 → 27 µs at 42 cores) |
| fp16 → fp32 input converted one element at a time | F16C, 8 at a time | same |
| The Q8 activation quantiser stayed scalar: an `assert` (no `-DNDEBUG` in extension builds) blocked vectorisation | hand-written AVX2 version | 2295 → 279 cycles per 256 values |
| Barrier arrival counter and release word shared a cache line with 20+ spinning threads | separate cache lines (ordering unchanged) | small |
| IQ3 inner loop: two register spills per iteration plus stack-reloaded constants, which made it vulnerable to **4K aliasing** (at ~2% of stack layouts it ran 30-39% slower) | hand-written spill-free Broadwell assembly with **no stack access**, a 6-shift index unpack and a one-broadcast header | −11% / −20% / −21% cycles at 1/2/4 tokens per call |
| IQ4 path: scalar per-group table select and duplicated loads | the optimised path for M=1, broadcast + blend nibble split | −5 to −10% |
| Built with `-mtune=znver3` (inherited) on Broadwell; 3 PLT hops per row from default visibility | `-mtune=broadwell -fno-semantic-interposition -flto` | −3.5% kernel |

**Lessons:**
- **Halving the CPU job's fixed cost did not move 1-user tok/s, but the CPU is still on the 1-user critical path.**
  - Before the configuration fixes, the GPU waited 15.3 ms of a 39.1 ms token for the CPU's result.
  - A typical 1-user job used about 4 cold experts and mostly streamed weights from RAM. Some larger cost-table cells
    measured slower in this build, amid substantial repeat noise.
  - My explanation for the +19.6% at 4 users is that cheaper cost-table cells shifted more work to the CPU (next
    section). I did not measure the policy's contribution separately.
- **Never add `-DNDEBUG` or `-ffp-contract` to get vectorisation.** GCC 13 contracts `a*b+c` into FMA in C++ even with
  `-std=c++17`, and that changed 6,416 of 40,000 quantised vectors in testing.
- **A test whose reference calls the same kernels can't catch kernel changes.** Use a differential harness against the
  previous binary.
- **Single cost-table cells vary ±15-30% between repeats at K≥3.** Judge builds by live A/B, not by a cell.
- **Ruled out on this CPU:** activation-side lookup tables (T-MAC style) and precomputed codebook-product tables, 8-13×
  slower than `vpmaddubsw` on Broadwell's single shuffle port; and a transposed 3-bit trellis, whose port-throughput
  floor is above the break-even point in every case.

### Two configuration fixes (2026-10-20)
Both came from reading the combine kernel's counters (`/stats`: `gpu_wait_ms_per_job` = GPU time spent waiting for the
CPU's partial result; `gpu_late` = layers where the CPU was already done) instead of guessing.

**1. Give the GPU's launch thread its own core.**
- **The problem:** the server's main thread queues every GPU kernel, but its affinity covered all 88 CPUs. I found it
  on CPU 22, alongside a spinning CPU-tier worker.
- **The test:** I ran unpinned, pinned, unpinned, then pinned, with two 1-user runs in each round:

  | Round | Main thread | tok/s |
  |---|---|---|
  | 1 | unpinned | 25.47 / 25.38 |
  | 2 | pinned | 28.27 / 28.73 |
  | 3 | unpinned | 24.58 / 23.07 |
  | 4 | pinned | 28.81 / 28.68 |

  The pinned runs stayed between 28.27 and 28.81 tok/s in this test.

**2. Don't price the CPU as if users share experts.**
- **The measurement:** four single-user decodes with different prompts were traced and overlaid layer by layer, as if
  batched.
  - 96.2% of the cold experts four users need are needed by only one of them; sharing saves 4% of the weight bytes.
  - The policy's M=2/4 cost rows were measured with all tokens using the same experts, which is far costlier per
    expert. So the policy over-priced the CPU.
- **The effect:** at 4 users it shipped ~2.3 experts per layer over PCIe (the policy prices each at 1.3 ms, from an earlier estimate), and the GPU
  was the slower side in 92% of measured MoE-layer combines while the CPUs had slack.
- **The workaround:** `GLM53_CT_TZC`, the policy's assumed PCIe cost per expert. Raising it moves those experts back to
  the CPU.
- **One user:** in the tested workload I saw no PCIe spills at either setting. With the launch thread pinned, TZC 1.3
  gave 28.62/29.23 tok/s and 3.0 gave 29.28/28.72, with no clear change.
- **4 users:** one unpinned run per setting gave the totals below. The 2-user sweep did not show the same trend.

  | `GLM53_CT_TZC` | 1.3 | 2.0 | 2.5 | 3.0 | 4.0 |
  |---|---|---|---|---|---|
  | 4 users | 21.22 | 26.89 | 27.47 | 29.33 | 29.25 |

- **The proper fix** is cost rows measured with realistic (disjoint) routing; until then 3.0 compensates.

## The measured profile

Upstream's `docker/entrypoint.sh` sets the base `fast`-mode settings (first block). Run natively, the measured
server's full environment is (paths shortened; give each `GLM53_CT_CFLAGS` set its own `GLM53_CT_BUILD` directory):

```bash
# upstream fast-mode base (what entrypoint.sh exports)
export GLM53_MODE=fast GLM53_CPU_TIER=1 GLM53_EC=1 GLM53_K_FTSPLIT=1 GLM53_K_HCFUSE=1 GLM53_K_OVL=1 \
  GLM53_ZC_VRAM=0 GLM53_EC_RESERVE_GB=1.5 GLM53_EC_STAGE_GB=2.6 GLM53_EC_ELASTIC_GB=10 \
  GLM53_ZC_STATS=$PWD/data/stats_own_dec.json GLM53_EC_WARM=$PWD/data/stats_own_dec.json \
  GLM53_TRITON_PIN=$PWD/data/triton_pin_exact.json
# upstream knobs, tuned for this box
export GLM53_CT_CPUS=2-43 GLM53_CT_SWZ=1 GLM53_CT_B=0.31 GLM53_CT_TZC=1.3   # 1.0 before N32CTRL; 1.3 lets the faster CPU take more misses
# this branch
export GLM53_CT_SPLIT1=2 GLM53_CT_SOCKQ=1 GLM53_CT_P0BLK=1 GLM53_ZC_THP=1 GLM53_LA=1 GLM53_CT_SWIGLU_CLAMP=1 \
  GLM53_EC_BG=1 GLM53_CT_FUSE=1 GLM53_CT_FUSE_MAXN=1 GLM53_CT_PF_LEAN=1 \
  GLM53_CT_CFLAGS="-mavx2 -mfma -mf16c -mtune=znver3 -DFT_OPT_BSUM_SHUF=1 -DFT_OPT_FLOAT_EXTRACT=1 \
    -DFT_OPT_EXP_EXTRACT=1 -DFT_OPT_BSUM_MADD_N=32 -DFT_OPT_CTRL_REUSE=1"   # N32CTRL (2026-10-16)
python glm53/serve.py -m /path/to/GLM-5.3-Flash-exl3-3.05bpw -cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4 \
  --host 0.0.0.0 --port 30000 --served-name glm-5.3-flash
```

## Adapting it to your machine

- **Hardware floor:** CUDA GPU with 24 GB (built and tested on sm_86, `TORCH_CUDA_ARCH_LIST=8.6`); x86 CPU with AVX2,
  FMA and F16C. Model: `turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw` (125 GB), not `main`.
- **RAM:** about 225 GB for the process with `GLM53_CT_SWZ=1` (two copies of the routed experts: one pinned for the
  GPU's PCIe reads, one laid out for the CPU kernel). `GLM53_CT_SWZ=0` keeps one copy (~115 GB less), about 3% slower here.
- **Pinned memory:** ~115 GB is registered with CUDA; raise the memlock limit (`ulimit -l unlimited`, or
  `--ulimit memlock=-1` in Docker).
- **`GLM53_CT_CPUS`:** physical cores only. `2-43` is this box's two sockets minus cores 0-1; its SMT siblings are
  44-87, and adding them was 9% slower.
- **`GLM53_CT_B` / `GLM53_CT_TZC` must be refit.** They are the cost model's CPU time per expert and GPU zero-copy
  time per expert, in the same units, and decide which cache misses the CPU computes. 0.31 / 1.0 fit 42 Broadwell
  cores on PCIe 3.0. Refit from `/stats`: aim for the CPU job time (`host_busy_ms_per_job`) and the GPU's wait on it
  (`gpu_wait_ms_per_job`) to be about equal, then confirm with a few decode runs.
- **Expect:** ~6.5 min to load; speed rising as the VRAM expert cache warms; slow prefill (~119 tok/s on a
  2k-token prompt here); one stream at a time.

## Correctness

The CPU-kernel changes (`SPLIT1`, `P0BLK`, `SOCKQ`, the Broadwell compile flags) were checked bit-identical to the
stock kernel with upstream's `GLM53_CT_VALIDATE=1` panel and a same-binary harness. `GLM53_CT_SWIGLU_CLAMP` changes
numerics on purpose (it matches the GPU path). `fast` mode as a whole is not bit-identical to stock exllamav3 and not
run-to-run deterministic; see upstream's README and the field note.

## License

MIT, as upstream (see `LICENSE`). Upstream design and code by 0xSero and contributors; the patches on this branch by
Nathan Keilar.

## Reproducing the decode-kernel A/B

`tools/bench/kernel-ab-harness.cpp` drives the CPU tier on synthetic EXL3 layers (any 96 bytes are a valid trellis
tile) and dumps raw outputs, or times decode-shaped calls with `HARNESS_TIME=1`. Bit-exactness check for a flag set:

```bash
SRV="-mavx2 -mfma -mf16c -mtune=znver3 -DFT_OPT_BSUM_SHUF=1 -DFT_OPT_FLOAT_EXTRACT=1"
N32="-DFT_OPT_EXP_EXTRACT=1 -DFT_OPT_BSUM_MADD_N=32 -DFT_OPT_CTRL_REUSE=1"
g++ -O3 -std=c++17 $SRV      -pthread -DHAVE_SPLIT1 -I kernels/cpu_avx2 tools/bench/kernel-ab-harness.cpp -o base
g++ -O3 -std=c++17 $SRV $N32 -pthread -DHAVE_SPLIT1 -I kernels/cpu_avx2 tools/bench/kernel-ab-harness.cpp -o n32
for swz in 0 1; do HARNESS_SWZ=$swz ./base a$swz.bin; HARNESS_SWZ=$swz ./n32 b$swz.bin; cmp a$swz.bin b$swz.bin; done
```

Repeat with `-ffp-contract=off` added to both; GCC's default contraction can move last bits after any header edit.
