# Build the E1 bank and serve it

This directory contains the public E1 build recipe for GLM-5.3-Flash. It
rebuilds routed layers 3–44 from Unsloth's Q8_0 GGUF, captures actual decode
inputs from the EXL3 model, replays quality, measures local costs, and prepares
the native modules required by `glm53/lut_manifest.py`. Conversion alone never
admits serving. I16 real quality, local measured costs, verified bank bytes,
and a matching precompile receipt are separate startup requirements.

## Reproducibility boundary

**Historical whole-file hashes are not portable.** The pinned quantizer writes
its imatrix filename into GGUF metadata. The historical bank used an absolute
machine-specific filename. The public converter instead passes a stable relative
filename (`blk.03.imatrix.dat`, etc.), preserving the recipe, extraction and
per-expert normalization while avoiding machine-specific metadata. Changing
that string changes the GGUF header, padding and SHA-256. Matching the historical
whole-file hashes under these public-path constraints is therefore unresolved;
this recipe must not be described as a verified byte-identical rebuild of the
historical files. Full-size conversion and GPU capture have not been rerun as
part of this offline publication. The arithmetic and admission contracts have
offline fixture coverage.

[`reference-bank.json`](reference-bank.json) is the complete historical manifest,
with all 42 known-good whole-file SHA-256s, offsets, sizes and tensor layouts.
Its SHA-256 is
`e7d563c720e0186eb17e6b3dc40a28654bc78add7ace5dbdf7d5ddb7bd8cf297`.
For example, historical `blk.03.lut.gguf` is
`2faa9a83a4064f010d26e68473f60563288d4c67c4ea231a3ec4a406095b4a6c`.
Use the **newly generated** manifest for a public rebuild and generate new real
quality reports bound to that bank. Do not substitute the reference manifest
for files with different hashes.

**Tensor payloads are comparable.**
- [`reference-payload.json`](reference-payload.json) has, for all 42 historical layers, the SHA-256 of each file
  from its `data_start` to the end. That covers the tensor data only and excludes the GGUF metadata that carries
  the imatrix filename.
- The 42 payloads sum to exactly 143,879,307,264 bytes, the canonical bank size the preflight pins.
- A rebuild whose payload hashes all match has the same quantised weights as the historical bank.
- Determinism of the full public rebuild has not been demonstrated yet, so treat a match as confirmation and a
  mismatch as a reason to investigate. Either way, serve with your own manifest and your own bank-bound reports.

## Prerequisites and capacity

Use Linux x86-64 with AVX2, FMA and F16C, Python 3.10+, Git, CMake, a C++17
compiler, Ninja, and enough local SSD space. CPU conversion and its unit tests
use the Python standard library. The numerical replay also requires `g++` and
the included MIT-licensed [`../lutbench`](../lutbench) codec sources.

Budget approximately **480 GiB for the bank rebuild**, plus reserves and the
separate EXL3/comparator artifacts for quality. This is roughly 515 decimal GB;
460 decimal GB is insufficient for all sources plus the completed bank.

| Item | Required storage |
| --- | ---: |
| Eight Q8_0 shards plus imatrix | 341,494,653,824 bytes (318.0 GiB) |
| Canonical bank tensor payload | 143,879,307,264 bytes (134.0 GiB) |
| Standalone GGUF headers | about 0.4 GB across 42 files |
| One extracted Q8 triplet | 7,700,742,144 bytes |
| Retained Q8 triplets for layers 3 and 11 | 15,401,484,288 bytes |
| EXL3 3.05bpw model, required for capture/serving | about 125.3 GB extra |
| UD-IQ4_XS comparison tensors and fetch scratch | allow 20 GB extra |

Allow **650 GiB or more** when keeping every source, capture prerequisite,
comparator and scratch file together. Do not count a hard-link publication as
a second copy. Conversion needs only a few GiB of working RAM beyond the OS
cache; real capture/serving needs much more. A 512 GiB RAM machine is a practical
reference configuration for the combined EXL3 and LUT serving process.

For capture and precompile, use the runtime pinned by this repository:

- PyTorch with CUDA support, CUDA toolkit with `nvcc`, and matching compiler.
  The measured reference used CUDA 13.0 and an RTX 3090 (`--cuda-arch 8.6`).
- exllamav3 **1.5.1**, source revision
  `958ec933361b24eb8426ec7222e5b0062a679dcd`; install its normal dependencies,
  including `safetensors`, `tokenizers`, `numpy`, `psutil` and `marisa-trie`.
  Serving also uses the dependencies installed by the repository's
  [`../../docker/Dockerfile`](../../docker/Dockerfile).
- EXL3 model `turboderp/GLM-5.3-Flash-exl3`, **3.05bpw** revision
  `332ab457b709b7ba30dd9a448be5de03b80a7ac9`. The complete published-file
  inventory is [`exl3-source.json`](exl3-source.json).

The unchanged serving preflight currently requires **42 distinct physical
worker cores plus a separate physical coordinator core**. Local cost placement
uses two NUMA nodes. CPU numbers below describe the measured dual-socket
machine; check `lscpu -e=CPU,CORE,SOCKET,NODE` and select your own physical cores.
SMT siblings do not count as different physical cores. A smaller machine can
build and replay the bank, but does not meet this serving contract.

All commands below run from the repository root. `e1-work/` is gitignored.
Use a clean, committed serving checkout: precompile receipts bind its exact Git
revision, native source hashes and runtime versions. Acquisition commands are
the only steps that need network access. For an offline machine, stage the
pinned files from another machine and use the verification commands instead.

The small `maintenance_guard.py` helper now provides only optional local RAM
and disk headroom checks, CPU affinity, niceness and immutable publication.
Its filename persists because the real-report schema binds its source hash.
Capture, comparator fetch and replay accept `--min-ram-gib` and
`--min-disk-gib`; zero disables the corresponding headroom check. Conversion
and the cost/precompile tools expose their own CPU/thread/job settings. None
of these tools controls another process or requires a maintenance window.

## 1. Fetch and verify the exact Q8_0 source

```bash
python3 tools/e1/fetch_sources.py plan
python3 tools/e1/fetch_sources.py fetch --source ./e1-work/source
```

This pins Unsloth's `GLM-5.3-Flash-GGUF` revision
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, all eight shard lengths and SHA-256s,
and imatrix SHA-256
`27d2ef6b6be0e314b373080f96322a1e4668b09b192664f3edc231c3b7c2d749`.
`plan` prints every exact source pin. Downloaded files remain `.partial` until
their complete SHA matches; resumed requests must return the exact byte range.
`EXPECTED` records the complete canonical inventory. `FETCH-DONE` contains
`DONE` only after all nine files pass verification. Existing published files
are rehashed and never overwritten.

Offline staging uses the same relative directory layout (`Q8_0/…` plus
`imatrix_unsloth.gguf_file`):

```bash
python3 tools/e1/fetch_sources.py verify --source ./e1-work/source
```

## 2. Build the pinned CPU quantizer

Obtain `https://github.com/ikawrakow/ik_llama.cpp` at exactly
`5bf8f0fe4db98e9560de0f495232f8d3a733c678`. The following acquisition is for a
connected preparation machine; an offline source checkout at the same revision
works identically. Keep the source clean and build out of tree.

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp ./e1-work/ik_llama.cpp
git -C ./e1-work/ik_llama.cpp checkout 5bf8f0fe4db98e9560de0f495232f8d3a733c678
cmake -S ./e1-work/ik_llama.cpp -B ./e1-work/ik-build \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
nice -n 10 cmake --build ./e1-work/ik-build --target llama-quantize -j 2
python3 tools/e1/quantize_s1.py build-receipt \
  --quantizer ./e1-work/ik-build/bin/llama-quantize \
  --quantizer-build-dir ./e1-work/ik-build \
  --output ./e1-work/ik-build/reviewed-build.json
```

The receipt binds the pinned source revision, clean tracked sources, CPU-only
backend configuration, executable and linked llama/ggml library hashes. Keep
those shared libraries beside their reviewed executable when relocating a build.

## 3. Convert all 42 routed-expert triplets

```bash
python3 tools/e1/quantize_s2.py run \
  --source ./e1-work/source --output ./e1-work/bank \
  --scratch ./e1-work/scratch \
  --quantizer ./e1-work/ik-build/bin/llama-quantize \
  --build-receipt ./e1-work/ik-build/reviewed-build.json \
  --threads 2 --nice 10
```

Optional `--cpus` selects CPU numbers or ranges, for example `0-3,8-11`. Set `--threads` to the
number you intend to use; using more physical cores can substantially reduce
conversion time. The historical 44-thread and two-thread runs matched once the
embedded imatrix filename was held constant. No service or other process is
paused, stopped, inspected or contacted by this public tool.

| Actual layer | Gate / up | Down |
| --- | --- | --- |
| Ordinary layers 3–44 | IQ3_K | IQ4_KS |
| 11 | IQ4_KS | Q6_K |
| 12 and 44 | IQ3_K | Q6_K |

The base ftype is **IQ3_K**, with anchored `--custom-q` rules for every complete
tensor name and ik's exact case-sensitive type spellings. Do not substitute
`COPY`, `--pure` or a different quantizer revision. Only routed gate/up/down
tensors enter each standalone quantizer input; all `split.*` metadata is removed.
The GGUF v3 imatrix is converted to ik's legacy format by computing
`in_sum2[expert, channel] / count[expert]`, rounding each result to float32,
and writing `ncall=1`.
Zero expert counts, nonfinite data, negative sums or incomplete coverage fail.
ik's supplied adapter divides all experts by `counts[0]`; it is unsuitable here.

Outputs include `bank/blk.03.lut.gguf` through `bank/blk.44.lut.gguf`, the
`bank/e1-lut-bank.json` manifest (`e1-lut-bank/v1`), conversion receipts and the
retained layer-3/11 Q8 inputs plus `bank/s1-manifest.json` for real replay.
The byte ledger includes the **extra four bytes per IQ4_KS row**:
143,879,307,264 canonical payload bytes and 144,063,332,352 bytes for TP2 placement.

The legacy manifest fields `accepted_quality_verdict`, `accepted_proxy_sha256`
and `private_manifest_sha256` remain required by the existing schema. The
historical proxy digests identify prior recipe evidence; they do not describe a
new quality run. `private_manifest_sha256` binds the locally produced conversion
ledger. `real_pass` stays false. Fresh real replay supplies serving admission.

Re-running with the same arguments verifies and reuses completed receipts.
Published artifacts are immutable; conflicting outputs stop the build. Use new
output paths for a different recipe or tool revision. Scratch pending files
are not admitted as published weights.

## 4. Verify the finished bank

```bash
python3 -m glm53.lut_manifest \
  --manifest ./e1-work/bank/e1-lut-bank.json --verify-files
```

This streams every file, checks its complete SHA against **its own generated
manifest**, and verifies tensor names, types, dimensions, offsets, canonical
bytes and TP2 row-scale arithmetic. It grants no quality or serving admission.
Compare your tensor payloads with the historical bank's (each file is hashed from its own `data_start`):

```bash
python3 - <<'EOF'
import hashlib, json
from pathlib import Path
bank = Path('./e1-work/bank')
mine = {row['layer']: row['artifact'] for row in json.loads((bank / 'e1-lut-bank.json').read_text())['layers']}
ref = {row['layer']: row for row in json.loads(Path('tools/e1/reference-payload.json').read_text())['layers']}
bad = 0
for layer, r in sorted(ref.items()):
    a = mine[layer]; h = hashlib.sha256()
    with open(bank / a['path'], 'rb') as f:
        f.seek(a['data_start'])
        while chunk := f.read(64 << 20):
            h.update(chunk)
    ok = h.hexdigest() == r['payload_sha256']; bad += not ok
    print('OK ' if ok else 'DIFF', layer, a['path'])
print('all 42 tensor payloads match the historical bank' if not bad else f'{bad} layer(s) differ')
EOF
```

## 5. Prepare real capture and the comparison weights

Acquire the pinned EXL3 files into `./e1-work/exl3`, then generate a local,
fully hashed model ledger and the four-prompt corpus:

```bash
hf download turboderp/GLM-5.3-Flash-exl3 \
  --revision 332ab457b709b7ba30dd9a448be5de03b80a7ac9 \
  --local-dir ./e1-work/exl3
python3 tools/e1/capture_activations.py prepare-model-ledger \
  --model-dir ./e1-work/exl3 --out ./e1-work/model-ledger.json
python3 tools/e1/capture_activations.py prepare-corpus --out ./e1-work/corpus.json
python3 tools/e1/quality_checkpoint.py build --out ./e1-work/replay/quality_replay.so
python3 tools/e1/fetch_unsloth_baseline.py \
  --layers 3,11 --out ./e1-work/unsloth
```

The baseline fetch uses [`ud-source.json`](ud-source.json), a pinned publisher
header inventory for the actual layer-3 and layer-11 UD-IQ4_XS tensors. Bounded
range requests reject wrong status, ranges, lengths, encodings or publisher
identity before reading the body. The range ledger records chunk hashes for
later offline replay. Never compare with a renamed layer or synthetic proxy.

Precompile both capture and LUT variants from the final committed serving
revision. Compilation needs the CUDA toolkit and the installed runtime, but
does not initialize a GPU context:

```bash
E1_REV=$(git rev-parse HEAD)
python3 tools/e1/precompile_extension.py --variant capture --revision "$E1_REV" \
  --build-root ./e1-work/precompiled --cuda-arch 8.6 --jobs 2
python3 tools/e1/precompile_extension.py --variant lut --revision "$E1_REV" \
  --build-root ./e1-work/precompiled --cuda-arch 8.6 --jobs 2
```

`--cuda-home` can name an existing toolkit. `--plan` checks and prints a build
plan without importing Torch. Every build variant requires a new directory;
choose another `--build-root` after a failure or source revision change.

## 6. Capture actual routed decode inputs and replay quality

Use a GPU with enough free VRAM and enough RAM for the pinned EXL3 model. The
capture loads the model in its own process. Do not point it at an unrelated
running service. It records actual normalized inputs, original top-8 routes,
token trajectories, runtime/model ledgers, static scores and corpus hashes.

```bash
E1_CAPTURE_PRE=./e1-work/precompiled/capture/reviewed-precompile.json
python3 tools/e1/capture_activations.py capture \
  --serving-source . --serving-revision "$E1_REV" \
  --model-dir ./e1-work/exl3 --model-ledger ./e1-work/model-ledger.json \
  --ec-precompile "$E1_CAPTURE_PRE" \
  --ec-precompile-sha256 "$(sha256sum "$E1_CAPTURE_PRE" | cut -d' ' -f1)" \
  --corpus ./e1-work/corpus.json --stats ./data/stats_own_dec.json \
  --layers 3,11 --tokens 32 --cache-size 8192 --device 0 \
  --out ./e1-work/capture
python3 tools/e1/quality_checkpoint.py run \
  --capture ./e1-work/capture/capture.json \
  --quant-manifest ./e1-work/bank/s1-manifest.json \
  --unsloth-manifest ./e1-work/unsloth/unsloth.json \
  --library ./e1-work/replay/quality_replay.so \
  --bank-manifest ./e1-work/bank/e1-lut-bank.json \
  --activation i16 --batch 1 --out ./e1-work/quality-i16.json
```

The mandatory I16 report uses `e1-real-quality/v1`. Exit 0 means a completed
`REAL-PASS`; exit 2 means `REAL-FAIL` or `BLOCKED`. Gates compare per-expert median
and route-frequency-weighted NMSE against the actual UD comparator, both per
layer and pooled, and check TP2/full-down drift. Missing coverage or provenance
cannot become a pass. The fixed capture requires at least four unique prompts
across prose, code and firmware classes. Start with 32 inputs per prompt; use
more if actual routed coverage is insufficient.

For optional Q8, repeat exactly the replay command with
`--activation q8 --i16-result ./e1-work/quality-i16.json` and a new output
`./e1-work/quality-q8.json`. Keep the same bank, capture, weights, replay library
and batch so the I16 and Q8 `binding` objects are identical. Q8 does not replace
I16. A failed capture cannot resume its recurrent model state: retain its
directory and choose a new capture output. Replay can use an existing complete
capture, always writing a new report path.

## 7. Measure costs on the serving machine

```bash
E1_BANK=./e1-work/bank/e1-lut-bank.json
E1_BANK_SHA=$(sha256sum "$E1_BANK" | cut -d' ' -f1)
E1_PRE=./e1-work/precompiled/lut/reviewed-precompile.json
E1_PRE_SHA=$(sha256sum "$E1_PRE" | cut -d' ' -f1)
python3 tools/e1/lut_cost.py extract \
  --serving-revision "$E1_REV" --bank-manifest "$E1_BANK" \
  --bank-manifest-sha256 "$E1_BANK_SHA" --layers 3,11,12 \
  --out ./e1-work/cost-fixtures
E1_FIXTURES=./e1-work/cost-fixtures/fixture-manifest.json
python3 tools/e1/lut_cost.py measure \
  --serving-revision "$E1_REV" --bank-manifest "$E1_BANK" \
  --bank-manifest-sha256 "$E1_BANK_SHA" \
  --fixtures "$E1_FIXTURES" \
  --fixtures-sha256 "$(sha256sum "$E1_FIXTURES" | cut -d' ' -f1)" \
  --precompile "$E1_PRE" --precompile-sha256 "$E1_PRE_SHA" \
  --cpus 2-43 --coordinator-cpu 0 --activation i16 \
  --samples 21 --warmup 3 --act-limit 10 \
  --out ./e1-work/cost-i16.json
```

Run on an otherwise quiet machine with the coordinator and all workers allowed
by its cpuset. The three real eight-expert fixtures cover ordinary, upgraded
gate/up and upgraded down profiles. Every M=1–4 / K=0–8 cell retains raw samples
and its recomputable median. The loaded fixture weights are flushed before each
timed call; flush and ingress preparation stay outside the timer. The resulting
`e1-lut-cost/v1` table binds topology, exact workers, activation, bank, fixtures,
native build SHA and measured placement. Partial or warm diagnostic tables do
not admit startup. For Q8 serving, measure a separate Q8 table with the same
arguments and `--activation q8 --out ./e1-work/cost-q8.json`.

## 8. Verify real admission and serve

Inspect the reports and record the exact artifact hashes you accept. The
following profile computes hashes from your inspected local outputs. On startup,
preflight recomputes the quality gates and checks those hashes; no JSON approval
flag bypasses validation.

```bash
export GLM53_CPU_TIER=1 GLM53_EC=1
export GLM53_CT_FORMAT=lut GLM53_CT_LUT_ACT=i16
export GLM53_CT_LUT_MANIFEST="$PWD/e1-work/bank/e1-lut-bank.json"
export GLM53_CT_LUT_QUALITY="$PWD/e1-work/quality-i16.json"
export GLM53_CT_LUT_QUALITY_SHA256=$(sha256sum "$GLM53_CT_LUT_QUALITY" | cut -d' ' -f1)
export GLM53_CT_LUT_COST="$PWD/e1-work/cost-i16.json"
export GLM53_CT_LUT_COST_SHA256=$(sha256sum "$GLM53_CT_LUT_COST" | cut -d' ' -f1)
export GLM53_CT_LUT_PRECOMPILE="$PWD/e1-work/precompiled/lut/reviewed-precompile.json"
export GLM53_CT_LUT_PRECOMPILE_SHA256=$(sha256sum "$GLM53_CT_LUT_PRECOMPILE" | cut -d' ' -f1)
export GLM53_CT_LUT_BATCH_POLICY=bounded GLM53_CT_MAXN=32 GLM53_CT_MAXBSZ=4
export GLM53_CT_CPUS=2-43 GLM53_CT_THREADS=42 GLM53_CT_COORD_CPU=0
export GLM53_CT_TIMEOUT_S=2
export GLM53_ZC_STATS="$PWD/data/stats_own_dec.json"
export GLM53_EC_WARM="$PWD/data/stats_own_dec.json"
export GLM53_TRITON_PIN="$PWD/data/triton_pin_exact.json"
export GLM53_ZC_VRAM=0
export GLM53_EC_RESERVE_GB=1.5 GLM53_EC_STAGE_GB=2.6 GLM53_EC_ELASTIC_GB=10
export GLM53_ZC_HOME_NODE=0 GLM53_ZC_NODE1_ARENAS=0
python3 -m glm53.lut_manifest --admit-quality
python3 glm53/serve.py -m ./e1-work/exl3 -cs 8192 \
  --max-batch-size 4 -chunk_size 8192 -ambs 4
```

`GLM53_ZC_VRAM=0` keeps all routed experts in pinned host memory for GPU
zero-copy reads; the value counts experts per layer, not GiB. The expert-cache
budgets above are examples for a 24 GiB GPU. On two-socket systems,
`GLM53_ZC_HOME_NODE` selects the
GPU-local preferred NUMA node for pinned EXL3 arenas;
`GLM53_ZC_NODE1_ARENAS=N` places the first N arenas preferentially on node 1
before the home-node preference. Check the GPU's PCI NUMA node and available
RAM instead of copying these settings blindly. These environment variables
must be set before model import. LUT cost topology and worker CPU settings
must match the machine on which the table was measured.

For Q8, set the matching cost table, keep the mandatory I16 report, and add:

```bash
export GLM53_CT_LUT_ACT=q8
export GLM53_CT_LUT_Q8_QUALITY="$PWD/e1-work/quality-q8.json"
export GLM53_CT_LUT_Q8_QUALITY_SHA256=$(sha256sum "$GLM53_CT_LUT_Q8_QUALITY" | cut -d' ' -f1)
export GLM53_CT_LUT_COST="$PWD/e1-work/cost-q8.json"
export GLM53_CT_LUT_COST_SHA256=$(sha256sum "$GLM53_CT_LUT_COST" | cut -d' ' -f1)
```

## Timing expectations and offline tests

Measured values below come from **two Xeon E5-2699 v4 CPUs plus an RTX 3090**;
they are historical measurements, not results of the public recipe run.

| Step | Measured reference / planning estimate |
| --- | --- |
| Q8 download | Estimated 57 minutes at sustained 100 MB/s; network-dependent |
| Full source hashing | Estimated 6–30 minutes at 1 GB/s–200 MB/s |
| Quantizer build / receipt | Estimated 5–20 minutes, compiler-dependent |
| Ordinary layer 3, 44 threads | Measured quantizer 147.312 s wall / 138.980 s unpaused; complete checked invocation 234.253 s; peak RSS 3.054 GiB |
| Ordinary layer 3, two threads | Measured quantizer 1,001.43 s unpaused |
| Full 42-layer conversion | Estimated 2.6–3+ hours at 44 threads; roughly 12–18 hours at two threads, plus first source verification; not measured end-to-end here |
| Bank hashing | Estimated 2.5–12 minutes at 1 GB/s–200 MB/s |
| Precompile both variants | Estimated 5–20 minutes depending on toolchain/jobs |
| Real capture and CPU replay | Estimated 10–45 minutes; model load, coverage and CPU selection dominate |
| Three-profile local cost grid | Estimated 1–10 minutes per activation, plus fixture verification/extraction |

No unit test requires network, a GPU or real weights:

```bash
mkdir -p ./e1-work/test-tmp
TMPDIR="$PWD/e1-work/test-tmp" python3 -m unittest discover -s tools/e1 -p 'test_*.py' -v
TMPDIR="$PWD/e1-work/test-tmp" python3 -m unittest discover -s tests -p 'test_lut_manifest.py' -v
```

Tests use small fixtures for normalization, tensor layouts, source tampering,
immutable publication, replay numerics, real-report provenance rejection,
cost-grid/raw-sample validation and precompile plans. AVX2 numerical tests build
the small replay library when a compatible compiler/CPU is available. They do
not publish a claim that real model quality passed.
