# How far can hand fusion take LBM?

A small CUDA worklog for a concrete question: **can a hand-written LBM
megakernel beat a good one-step implementation, and which transformations
should an MLIR compiler learn?** The experiment is D3Q19 FP32 Lattice
Boltzmann on a fixed 120 × 120 × 150 lid-driven cavity, starting from the
[Enzyme-GPU-Tests LBM implementation](https://github.com/wsmoses/Enzyme-GPU-Tests/tree/mlir/LBM).
This repository builds and runs independently of that project and Enzyme.

<!-- RESULT_HEADLINE_BEGIN -->
**Best measured: D-V2-256, 135.82 µs/physical step (2.14× B0), versus B0 at 290.13 µs.** That is 135.82 ms versus 290.13 ms for 1000 timesteps. The large gain comes from two-step fusion; it establishes that hand optimization helps, not that the optimum has been found.
<!-- RESULT_HEADLINE_END -->

**Scope:** B0 already combines collision and streaming within one timestep.
The six spatial controls below still launch once per timestep. V1/V2 fuse
**two** timesteps per launch, keeping the odd-step intermediate on chip.
A 1000-step simulation therefore uses 500 temporal launches. A single launch
covering the entire simulation, and the performance ceiling of that design,
remain open questions.

## Performance at a glance

<!-- RESULTS_BEGIN -->
NVIDIA GeForce RTX 5090, driver 575.57.08, CUDA 12.9, clang version 23.0.0git (`sm_120`, `-O3 -ffast-math`), measured 2026-09-24. Default cavity, no obstacle file. Pooled medians of 40 measured trials per variant at **1000 physical steps/trial**, across 2 seeded shuffled process rounds; each process has five warm-ups and twenty trials. Both device buffers are restored outside each timed interval. Allocation, validation, and output are excluded.

**One-step spatial controls**

| Kernel | µs / physical step ↓ | Speedup over B0 ↑ |
| --- | ---: | ---: |
| B0 | 290.13 | 1.00× |
| C1 | 289.42 | 1.00× |
| C2 | 290.59 | 1.00× |
| C3 | 288.04 | 1.01× |
| S2-C128 | 302.87 | 0.96× |
| S2-D128 | 290.34 | 1.00× |

The best C1–C3 mapping is C3, with -0.72% time versus B0. S2-C128 changes time by +4.39%; S2-D128 by +0.07%.

**Two-step fusion** (C = cube, D = elongated; suffix = workers)

| Kernel | µs / physical step ↓ | Speedup over B0 ↑ |
| --- | ---: | ---: |
| D-V1-128 | 149.03 | 1.95× |
| D-V1-256 | 137.21 | 2.11× |
| D-V2-128 | 149.34 | 1.94× |
| D-V2-256 | 135.82 | 2.14× |
| C-V2-256 | 142.12 | 2.04× |
| T30-V2-128 | 144.94 | 2.00× |

Speedup = B0 time / candidate time. Values below 1 mean slower than B0. Treat sub-percent differences cautiously.

<!-- RESULTS_END -->

Times are per **physical timestep**, including GPU launch gaps. These are
fresh measurements of this repository, not a mixture of the earlier NSYS
and CUDA-event experiments. Consolidating shapes into templates can change
code generation; all variants are rebuilt and retimed together. The compact [complete table](docs/BENCHMARKS.md)
includes both worker counts and the 100-step cross-check. The
[raw event trials](results/rtx5090/events.json) retain execution order,
rounds, build identity, and GPU telemetry.

## 1. B0: keep the original computation

[`b0.cuh`](kernels/b0.cuh) launches one block per X row: 120 threads,
120 × 150 blocks. Each thread reads 19 local populations, applies obstacle
bounce-back or fluid collision, and pushes the results to directional
neighbors. Populations use FP32 structure-of-arrays storage; flags are bytes.
All one-step variants call the same [original cell body](kernels/cell.cuh).

The useful population payload is 19 × 4 B reads + 19 × 4 B writes =
**152 B/cell/step**, or 328.32 MB per domain update. This is a useful-work
convention, not measured DRAM traffic; caches and transactions affect the
latter. B0 is already a substantial streaming kernel, so reducing arithmetic
or a short launch gap alone is unlikely to explain a twofold improvement.

## 2. C1, C2, C3: change where threads work

| Kernel | CUDA block | Grid | Change |
| --- | --- | --- | --- |
| [C1](kernels/c1.cuh) | 32 × 4 × 1 | 4 × 30 × 150 | One warp-wide X segment, four Y rows |
| [C2](kernels/c2.cuh) | 32 × 2 × 2 | 4 × 60 × 75 | Distribute those rows across Y and Z |
| [C3](kernels/c3.cuh) | 64 × 2 × 1 | 2 × 60 × 150 | Two warp-wide X segments, two Y rows |

All use 128 workers and one site per worker, with a physical-domain guard.
There is no shared memory or temporal reuse. C1/C2's last X block has
24 useful columns; C3's has 56. Changing the mapping does not eliminate
the edge waste. These controls test whether launch geometry alone is enough;
the substantial gains arrive with temporal fusion.

## 3. S2-C128 and S2-D128: establish matched tiles

[`S2-C128`](kernels/s2_c128.cuh) assigns an 8 × 8 × 8 cube to 128 workers;
[`S2-D128`](kernels/s2_d128.cuh) assigns a 16 × 8 × 4 elongated tile.
Both traverse 512 sites in an X-fast strided loop, usually four sites per
worker. Their grids are 15 × 15 × 19 and 8 × 15 × 38, respectively.
Physical bounds handle the final partial tiles.

**“S2” is the experiment-stage name, not two timesteps.** These kernels
still read and write global populations every step. They establish the
mapping cost before adding temporal reuse. The same kernels can launch
256 workers, giving matched controls for the 256-worker temporal variants.
An 8-wide cube can hurt transactions despite looking geometrically compact.

## 4. V1: fuse two steps, store producer state

[`temporal_v1.cuh`](kernels/temporal_v1.cuh) computes the first collision
over a core tile plus a one-cell halo. It stores all 19 post-collision
populations at every halo-box producer in shared memory. After one block
barrier, each core cell gathers its streamed intermediate populations,
collides again, and performs the original global scatter.

```text
f(t) in global memory
  -> collide producers in R + halo
  -> shared post-collision producer state
  -> block barrier
  -> gather into R, collide, scatter
  -> f(t+2) in global memory
```

Neighboring blocks independently recompute overlapping first-step halo
producers. This duplication lets each block finish both steps without
communicating with other blocks. The outer launch boundary still orders
the next pair of timesteps. Both V1 and V2 retain the full producer box,
including its eight unused corners.

For the 16 × 8 × 4 tile, shared storage is **19 × 18 × 10 × 6 × 4 =
82,080 B**. This fits only one block per SM on the tested GPU. Nevertheless,
eliminating the global odd-step intermediate gives a large measured gain.
128 and 256 workers are separate launch choices, not different precision
or collision algorithms.

## 5. V2: store only the intermediate that gets consumed

[`temporal_v2.cuh`](kernels/temporal_v2.cuh) performs the same producer work,
but scatters only populations entering the core into shared memory:

```text
producer p, direction q -> intermediate_R[q][p + c_q], if p + c_q is in R
block barrier          -> read all 19 populations locally at each core cell
                       -> second collision -> original global scatter
```

The 16 × 8 × 4 intermediate is now **19 × 512 × 4 = 38,912 B**.
Measured runtime resource queries permit two blocks per SM. There is still
only one removed global intermediate, exactly as in V1: halving shared
storage does not imply another twofold speedup. It also changes addressing,
store predicates, register use, and shared transactions.

The same templates include cube variants (8 × 8 × 8) and **T30** variants
(30 × 4 × 4, V2 only). T30 has a 32 × 6 × 6 producer box and a 36,480 B
shared core. Its halo traversal puts one 32-site row in a warp, but global
addresses are not automatically sector-aligned, and its flat second-phase
loop splits 30-site core rows across warps. These competing costs are why
we measure shapes rather than select them from shared-memory size alone.

<!-- LESSONS_BEGIN -->
In this campaign, D-V2-256 takes 135.82 µs/step: 2.07× faster than its same-worker one-step control, S2-D256 (281.69 µs). V2 changes time by -1.01% versus D-V1-256; shared capacity alone does not predict the time change.

The fastest cube V2 (C-V2-256) takes 142.12 µs; the fastest T30 V2 (T30-V2-128) takes 144.94 µs. At 128 workers, T30 V2 changes time by -2.95% versus D V2; at 256 workers, by +8.83%. A shape that helps one worker count need not improve the best overall configuration. Small percentage differences are descriptive; two process rounds do not establish statistical significance.
<!-- LESSONS_END -->

## Correctness is part of the transformation

The numerical expressions and their order are preserved. A source gate
checks the [temporal collision](kernels/collision.cuh) against B0; full
allocation comparisons check populations, flag bytes, unused flag-slot
bytes, padding, and margins. No error tolerance is used.

The inherited layout has zero Y padding: out-of-range Y coordinates can
alias physical sites in adjacent Z planes. Producer validity must therefore
use the **flattened address**, not geometric XYZ clipping. Slots without a
physical producer keep their initial values in both buffers. V2 has one
writer for each `(core cell, direction)` slot, including the fallback for
missing producers; all workers reach the barrier. The final scatter is
injective for each direction, so crossing a tile boundary does not imply
a write race.

<!-- VALIDATION_BEGIN -->
The shipped standalone campaign passed all **20 variants**: full-allocation bitwise comparisons, velocity output parity, and invalid-count checks. It includes default histories at 1000/1002 steps, patterned input at 1000, and external obstacles at 102. All variants also passed memcheck, racecheck, synccheck, and global-memory initcheck on six-step sentinel/obstacle input. B0 velocity output at 2/6/100 steps additionally matches the independently built original executable. See [the validation contract](docs/VALIDATION.md) and [saved gates](results/rtx5090/validation.json).
<!-- VALIDATION_END -->

## Build and reproduce

Requirements: Linux, Python 3.9+ (standard library only), a CUDA-capable
Clang supporting `sm_120`, CUDA 12.9, and a compatible NVIDIA GPU. The
recorded compiler is Clang 23.0.0git, LLVM commit
`c813428fb1b5678a6e4c541f4aee82a7977e95ea`. Other toolchains/architectures
need their own correctness gates and measurements. Nsight is unnecessary
for building or event timing; Compute Sanitizer is optional.

```sh
# Point at your CUDA-capable Clang; this is not a dependency on the parent repo.
export CUDA_CXX=/path/to/clang++
export CUDA_PATH=/usr/local/cuda-12.9
export CUDA_VISIBLE_DEVICES=0
make
./build/lbm --list
./build/lbm S2-D128 --check 100
./build/lbm D-V2-256 --check 1000 --patterned
./build/lbm D-V2-256 100 -o velocity.dat

make check    # all variants: boundary sentinels, output parity, invalid arguments
make bench    # fresh matched event trials, requires passing validation
# Optional full history matrix + four sanitizers, as used for the shipped results:
python3 scripts/validate.py --full --sanitizers
python3 scripts/report.py --results results/local --output docs/LOCAL_RESULTS.md
```

The default scripts write to ignored `results/local/`; they preserve the
shipped `results/rtx5090/` measurements. A benchmark run refuses a changed
binary/source snapshot and refuses to overwrite existing event trials.
Use a new `--out` directory for a new validation/benchmark campaign.
`make report` regenerates the shipped summary without running GPU work.

## What should MLIR learn from this?

The immediate target is a dependence-aware **two-timestep temporal tiling
pass**: derive the producer halo, duplicate boundary work deliberately,
promote the consumed intermediate to workgroup memory, insert one block
barrier, and preserve final scatter ownership. The hand-written variants
provide executable references and matched controls for that transformation.
The [GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) provides launches,
workgroup memory, and barriers; deciding when this transformation is legal
and profitable is the compiler work. See the short
[pass roadmap](docs/MLIR_FUSION.md).

A whole-simulation megakernel needs a further dependency/synchronization
strategy; a block barrier cannot order different blocks. Longer temporal
tiles, rolling storage, clusters, input staging, and submission controls
are separate experiments. Reduced precision changes the numerical contract.
None is implemented or credited with the speedup reported here.

The presentation is inspired by
[Si Boehm's CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM):
show the kernel change, the result, and the lesson. Upstream notices are
preserved; [NOTICE](NOTICE) and [provenance](docs/provenance.json) identify
the imported code and source snapshot.
