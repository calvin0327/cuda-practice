## NCU Summary (v0–v5)

### Key metrics comparison (from `csrc/cuda/softmax/ncu/`)

| Version | Duration | DRAM Throughput (% Max BW) | Memory Throughput (GB/s) | Compute (SM) Throughput | Achieved Occupancy | Main takeaway |
|---|---:|---:|---:|---:|---:|---|
| v0 | 3.12 ms | 32.43% (Max BW 56.42%) | 194.44 | 2.94% | 3.75% | Grid is too small (0.11 waves/SM) + uncoalesced globals; also incorrect |
| v1 | 886.56 µs | 76.24% | 457.01 | 9.87% | 30.94% | Starting point for memory-bound: bandwidth-dominated; warps often wait on L1TEX data |
| v2 | 658.02 µs | 79.63% | 477.30 | 11.99% | 31.16% | Still DRAM-limited; better bandwidth utilization |
| v3 | 657.89 µs | 79.90% | 478.93 | 11.84% | 31.25% | Essentially the same as v2 (and the report shows a kernel name mismatch) |
| v4 | 540.80 µs | 92.53% | 554.70 | 9.83% | 31.36% | Near the DRAM roofline; further gains require reducing memory traffic |
| v5 | 72.13 µs | 85.17% | 509.87 | 12.67% | 30.01% | Order-of-magnitude lower latency; still mostly memory-bound, but far more efficient overall |

### One-line summary per version
- **v0**: Underutilized GPU + severe uncoalesced memory access (and incorrect results).
- **v1–v3**: DRAM-bandwidth-bound; ramping from ~76% to ~80% BW, with v2/v3 plateauing.
- **v4**: Bandwidth utilization near peak (~92%); pure “move data faster” optimizations see diminishing returns.
- **v5**: Very low latency at high BW utilization; next gains likely come from fewer DRAM bytes and better fusion.