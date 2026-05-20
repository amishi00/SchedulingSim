# Optimized Simulation of Discrete Rank-Based Scheduling

**Authors:** Amishi Gupta, Jasmine Si, Michael Sidoti

**Course:** Cornell ORIE 4999, Spring 2026

Done under the guidance of Professor Ziv Scully and Amit Harlev.

## Overview

This project simulates single-server job queues governed by **discrete rank functions** — scheduling policies that prioritize jobs based on how long they have been served (their *age*). The scheduler always runs the job with the lowest rank.

A discrete rank function is represented as a list of `(age, rank)` breakpoints. The resolution parameter `k` densifies the breakpoint list by a factor of `k`, allowing us to benchmark how each simulator scales with rank function complexity.

Three simulation approaches are compared:

| Simulator | Description |
|---|---|
| **Naive** | Fully event-driven; steps through every rank breakpoint one at a time. Simple but slow for large `k`. |
| **Bisect** | Uses binary search over rank levels and group advancement to skip directly to the largest reachable rank threshold before the next arrival or completion. |
| **Bisect + NumPy** | Same bisection logic as Bisect, but inner loops are replaced with vectorized NumPy operations for lower per-iteration overhead. |
| **LAS Event-Driven** | Reference implementation of Least Attained Service (`rank = age`). No rank lookup needed; used as a performance baseline. |

## Files

| File | Purpose |
|---|---|
| `plot_runtimes.py` | Main benchmarking script. Runs Naive, Bisect, and Bisect+NumPy across `k=1..max_k` and plots wall-clock runtime vs `k`. |
| `naive.py` | Naive event-driven simulator. Importable module with `run(k, ...)`. |
| `unoptimized_bisect.py` | Bisect simulator (pure Python). Importable module with `run(k, ...)`. |
| `bisect_np.py` | Bisect + NumPy simulator. Importable module with `run(k, ...)`. |
| `las_event_driven.py` | LAS event-driven simulator (no `k` parameter). Importable with `run(N, ...)`. |
| `job_queue.py` | `Job` dataclass used by all simulators. |
| `piecewise_np.py` | NumPy-vectorized `RankIndex` used internally by `bisect_np`. |
| `piecewiseNEW.py` | Pure-Python `RankIndex` used internally by `unoptimized_bisect`. |

## Running `plot_runtimes.py`

```bash
python plot_runtimes.py <max_k>
```

**Example** — sweep `k` from 1 to 10:
```bash
python plot_runtimes.py 10
```

This prints a runtime table and opens a matplotlib plot. Key parameters are set at the top of the file:

```python
N                 = 1000        # number of jobs
MEAN_INTERARRIVAL = 1.0         # mean inter-arrival time (Exponential)
MEAN_SIZE         = 3.0         # mean job size (Exponential)
SEED              = 3           # random seed
RANK_DATA         = RANK_PREPROVIDED   # rank function to use (see below)
```

Two rank datasets are predefined in the file:
- `RANK_PREPROVIDED` — the 400-point research rank dataset.
- `RANK_LAS` — discretized LAS identity rank (`[(i/100, i/100) for i in range(10000)]`).

To include the LAS event-driven baseline as a horizontal reference line, uncomment the three lines marked in the file (search for `LAS Event-Driven`).

## Running Individual Simulators

Each of `naive.py`, `unoptimized_bisect.py`, and `bisect_np.py` can be run directly for a single `k` value. They use their own internal default rank data and `N=200` jobs.

```bash
python naive.py <k>
python unoptimized_bisect.py <k>
python bisect_np.py <k>
```

**Example:**
```bash
python bisect_np.py 5
```


## Dependencies

```
python
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```
