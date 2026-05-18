import argparse
import matplotlib.pyplot as plt

# ── Shared simulation parameters ─────────────────────────────────────────────
# Change any value here and both simulators will use the new value automatically.
N                 = 200   # number of jobs
MEAN_INTERARRIVAL = 1.0   # mean inter-arrival time
MEAN_SIZE         = 3.0   # mean job size
SEED              = 0     # random seed
# ─────────────────────────────────────────────────────────────────────────────

import naive
import unoptimized_bisect

_parser = argparse.ArgumentParser(description="Run both simulators for k=1..max_k and plot runtimes.")
_parser.add_argument("max_k", type=int, help="Upper bound for k (inclusive)")
_args = _parser.parse_args()

k_values   = list(range(1, _args.max_k + 1))
naive_times  = []
bisect_times = []

for k in k_values:
    print(f"\n{'='*50}")
    print(f"[Naive] k={k}")
    t_naive = naive.run(k, N=N, mean_interarrival=MEAN_INTERARRIVAL, mean_size=MEAN_SIZE, seed=SEED)
    naive_times.append(t_naive)

    print(f"\n[Bisect] k={k}")
    t_bisect = unoptimized_bisect.run(k, N=N, mean_interarrival=MEAN_INTERARRIVAL, mean_size=MEAN_SIZE, seed=SEED)
    bisect_times.append(t_bisect)

print(f"\n{'='*50}")
print(f"{'k':<6} {'Naive (s)':<14} {'Bisect (s)'}")
print(f"{'-'*6} {'-'*14} {'-'*10}")
for k, tn, tb in zip(k_values, naive_times, bisect_times):
    print(f"{k:<6} {tn:<14.4f} {tb:.4f}")

plt.figure()
plt.plot(k_values, naive_times,  marker='o', label='Naive')
plt.plot(k_values, bisect_times, marker='o', label='Bisect')
plt.xlabel('k')
plt.ylabel('Wall-clock time (seconds)')
plt.title(f'Naive vs Bisect Runtime  |  N={N}, seed={SEED}, λ=1/{MEAN_INTERARRIVAL}, μ=1/{MEAN_SIZE}')
plt.legend()
plt.tight_layout()
plt.show()
