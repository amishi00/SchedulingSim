import argparse
import matplotlib.pyplot as plt
 
# ── Shared simulation parameters ─────────────────────────────────────────────
N                 = 200   # number of jobs
MEAN_INTERARRIVAL = 1.0   # mean inter-arrival time
MEAN_SIZE         = 3.0   # mean job size
SEED              = 3     # random seed
# ─────────────────────────────────────────────────────────────────────────────
 
import naive
import unoptimized_bisect
import bisect_np
 
_parser = argparse.ArgumentParser(description="Run naive, bisect, and bisect+numpy "
                                              "for k=1..max_k and plot runtimes.")
_parser.add_argument("max_k", type=int, help="Upper bound for k (inclusive)")
_args = _parser.parse_args()
 
k_values     = list(range(1, _args.max_k + 1))
naive_times  = []
bisect_times = []
np_times     = []
 
for k in k_values:
    print(f"\n{'='*50}")
    print(f"[Naive] k={k}")
    t = naive.run(k, N=N, mean_interarrival=MEAN_INTERARRIVAL, mean_size=MEAN_SIZE, seed=SEED)
    naive_times.append(t)
 
    print(f"\n[Bisect] k={k}")
    t = unoptimized_bisect.run(k, N=N, mean_interarrival=MEAN_INTERARRIVAL, mean_size=MEAN_SIZE, seed=SEED)
    bisect_times.append(t)
 
    print(f"\n[Bisect+NumPy] k={k}")
    t = bisect_np.run(k, N=N, mean_interarrival=MEAN_INTERARRIVAL, mean_size=MEAN_SIZE, seed=SEED)
    np_times.append(t)
 
print(f"\n{'='*50}")
print(f"{'k':<6} {'Naive (s)':<14} {'Bisect (s)':<14} {'Bisect+NP (s)'}")
print(f"{'-'*6} {'-'*14} {'-'*14} {'-'*13}")
for k, tn, tb, tnp in zip(k_values, naive_times, bisect_times, np_times):
    print(f"{k:<6} {tn:<14.4f} {tb:<14.4f} {tnp:.4f}")
 
plt.figure()
plt.plot(k_values, naive_times,  marker='o', label='Naive')
plt.plot(k_values, bisect_times, marker='o', label='Bisect')
plt.plot(k_values, np_times,     marker='o', label='Bisect + NumPy')
plt.xlabel('k')
plt.ylabel('Wall-clock time (seconds)')
plt.title(f'Naive vs Bisect vs Bisect+NumPy  |  N={N}, seed={SEED}, λ=1/{MEAN_INTERARRIVAL}, μ=1/{MEAN_SIZE}')
plt.legend()
plt.tight_layout()
plt.show()
