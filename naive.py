import argparse
import time
import job_queue
import numpy as np

EPS = 1e-6

rank_data = [(i/10, i/10) for i in range(500)]


def duplicate_step(k, rank_data):
    rd = sorted(rank_data, key=lambda x: x[0])
    return [(sum(rd[(i + j) // k][0] for j in range(k)) / k, rd[i // k][1]) for i in range(k * len(rd) - k + 1)]


def get_rank(age, rank_data):
    """Age -> Rank (last breakpoint at or before age)."""
    curr_rank = rank_data[0][1]
    for a, r in rank_data:
        if age >= a - EPS:
            curr_rank = r
        else:
            break
    return curr_rank


def next_breakpoint_after(age, rank_data):
    """Next breakpoint age strictly greater than `age`. Returns inf if none."""
    for a, _ in rank_data:
        if a > age + EPS:
            return a
    return float('inf')


def run(k, N=200, mean_interarrival=1.0, mean_size=3.0, seed=0, rank_data=rank_data):
    rank_k = duplicate_step(k, rank_data)

    def pick_running(jobs):
        return min(jobs, key=lambda j: (get_rank(j.age, rank_k), j.arrival_time, j.id))

    np.random.seed(seed)
    jobs = []
    _arrival = 0.0
    for _i in range(1, N + 1):
        _arrival += np.random.exponential(mean_interarrival)
        _size = np.random.exponential(mean_size)
        jobs.append(job_queue.Job(
            id=_i,
            arrival_time=_arrival,
            size=_size,
            age=0.0,
            remaining=_size,
            start=-1.0,
            end=-1.0,
        ))

    jobs_sorted = sorted(jobs, key=lambda j: (j.arrival_time, j.id))

    print("Starting simulation...")
    _t0 = time.perf_counter()

    t = 0.0
    S = []
    next_idx = 0
    running = None

    while next_idx < len(jobs_sorted) or S:
        if not S:
            next_arrival_time = jobs_sorted[next_idx].arrival_time
            if t < next_arrival_time:
                t = next_arrival_time
            while next_idx < len(jobs_sorted) and jobs_sorted[next_idx].arrival_time <= t + EPS:
                S.append(jobs_sorted[next_idx])
                next_idx += 1
            running = None
            continue

        if running is None or running not in S:
            running = pick_running(S)
        if running.start < 0:
            running.start = t

        dt_bp = next_breakpoint_after(running.age, rank_k) - running.age
        dt_complete = running.remaining
        if next_idx < len(jobs_sorted):
            dt_arrival = jobs_sorted[next_idx].arrival_time - t
        else:
            dt_arrival = float('inf')

        dt = min(dt_bp, dt_complete, dt_arrival)

        running.age += dt
        running.remaining -= dt
        t += dt

        if dt_complete <= dt_bp + EPS and dt_complete <= dt_arrival + EPS:
            running.end = t
            running.age = running.size
            running.remaining = 0.0
            S.remove(running)
            running = None
        elif dt_arrival <= dt_bp + EPS:
            while next_idx < len(jobs_sorted) and jobs_sorted[next_idx].arrival_time <= t + EPS:
                S.append(jobs_sorted[next_idx])
                next_idx += 1
        else:
            running = None

    total_delay = 0.0
    completed = 0
    for j in jobs:
        if j.end >= 0:
            total_delay += j.end - j.arrival_time
            completed += 1

    _t1 = time.perf_counter()
    print(f"\nCompleted: {completed}/{len(jobs)}")
    if completed:
        print(f"Mean Delay: {total_delay / completed:.4f}")
    print(f"Wall-clock time: {_t1 - _t0:.4f}s")

    return _t1 - _t0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument("k", type=int)
    _args = _parser.parse_args()
    run(_args.k)
