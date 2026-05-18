import numpy as np
import job_queue
import time 

EPS = 1e-6

def run(N=200, mean_interarrival=1.0, mean_size=3.0, seed=0):
    np.random.seed(seed)
    jobs = []
    arrival = 0.0
    for i in range(1, N + 1):
        arrival += np.random.exponential(mean_interarrival)
        size = np.random.exponential(mean_size)
        jobs.append(job_queue.Job(
            id=i,
            arrival_time=arrival,
            size=size,
            age=0.0,
            remaining=size,
            start=-1.0,
            end=-1.0,
        ))

    jobs_sorted = sorted(jobs, key=lambda j: (j.arrival_time, j.id))
    print("Starting simulation...")
    t0 = time.perf_counter()
    t = 0.0

    S = []
    next_idx = 0

    while next_idx < len(jobs_sorted) or S:
        if not S:
            t = jobs_sorted[next_idx].arrival_time
            while next_idx < len(jobs_sorted) and jobs_sorted[next_idx].arrival_time <= t + EPS:
                S.append(jobs_sorted[next_idx])
                next_idx += 1
            continue

        running = min(S, key=lambda j: (j.age, j.arrival_time, j.id))
        if running.start < 0:
            running.start = t

        dt_complete = running.remaining
        dt_arrival = jobs_sorted[next_idx].arrival_time - t if next_idx < len(jobs_sorted) else float('inf')

        # next time a different job becomes the LAS winner
        others = [j for j in S if j is not running]
        dt_preempt = min((j.age - running.age for j in others if j.age > running.age + EPS), default=float('inf'))

        dt = min(dt_complete, dt_arrival, dt_preempt)

        running.age += dt
        running.remaining -= dt
        t += dt

        if running.remaining <= EPS:
            running.end = t
            S.remove(running)
        elif dt_arrival <= dt_complete - EPS and dt_arrival <= dt_preempt - EPS:
            while next_idx < len(jobs_sorted) and jobs_sorted[next_idx].arrival_time <= t + EPS:
                S.append(jobs_sorted[next_idx])
                next_idx += 1
    t1 = time.perf_counter()

    total_delay = sum(j.end - j.arrival_time for j in jobs if j.end >= 0)
    completed = sum(1 for j in jobs if j.end >= 0)
    print(f"Completed: {completed}/{len(jobs)}")
    if completed:
        print(f"Mean Delay: {total_delay / completed:.4f}")
    print(f"Wall-clock time: {t1 - t0:.4f}s")

    return t1 - t0
