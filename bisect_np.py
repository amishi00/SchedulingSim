import argparse
import piecewise_np
import job_queue
import numpy as np
import math
import time
from bisect import bisect_left
import cProfile
import pstats
from pstats import SortKey
 
EPS = 1e-6
 
 
def load_jobs_from_file(path):
    jobs = []
 
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
 
            if not line:
                continue
 
            arrival_str, size_str = line.split(",")
            arrival_time = float(arrival_str)
            size = float(size_str)
 
            jobs.append(job_queue.Job(id=i, arrival_time=arrival_time, size=size, age=0.0, remaining=size, start=-1.0, end=0.0))
 
            if i % 100 == 0:
                print(f"Loaded {i} jobs")
 
    return jobs
 
 
def total_time_to_rank(rank, rank_index, jobs):
    """
    Vectorized version: one rank_inv_batch call over all active jobs.
    Returns (total_time, list_of_(job, time_to_complete)_for_those_that_finish).
    """
    if not jobs:
        return 0.0, []
 
    ages = np.fromiter((j.age for j in jobs), dtype=np.float64, count=len(jobs))
    sizes = np.fromiter((j.size for j in jobs), dtype=np.float64, count=len(jobs))
    remaining = sizes - ages
 
    # Skip already-completed jobs (remaining <= EPS) by giving them 0 contribution.
    active_mask = remaining > EPS
    if not active_mask.any():
        return 0.0, []
 
    target_ages = rank_index.rank_inv_batch(ages, rank)
    time_needed = target_ages - ages
    time_to_complete = remaining
 
    # A job "completes en route" if time_to_complete <= time_needed + EPS.
    completes_mask = active_mask & (time_to_complete <= time_needed + EPS)
    advances_mask = active_mask & ~completes_mask
 
    total = float(time_to_complete[completes_mask].sum()
                  + time_needed[advances_mask].sum())
 
    # Build the completed list (small, so a Python loop over the mask is fine).
    completed = [(jobs[i], float(time_to_complete[i]))
                 for i in np.where(completes_mask)[0]]
 
    return total, completed
 
 
def rank_inv_completion(job, target_rank, rank_index):
    remaining = job.size - job.age
    age = job.age
 
    if rank_index.apply_rank(age) >= target_rank - EPS:
        return age
 
    target_age = rank_index.rank_inv(age, target_rank)
 
    if target_age <= age + EPS:
        return None
 
    if target_age - age >= remaining - EPS:
        return None
 
    return target_age
 
 
def advance_lowest_jobs(remaining, jobs, rank_index, result_ages, t_now, completed,
                       sticky_job=None):
    """
    Vectorized: result_ages is treated as an ndarray internally. The
    per-iteration "find lowest-rank active job" uses one apply_rank_batch
    call plus np.lexsort/argmin. Returns result_ages as a list to preserve
    the original caller contract.
    """
    n = len(jobs)
    # Work with an ndarray internally; convert back at exit.
    ages_arr = np.asarray(result_ages, dtype=np.float64)
    sizes_arr = np.fromiter((j.size for j in jobs), dtype=np.float64, count=n)
    arrivals_arr = np.fromiter((j.arrival_time for j in jobs), dtype=np.float64, count=n)
    ids_arr = np.fromiter((j.id for j in jobs), dtype=np.int64, count=n)

    pos_by_objid = {id(j): k for k, j in enumerate(jobs)}
 
    def _return(stk):
        return ages_arr.tolist(), remaining, t_now, completed, stk
 
    while remaining > EPS:
        if sticky_job is not None:
            i = pos_by_objid.get(id(sticky_job))
            if i is None:
                sticky_job = None
                continue
            job = jobs[i]
            age = float(ages_arr[i])
            if sizes_arr[i] - age <= EPS:
                sticky_job = None
                continue
        else:
            # Vectorized candidate selection.
            active = sizes_arr - ages_arr > EPS
            if not active.any():
                break
 
            # apply_rank over all jobs at once, then pick the lex-min
            # (rank, arrival, id) among active ones.
            ranks_arr = rank_index.apply_rank_batch(ages_arr)
            # Mask out inactive jobs by setting their sort keys to +inf.
            BIG = np.inf
            rk = np.where(active, ranks_arr, BIG)
            ar = np.where(active, arrivals_arr, BIG)
            iid = np.where(active, ids_arr.astype(np.float64), BIG)
 
            # lexsort uses the LAST key as primary; we want primary=rank,
            # secondary=arrival, tertiary=id. So pass in reverse-priority order.
            order = np.lexsort((iid, ar, rk))
            i = int(order[0])
            job = jobs[i]
            age = float(ages_arr[i])
 
        next_age = rank_index.next_breakpoint_after(age)
        if next_age == math.inf:
            dt = min(job.size - age, remaining)
        else:
            dt = min(job.size - age, next_age - age, remaining)
 
        if dt <= EPS:
            break
 
        ages_arr[i] += dt
        remaining -= dt
        t_now += dt
 
        if ages_arr[i] >= job.size - EPS:
            ages_arr[i] = job.size
            completed.append((job, t_now))
            return _return(None)
 
        if next_age != math.inf and abs(float(ages_arr[i]) - next_age) <= EPS:
            sticky_job = None
        else:
            sticky_job = job
 
    return _return(sticky_job)
 
 
def find_rank_after_time(T, old_rank_data, rank_index, jobs, t_now=0.0,
                         sticky_job=None):
    """
    Run the lowest-rank-first scheduler for time T. Bisection fast-forwards
    through breakpoints when no job is mid-segment. When a job is mid-segment
    (sticky_job set), the bisection is skipped and use
    advance_lowest_jobs which honors the lock
    """
    active_jobs = [j for j in jobs if j.size - j.age > EPS]
    if not active_jobs:
        return 0, [j.age for j in jobs], [], t_now, sticky_job
 
    if sticky_job is not None:
        result_ages = [job.age for job in jobs]
        completed = []
        result_ages, _, t_now, completed, sticky_job = advance_lowest_jobs(
            T, jobs, rank_index, result_ages, t_now, completed, sticky_job
        )
        return 0, result_ages, completed, t_now, sticky_job
 
    min_rank = min(rank_index.apply_rank(job.age) for job in active_jobs)
    rank_data = rank_index.rank_data
    all_ranks = rank_index.unique_ranks
 
    completed = []
 
    if not all_ranks:
        result_ages = [job.age for job in jobs]
        result_ages, _, t_now, completed, sticky_job = advance_lowest_jobs(
            T, jobs, rank_index, result_ages, t_now, completed, sticky_job
        )
        return min_rank, result_ages, completed, t_now, sticky_job
 
    lo, hi = 0, len(all_ranks) - 1
 
    while hi - lo > 1:
        mid = (lo + hi + 1) // 2
        t, completions = total_time_to_rank(all_ranks[mid], rank_index, jobs)
        if t > T + EPS:
            hi = mid
        elif completions:
            hi = mid
        else:
            lo = mid
 
    target_rank = all_ranks[lo]
    closest_time, completed = total_time_to_rank(target_rank, rank_index, jobs)
    t_now += closest_time
 
    if jobs:
        ages_np = np.fromiter((j.age for j in jobs), dtype=np.float64, count=len(jobs))
        sizes_np = np.fromiter((j.size for j in jobs), dtype=np.float64, count=len(jobs))
        remaining_np = sizes_np - ages_np
 
    
        curr_ranks = rank_index.apply_rank_batch(ages_np)
        already_at = curr_ranks >= target_rank - EPS
 
        target_ages = rank_index.rank_inv_batch(ages_np, target_rank)
 
        no_progress = target_ages <= ages_np + EPS
        would_complete = (target_ages - ages_np) >= (remaining_np - EPS)
        returns_none = (~already_at) & (no_progress | would_complete)
 
        result_ages_np = np.where(already_at, ages_np,
                          np.where(returns_none, sizes_np, target_ages))
        result_ages = result_ages_np.tolist()
    else:
        result_ages = []
 
    remaining = T - closest_time
 
    if remaining > EPS:
        result_ages, _, t_now, completed, sticky_job = advance_lowest_jobs(
            remaining, jobs, rank_index, result_ages, t_now, completed, sticky_job
        )
 
    return target_rank, result_ages, completed, t_now, sticky_job
 
 
def sim_arrival(n, rank_data, mean_interarrival, mean_size, seed=0):
    np.random.seed(seed)
 
    _t0 = time.perf_counter()
 
    time_passed = 0.0
    current_jobs = []
    all_jobs = []
    jobs_generated = 0
    job_id = 1
 
    print("Starting simulation...", flush=True)
 
 
    rank_index = piecewise_np.RankIndex(list(rank_data))
    next_arrival_time = np.random.exponential(mean_interarrival)
 
 
    running = None
 
    def pick_running_job(jobs):
        candidates = [j for j in jobs if j.size - j.age > EPS]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda j: (rank_index.apply_rank(j.age), j.arrival_time, j.id),
        )
 
    while True:
        next_arrival = next_arrival_time if jobs_generated < n else float("inf")
 
        if running is None or running not in current_jobs:
            running = pick_running_job(current_jobs)
 
        if running is not None:
            next_bp = rank_index.next_breakpoint_after(running.age)
            remaining_work = running.size - running.age
            if next_bp == math.inf:
                seg_dt = remaining_work
            else:
                seg_dt = min(next_bp - running.age, remaining_work)
            segment_end = time_passed + seg_dt
        else:
            segment_end = time_passed
 
        boundary = max(next_arrival, segment_end)
 
        if current_jobs and (boundary == float("inf") or time_passed < boundary - EPS):
            if boundary == float("inf"):
                budget = sum(max(0.0, j.size - j.age) for j in current_jobs) + 1.0
            else:
                budget = boundary - time_passed
 
            sticky_job = running
 
            _, result_ages, completed, t_now, returned_sticky = find_rank_after_time(
                budget, list(rank_data), rank_index, current_jobs,
               time_passed, sticky_job
            )
 
            for i, j in enumerate(current_jobs):
                if j.end > 0:
                    continue
                j.age = result_ages[i]
                j.remaining = max(0.0, j.size - j.age)
 
            for job_obj, t_finish in completed:
                if job_obj.end == 0:
                    job_obj.end = t_finish
                    job_obj.age = job_obj.size
                    job_obj.remaining = 0.0
 
            if t_now <= time_passed + EPS:
                break
 
            time_passed = t_now
 
            # Carry forward the sticky running pointer.
            running = returned_sticky
 
            current_jobs = [j for j in current_jobs if j.size - j.age > EPS]
 
            if running is not None and running not in current_jobs:
                running = None
 
            if completed:
                continue
        else:
            if boundary < float("inf") and time_passed < boundary - EPS:
                time_passed = boundary
 
       
        while jobs_generated < n and next_arrival_time <= time_passed + EPS:
            size = np.random.exponential(mean_size)
            job = job_queue.Job(
                id=job_id,
                arrival_time=next_arrival_time,
                size=size,
                age=0.0,
                remaining=size,
                start=-1.0,
                end=0.0,
            )
            current_jobs.append(job)
            all_jobs.append(job)
            jobs_generated += 1
            job_id += 1
 
            if jobs_generated < n:
                next_arrival_time = next_arrival_time + np.random.exponential(mean_interarrival)
            else:
                next_arrival_time = float("inf")
 
        if not current_jobs and jobs_generated >= n:
            break
 
    total_delay = 0.0
    completed_count = 0
 
    for j in all_jobs:
        if j.end > 0 and j.size - j.age <= EPS:
            total_delay += j.end - j.arrival_time
            completed_count += 1
 
    _t1 = time.perf_counter()
    wall_clock = _t1 - _t0
 
    print(f"Completed: {completed_count}/{len(all_jobs)}", flush=True)
    if completed_count:
        print(f"Mean Delay: {total_delay / completed_count:.4f}", flush=True)
    print(f"Wall-clock time: {wall_clock:.4f}s", flush=True)
 
    return all_jobs, wall_clock
    
rank_data = [(i/10, i/10) for i in range(500)]

def duplicate_step(k, rank_data):
    rd = sorted(rank_data, key=lambda x: x[0])
    return [(sum(rd[(i + j) // k][0] for j in range(k)) / k, rd[i // k][1]) for i in range(k * len(rd) - k + 1)]
def run(k, N=200, mean_interarrival=1.0, mean_size=3.0, seed=0, rank_data=rank_data):
    rank_k = duplicate_step(k, rank_data)
    _, wall_clock = sim_arrival(N, rank_k, mean_interarrival, mean_size, seed)
    return wall_clock
 
 
if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument("k", type=int)
    _args = _parser.parse_args()
    run(_args.k)