import argparse
import piecewiseNEW
import job_queue
from snapshot_visualizer import SnapshotVisualizer
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
    total = 0.0
    completed = []

    for job in jobs:
        if job.size - job.age <= EPS:
            continue
        target_age = rank_index.rank_inv(job.age, rank)
        time_needed = target_age - job.age
        time_to_complete = job.size - job.age
        if time_to_complete <= time_needed + EPS:
            total += time_to_complete
            completed.append((job, time_to_complete))
        else:
            total += time_needed

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

    sticky_job: if not None, the job-object that is currently mid-segment and
    must NOT be preempted until it crosses its next breakpoint or completes.
    Returns the job-object that is mid-segment when budget exhausts, or None if the last action
    was a breakpoint crossing or a completion.
    """
    while remaining > EPS:
        if sticky_job is not None:
            try:
                i = jobs.index(sticky_job)
            except ValueError:
                sticky_job = None
                continue
            job = jobs[i]
            age = result_ages[i]
            if job.size - age <= EPS:
                sticky_job = None
                continue
        else:
            candidates = []
            for k, (job, age) in enumerate(zip(jobs, result_ages)):
                if job.size - age <= EPS:
                    continue
                rank = rank_index.apply_rank(age)
                candidates.append((rank, job.arrival_time, job.id, k, job, age))
            if not candidates:
                break
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))
            _, _, _, i, job, age = candidates[0]

        next_age = rank_index.next_breakpoint_after(age)
        if next_age == math.inf:
            dt = min(job.size - age, remaining)
        else:
            dt = min(job.size - age, next_age - age, remaining)

        if dt <= EPS:
            break

        result_ages[i] += dt
        remaining -= dt
        t_now += dt

        if result_ages[i] >= job.size - EPS:
            result_ages[i] = job.size
            completed.append((job, t_now))
            return result_ages, remaining, t_now, completed, None

        if next_age != math.inf and abs(result_ages[i] - next_age) <= EPS:
            # Breakpoint crossing - lock releases for next iteration
            sticky_job = None
        else:
            # Mid-segment - this job is the new sticky target if budget runs out
            sticky_job = job

    return result_ages, remaining, t_now, completed, sticky_job


def find_rank_after_time(T, old_rank_data, rank_index, jobs, snapshot, t_now=0.0,
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
        snapshot.add_snapshot(result_ages.copy())
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
        snapshot.add_snapshot(result_ages.copy())
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

    result_ages = [
        job.size if (age := rank_inv_completion(job, target_rank, rank_index)) is None
        else age
        for job in jobs
    ]

    snapshot.add_snapshot(result_ages.copy())

    remaining = T - closest_time

    if remaining > EPS:
        result_ages, _, t_now, completed, sticky_job = advance_lowest_jobs(
            remaining, jobs, rank_index, result_ages, t_now, completed, sticky_job
        )
        snapshot.add_snapshot(result_ages.copy())

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


    rank_index = piecewiseNEW.RankIndex(list(rank_data))
    snapshot_visualizer = SnapshotVisualizer(rank_data, [])
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
                snapshot_visualizer, time_passed, sticky_job
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

        snapshot_visualizer.add_snapshot([j.age for j in current_jobs])

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

    return all_jobs, snapshot_visualizer, wall_clock
    
# rank_data = [(i/100,i/100)for i in range(100000)]
rank_data = [
    (0.0, 3.359),
    (0.1, 3.478),
    (0.2, 3.605),
    (0.3, 3.739),
    (0.4, 3.879),
    (0.5, 4.024),
    (0.6, 4.173),
    (0.7, 4.326),
    (0.8, 4.481),
    (0.9, 4.637),
    (1.0, 4.794),
    (1.1, 4.950),
    (1.2, 5.104),
    (1.3, 5.255),
    (1.4, 5.401),
    (1.5, 5.543),
    (1.6, 5.679),
    (1.7, 5.808),
    (1.8, 5.929),
    (1.9, 6.041),
    (2.0, 6.144),
    (2.1, 6.237),
    (2.2, 6.319),
    (2.3, 6.391),
    (2.4, 6.451),
    (2.5, 6.500),
    (2.6, 6.537),
    (2.7, 6.563),
    (2.8, 6.578),
    (2.9, 6.582),
    (3.0, 6.576),
    (3.1, 6.560),
    (3.2, 6.534),
    (3.3, 6.501),
    (3.4, 6.460),
    (3.5, 6.413),
    (3.6, 6.360),
    (3.7, 6.302),
    (3.8, 6.240),
    (3.9, 6.176),
    (4.0, 6.110),
    (4.1, 6.043),
    (4.2, 5.975),
    (4.3, 5.907),
    (4.4, 5.840),
    (4.5, 5.773),
    (4.6, 5.707),
    (4.7, 5.642),
    (4.8, 5.577),
    (4.9, 5.512),
    (5.0, 5.447),
    (5.1, 5.381),
    (5.2, 5.314),
    (5.3, 5.244),
    (5.4, 5.172),
    (5.5, 5.096),
    (5.6, 5.016),
    (5.7, 4.932),
    (5.8, 4.843),
    (5.9, 4.749),
    (6.0, 4.651),
    (6.1, 4.547),
    (6.2, 4.439),
    (6.3, 4.327),
    (6.4, 4.211),
    (6.5, 4.093),
    (6.6, 3.973),
    (6.7, 3.853),
    (6.8, 3.733),
    (6.9, 3.614),
    (7.0, 3.498),
    (7.1, 3.385),
    (7.2, 3.276),
    (7.3, 3.172),
    (7.4, 3.074),
    (7.5, 2.982),
    (7.6, 2.897),
    (7.7, 2.817),
    (7.8, 2.744),
    (7.9, 2.677),
    (8.0, 2.615),
    (8.1, 2.558),
    (8.2, 2.505),
    (8.3, 2.456),
    (8.4, 2.408),
    (8.5, 4.062),
    (8.6, 4.015),
    (8.7, 3.968),
    (8.8, 3.918),
    (8.9, 3.866),
    (9.0, 3.809),
    (9.1, 3.747),
    (9.2, 3.680),
    (9.3, 3.606),
    (9.4, 3.525),
    (9.5, 3.436),
    (9.6, 3.340),
    (9.7, 3.236),
    (9.8, 3.125),
    (9.9, 3.006),
    (10.0, 2.879),
    (10.1, 2.746),
    (10.2, 2.606),
    (10.3, 2.461),
    (10.4, 2.312),
    (10.5, 2.158),
    (10.6, 2.002),
    (10.7, 1.845),
    (10.8, 1.687),
    (10.9, 1.530),
    (11.0, 1.375),
    (11.1, 1.225),
    (11.2, 1.079),
    (11.3, 0.939),
    (11.4, 0.808),
    (11.5, 0.685),
    (11.6, 0.573),
    (11.7, 0.473),
    (11.8, 0.385),
    (11.9, 0.312),
    (12.0, 0.253),
    (12.1, 0.210),
    (12.2, 0.184),
    (12.3, 0.175),
    (12.4, 0.184),
    (12.5, 0.211),
    (12.6, 0.256),
    (12.7, 0.320),
    (12.8, 0.402),
    (12.9, 0.503),
    (13.0, 0.621),
    (13.1, 0.757),
    (13.2, 0.909),
    (13.3, 1.078),
    (13.4, 1.261),
    (13.5, 1.459),
    (13.6, 1.670),
    (13.7, 1.892),
    (13.8, 2.126),
    (13.9, 2.368),
    (14.0, 2.619),
    (14.1, 2.876),
    (14.2, 3.139),
    (14.3, 3.405),
    (14.4, 3.673),
    (14.5, 3.942),
    (14.6, 4.210),
    (14.7, 4.476),
    (14.8, 4.739),
    (14.9, 4.998),
    (15.0, 5.250),
    (15.1, 5.496),
    (15.2, 5.733),
    (15.3, 5.962),
    (15.4, 6.180),
    (15.5, 6.388),
    (15.6, 6.584),
    (15.7, 6.768),
    (15.8, 6.940),
    (15.9, 7.099),
    (16.0, 5.045),
    (16.1, 5.178),
    (16.2, 5.297),
    (16.3, 5.403),
    (16.4, 5.495),
    (16.5, 5.575),
    (16.6, 5.642),
    (16.7, 5.697),
    (16.8, 5.740),
    (16.9, 5.772),
    (17.0, 5.793),
    (17.1, 5.804),
    (17.2, 5.807),
    (17.3, 5.800),
    (17.4, 5.786),
    (17.5, 5.766),
    (17.6, 5.739),
    (17.7, 5.707),
    (17.8, 5.671),
    (17.9, 5.632),
    (18.0, 5.590),
    (18.1, 5.547),
    (18.2, 5.503),
    (18.3, 5.459),
    (18.4, 5.416),
    (18.5, 5.374),
    (18.6, 5.333),
    (18.7, 5.296),
    (18.8, 5.262),
    (18.9, 5.231),
    (19.0, 5.204),
    (19.1, 5.181),
    (19.2, 5.162),
    (19.3, 5.149),
    (19.4, 5.139),
    (19.5, 5.134),
    (19.6, 5.134),
    (19.7, 5.138),
    (19.8, 5.146),
    (19.9, 5.157),
    (20.0, 5.172),
    (20.1, 5.189),
    (20.2, 5.209),
    (20.3, 5.231),
    (20.4, 5.253),
    (20.5, 5.277),
    (20.6, 5.299),
    (20.7, 5.321),
    (20.8, 5.341),
    (20.9, 5.359),
    (21.0, 5.373),
    (21.1, 5.384),
    (21.2, 5.389),
    (21.3, 5.390),
    (21.4, 5.384),
    (21.5, 5.371),
    (21.6, 5.351),
    (21.7, 5.324),
    (21.8, 5.287),
    (21.9, 5.242),
    (22.0, 5.188),
    (22.1, 5.124),
    (22.2, 5.051),
    (22.3, 4.967),
    (22.4, 4.874),
    (22.5, 4.771),
    (22.6, 4.658),
    (22.7, 4.536),
    (22.8, 4.404),
    (22.9, 4.264),
    (23.0, 4.115),
    (23.1, 3.959),
    (23.2, 3.796),
    (23.3, 3.626),
    (23.4, 3.450),
    (23.5, 3.270),
    (23.6, 3.086),
    (23.7, 2.899),
    (23.8, 2.710),
    (23.9, 2.520),
    (24.0, 5.131),
    (24.1, 4.943),
    (24.2, 4.757),
    (24.3, 4.574),
    (24.4, 4.396),
    (24.5, 4.224),
    (24.6, 4.058),
    (24.7, 3.899),
    (24.8, 3.748),
    (24.9, 3.607),
    (25.0, 3.474),
    (25.1, 3.352),
    (25.2, 3.240),
    (25.3, 3.138),
    (25.4, 3.046),
    (25.5, 2.965),
    (25.6, 2.894),
    (25.7, 2.833),
    (25.8, 2.780),
    (25.9, 2.737),
    (26.0, 2.700),
    (26.1, 2.671),
    (26.2, 2.648),
    (26.3, 2.631),
    (26.4, 2.619),
    (26.5, 2.611),
    (26.6, 2.609),
    (26.7, 2.611),
    (26.8, 2.619),
    (26.9, 2.634),
    (27.0, 2.655),
    (27.1, 2.686),
    (27.2, 2.727),
    (27.3, 2.780),
    (27.4, 2.846),
    (27.5, 2.928),
    (27.6, 3.026),
    (27.7, 3.143),
    (27.8, 3.278),
    (27.9, 3.433),
    (28.0, 3.606),
    (28.1, 3.797),
    (28.2, 4.005),
    (28.3, 4.227),
    (28.4, 4.461),
    (28.5, 4.705),
    (28.6, 4.954),
    (28.7, 5.207),
    (28.8, 5.458),
    (28.9, 5.705),
    (29.0, 5.945),
    (29.1, 6.173),
    (29.2, 6.389),
    (29.3, 6.589),
    (29.4, 6.771),
    (29.5, 6.936),
    (29.6, 7.080),
    (29.7, 7.205),
    (29.8, 7.311),
    (29.9, 7.397),
    (30.0, 7.465),
    (30.1, 7.516),
    (30.2, 7.551),
    (30.3, 7.571),
    (30.4, 7.579),
    (30.5, 7.576),
    (30.6, 7.563),
    (30.7, 7.543),
    (30.8, 7.516),
    (30.9, 7.485),
    (31.0, 7.452),
    (31.1, 7.417),
    (31.2, 7.382),
    (31.3, 7.349),
    (31.4, 7.319),
    (31.5, 5.393),
    (31.6, 5.372),
    (31.7, 5.358),
    (31.8, 5.350),
    (31.9, 5.350),
    (32.0, 5.358),
    (32.1, 5.376),
    (32.2, 5.403),
    (32.3, 5.439),
    (32.4, 5.485),
    (32.5, 5.540),
    (32.6, 5.605),
    (32.7, 5.679),
    (32.8, 5.761),
    (32.9, 5.851),
    (33.0, 5.948),
    (33.1, 6.052),
    (33.2, 6.161),
    (33.3, 6.274),
    (33.4, 6.389),
    (33.5, 6.507),
    (33.6, 6.625),
    (33.7, 6.741),
    (33.8, 6.855),
    (33.9, 6.965),
    (34.0, 7.070),
    (34.1, 7.169),
    (34.2, 7.259),
    (34.3, 7.340),
    (34.4, 7.411),
    (34.5, 7.471),
    (34.6, 7.518),
    (34.7, 7.552),
    (34.8, 7.572),
    (34.9, 7.578),
    (35.0, 7.568),
    (35.1, 7.544),
    (35.2, 7.503),
    (35.3, 7.448),
    (35.4, 7.377),
    (35.5, 7.290),
    (35.6, 7.189),
    (35.7, 7.073),
    (35.8, 6.944),
    (35.9, 6.801),
    (36.0, 6.645),
    (36.1, 6.477),
    (36.2, 6.299),
    (36.3, 6.110),
    (36.4, 5.912),
    (36.5, 5.706),
    (36.6, 5.493),
    (36.7, 5.274),
    (36.8, 5.049),
    (36.9, 4.821),
    (37.0, 4.591),
    (37.1, 4.358),
    (37.2, 4.125),
    (37.3, 3.892),
    (37.4, 3.662),
    (37.5, 3.434),
    (37.6, 3.209),
    (37.7, 2.990),
    (37.8, 2.776),
    (37.9, 2.570),
    (38.0, 2.371),
    (38.1, 2.181),
    (38.2, 2.000),
    (38.3, 1.830),
    (38.4, 1.670),
    (38.5, 1.523),
    (38.6, 1.388),
    (38.7, 1.265),
    (38.8, 1.156),
    (38.9, 1.061),
    (39.0, 0.979),
    (39.1, 0.912),
    (39.2, 0.859),
    (39.3, 0.820),
    (39.4, 0.795),
    (39.5, 0.784),
    (39.6, 0.787),
    (39.7, 0.803),
    (39.8, 0.833),
    (39.9, 0.874),
    (40.0, 0.927),
]
def duplicate_step(k, rank_data):
    rd = sorted(rank_data, key=lambda x: x[0])
    return [(sum(rd[(i + j) // k][0] for j in range(k)) / k, rd[i // k][1]) for i in range(k * len(rd) - k + 1)]
def run(k, N=200, mean_interarrival=1.0, mean_size=3.0, seed=0):
    rank_k = duplicate_step(k, rank_data)
    _, _, wall_clock = sim_arrival(N, rank_k, mean_interarrival, mean_size, seed)
    return wall_clock


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument("k", type=int)
    _args = _parser.parse_args()
    run(_args.k)