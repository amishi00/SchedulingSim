
"""
Numpy-vectorized version of piecewiseNEW.
"""
 
from bisect import bisect_left, bisect_right
import math
import numpy as np
 
EPS = 1e-6
 
 
class RankIndex:
    def __init__(self, rank_data):
        self.rank_data = rank_data

        self.ages = np.array([a for a, r in rank_data], dtype=np.float64)
        self.ranks = np.array([r for a, r in rank_data], dtype=np.float64)
        self.unique_ranks = np.unique(self.ranks)

        self.unique_ranks_list = self.unique_ranks.tolist()
 
        B = self.ranks.shape[0]
        R = self.unique_ranks.shape[0]

        self._next_ge = np.full((B + 1, R), -1, dtype=np.int64)
 
        if B > 0:
            # eligible[i, q] = True if ranks[i] >= unique_ranks[q] - EPS
            eligible = self.ranks[:, None] >= (self.unique_ranks[None, :] - EPS)
 
           
            INF = B + 1  # any value larger than any valid index
            idx = np.where(eligible, np.arange(B)[:, None], INF)
            rev_min = np.minimum.accumulate(idx[::-1, :], axis=0)[::-1, :]
 
            # Where the propagated value is still INF, there's no such j (-> -1).
            self._next_ge[:B, :] = np.where(rev_min < B, rev_min, -1)
            # _next_ge[B, :] stays -1 (== None semantics)
 
     
        self.next_ge = [
            [None if v == -1 else int(v) for v in row]
            for row in self._next_ge
        ]

 
    def apply_rank(self, age):
        idx = int(np.searchsorted(self.ages, age + EPS, side='right')) - 1
        if idx < 0:
            return 0
        return float(self.ranks[idx])
 
    def rank_inv(self, age, target_rank):
        if self.apply_rank(age) >= target_rank - EPS:
            return age
        start_idx = int(np.searchsorted(self.ages, age + EPS, side='right'))
        if start_idx >= self.ages.shape[0]:
            return age
        q = int(np.searchsorted(self.unique_ranks, target_rank - EPS, side='left'))
        if q >= self.unique_ranks.shape[0]:
            return age
        j = int(self._next_ge[start_idx, q])
        if j < 0:
            return age
        return float(self.ages[j])
 
    def next_breakpoint_after(self, age):
        idx = int(np.searchsorted(self.ages, age + EPS, side='right'))
        if idx >= self.ages.shape[0]:
            return math.inf
        return float(self.ages[idx])
 
    # ── batch API (this is where the speedup lives) ──────────────────────
 
    def apply_rank_batch(self, ages):
        """
        ages: 1-D ndarray.
        Returns ndarray of the rank associated with each age (last breakpoint
        at or before age + EPS); returns 0.0 for ages before the first bp.
        """
        ages = np.asarray(ages, dtype=np.float64)
        idx = np.searchsorted(self.ages, ages + EPS, side='right') - 1
        # Where idx < 0, rank is 0.
        safe_idx = np.where(idx < 0, 0, idx)
        out = self.ranks[safe_idx]
        out = np.where(idx < 0, 0.0, out)
        return out
 
    def rank_inv_batch(self, ages, target_rank):
        """
        Vectorized inverse: for each age in `ages`, find the smallest stored
        age x >= that age whose rank >= target_rank. Returns the input age
        unchanged when the current rank already satisfies the target or no
        such x exists -- same semantics as the scalar method.
        """
        ages = np.asarray(ages, dtype=np.float64)
        out = ages.copy()
 
        # Already satisfied?
        curr = self.apply_rank_batch(ages)
        already_ok = curr >= target_rank - EPS
 
        # Which column of next_ge do we need?
        q = int(np.searchsorted(self.unique_ranks, target_rank - EPS, side='left'))
        if q >= self.unique_ranks.shape[0]:
            # No stored rank is large enough; leave everything as-is.
            return out
 
        # Indices into self.ages corresponding to "first breakpoint strictly
        # after each input age".
        start = np.searchsorted(self.ages, ages + EPS, side='right')
 
        # For rows where start is in range and not already_ok, look up next_ge.
        in_range = start < self.ages.shape[0]
        do_lookup = (~already_ok) & in_range
 
        if do_lookup.any():
            # Safe index for gather (clip out-of-range to 0; we'll mask after).
            safe_start = np.where(do_lookup, start, 0)
            j = self._next_ge[safe_start, q]  # shape == ages.shape, int64
            valid = do_lookup & (j >= 0)
            # gather target ages where valid, else keep current
            safe_j = np.where(valid, j, 0)
            target_ages = self.ages[safe_j]
            out = np.where(valid, target_ages, out)
 
        return out
 
    def next_breakpoint_after_batch(self, ages):
       
        ages = np.asarray(ages, dtype=np.float64)
        idx = np.searchsorted(self.ages, ages + EPS, side='right')
        out_of_range = idx >= self.ages.shape[0]
        safe_idx = np.where(out_of_range, 0, idx)
        out = self.ages[safe_idx]
        out = np.where(out_of_range, np.inf, out)
        return out
 
 

 
def apply_rank_func(rank_data, age):
    return RankIndex(rank_data).apply_rank(age)
 
 
def rank_inv(age, target_rank, rank_data):
    return RankIndex(rank_data).rank_inv(age, target_rank)
 
 
def time_to_rank(jobs, target_rank, rank_data, update_ages=False):
    rank_index = RankIndex(rank_data)
    ages = np.array([j.age for j in jobs], dtype=np.float64)
    targets = rank_index.rank_inv_batch(ages, target_rank)
    dt = float((targets - ages).sum())
    if update_ages:
        for j, ta in zip(jobs, targets):
            j.age = float(ta)
    return dt