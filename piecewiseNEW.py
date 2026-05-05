from bisect import bisect_left, bisect_right
import math

EPS = 1e-6


class RankIndex:
    """
    Precomputed index for rank_data.

    Lets us answer:
      apply_rank(age)
      rank_inv(age, target_rank)
      next_breakpoint_after(age)

    without scanning rank_data from scratch every time.
    """

    def __init__(self, rank_data):
        self.rank_data = rank_data
        self.ages = [a for a, r in rank_data]
        self.ranks = [r for a, r in rank_data]
        self.unique_ranks = sorted(set(self.ranks))

        B = len(self.ranks)
        R = len(self.unique_ranks)

        # next_ge[i][q] = first index j >= i
        # such that ranks[j] >= unique_ranks[q]
        self.next_ge = [[None] * R for _ in range(B + 1)]

        for q in range(R):
            self.next_ge[B][q] = None

        for i in range(B - 1, -1, -1):
            for q, target_rank in enumerate(self.unique_ranks):
                if self.ranks[i] >= target_rank - EPS:
                    self.next_ge[i][q] = i
                else:
                    self.next_ge[i][q] = self.next_ge[i + 1][q]

    def apply_rank(self, age):
        idx = bisect_right(self.ages, age + EPS) - 1

        if idx < 0:
            return 0

        return self.ranks[idx]

    def rank_inv(self, age, target_rank):
    
        if self.apply_rank(age) >= target_rank - EPS:
            return age

        # First breakpoint strictly after age.
        start_idx = bisect_right(self.ages, age + EPS)

        if start_idx >= len(self.ages):
            return age

        # Find smallest stored rank value >= target_rank.
        q = bisect_left(self.unique_ranks, target_rank - EPS)

        if q >= len(self.unique_ranks):
            return age

        j = self.next_ge[start_idx][q]

        if j is None:
            return age

        return self.ages[j]

    def next_breakpoint_after(self, age):
        idx = bisect_right(self.ages, age + EPS)

        if idx >= len(self.ages):
            return math.inf

        return self.ages[idx]


def apply_rank_func(rank_data, age):
   
    rank_index = RankIndex(rank_data)
    return rank_index.apply_rank(age)


def rank_inv(age, target_rank, rank_data):
   
    rank_index = RankIndex(rank_data)
    return rank_index.rank_inv(age, target_rank)


def time_to_rank(jobs, target_rank, rank_data, update_ages=False):
   
    rank_index = RankIndex(rank_data)
    dt = 0.0

    for j in jobs:
        target_age = rank_index.rank_inv(j.age, target_rank)
        dt += target_age - j.age

        if update_ages:
            j.age = target_age

    return dt