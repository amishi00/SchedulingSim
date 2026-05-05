class Job:
    def __init__(self, id: int, arrival_time:float, size:float, age:float, remaining:float, start:float, end:float):
        self.id = id
        self.arrival_time = arrival_time
        self.size = size
        self.age = age
        self.remaining = remaining
        self.start = start
        self.end = end

    def get_age(job):
        return job.age
    def get_remaining(job):
        return job.remaining


class SimpleJob:
    def __init__(self, id: int, age: float):
        self.id = id
        self.age = age


class Queue:
    def __init__(self, policy_type:str, curr_size: int, waiting: list[Job], running:Job, size_limit:float, smallest_age_job:Job):
        self.policy_type = policy_type
        self.curr_size = curr_size
        self.waiting = []
        self.running = None
        self.size_limit=size_limit
        self.smallest_age_job = None
