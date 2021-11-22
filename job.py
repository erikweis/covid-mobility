from __future__ import annotations


class Job:

    def __init__(self,
        idx:int,
        salary: int | float,
        location: None | tuple[int, int] = None
    ):
        self.idx = idx
        self.salary = salary
        self.location = location

