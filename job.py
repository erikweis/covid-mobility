from __future__ import annotations


class Job:

    def __init__(self,
        idx:int,
        salary: int | float,
        location: None | tuple[int, int] = None,
        remote_status: bool = False
    ):
        self.idx = idx
        self.salary = salary
        self.location = location
        self.remote_status = remote_status






