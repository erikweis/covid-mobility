from job import Job
from location import Location

def test_job_initialization():

    loc = Location(3,4,10)
    j = Job(3,location = loc,salary=50000)
    
    assert j.idx == 3
    assert j.location == loc
    assert j.salary == 50000
