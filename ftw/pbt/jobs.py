import abc
from acme import core as acme_core


class Job(acme_core.Worker):

    _job_pool = None

    def attach_job_pool(self, job_pool):
        self._job_pool = job_pool

    @abc.abstractmethod
    def run(self):
        """Starts the job."""
