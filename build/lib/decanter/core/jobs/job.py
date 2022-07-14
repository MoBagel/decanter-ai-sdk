# pylint: disable=invalid-name
"""This module defines jobs to handle all kinds of actions.

Handle the timeing of task's execution, and stores the result
of task in its attributes.
"""
import abc
import asyncio
import logging

from decanter.core.core_api import CoreAPI
from decanter.core.extra import CoreStatus
from decanter.core.extra.decorators import block_method


logger = logging.getLogger(__name__)


class Job:
    """Handle he timeing of task's execution.

    Every Job will have to wait for all the other Jobs in jobs list to be
    success before it starts to run its task.

    Attributes:
        id (str): ObjectId in 24 hex digits.
        status (str): Job status.
        result (Depends on type of Job.): Job result.
        task (:class:`~decanter.core.jobs.task.Task`): Task to be run by Job.
        jobs (list(:class:`~decanter.core.jobs.job.Job`)): List of
            :class:`~decanter.core.jobs.job.Job` that needs to be waited before
            running the task.
        name (str): Name to track Job progress.
        core_service (:class:`~decanter.core.core_api.api.CoreAPI`): Handle the
            calling of api.
    """
    def __init__(self, task, jobs=None, name=None):
        self.id = None
        self.status = CoreStatus.PENDING
        self.result = None
        self.task = task
        self.jobs = jobs
        self.name = name
        self.core_service = CoreAPI()

    def is_done(self):
        """
        Return:
            bool: True for task in `DONE_STATUS`, False otherwise.
        """
        return self.status in CoreStatus.DONE_STATUS

    def not_done(self):
        """
        Return:
            bool: True for task not in `DONE_STATUS`, False otherwise.
        """
        return not self.is_done()

    def is_success(self):
        """
        Return:
            bool: True for success, False otherwise.
        """
        return self.status == CoreStatus.DONE and self.result is not None

    def is_fail(self):
        """
        Return:
            bool: True for failed, False otherwise.
        """
        return self.status in CoreStatus.FAIL_STATUS or \
            (self.status == CoreStatus.DONE and self.result is None)

    async def wait(self):
        """Mange the Execution of task.

        A python coroutine be wapped as task and put in event loop once a Job
        is created.  When the event loop starts to run, wait function will
        wait for prerequired jobs in self.jobs list to be done, and continue
        to execute running the task if all prerequired jobs is successful.

        The coroutine will be done when the Job fininsh gettng the result from
        task.
        """
        if self.jobs is not None and self.status not in CoreStatus.DONE_STATUS:

            while not all(job.is_done() for job in self.jobs) and \
                    not any(job.is_fail() for job in self.jobs):
                ll = [job.status for job in self.jobs]
                logger.debug(
                    '[Job] \'%s\' waiting %s pre required jobs. jobs status: %s',
                    self.name, len(self.jobs), ll)
                await asyncio.sleep(5)

            # check if any pre_request_jobs has failed
            if not all(job.is_success() for job in self.jobs):
                message = ' '.join([job.name + ':' + job.status for job in self.jobs])
                for job in self.jobs:
                    logger.info(job.task.result)
                self.status = CoreStatus.FAIL
                logger.info(
                    '[Job] %s failed due to some job fail in jobs:[%s]',
                    self.name, message)

        if self.status in CoreStatus.DONE_STATUS:
            logger.info('[Job] %s failed status: %s', self.name, self.status)
            return

        self.status = CoreStatus.RUNNING
        self.task.run()

        while self.task.not_done():
            await self.update()
            if self.task.not_done():
                await asyncio.sleep(3)

        self.status = self.task.status
        logger.info('[Job] \'%s\' done status: %s id: %s', self.name, self.status, self.id)
        return

    async def update(self):
        """Update attributes from task's result.

        A python coroutine await by :func:`~decanter.core.jobs.job.Job.wait`.
        Will wait for task to update its result by await task.update(),
        then use the updated result from task to update Job's attributes.
        """
        await self.task.update()
        self.update_result(self.task.result)

    @abc.abstractmethod
    def update_result(self, task_result):
        """Update the task result to Job's attributes.

        Raises:
            NotImplementedError: If child class do not implement this function.
        """
        raise NotImplementedError('Please Implement update_result method')

    def stop(self):
        """Stop Job.

        Job will handle stoping itself by following conditions. If pending,
        will mark its status as fail, and do nothing and remains same status
        if it is done already. In running status the status will turn to fail,
        there are three conditions needed to be handle in terms of the status
        of task. It needs to call the api to stop task only if the task is in
        running status, else just mark task as fail if it haven't start running
        yet or else remains same done status.
        """
        if self.status == CoreStatus.PENDING:
            self.status = CoreStatus.FAIL
        elif self.status == CoreStatus.RUNNING:
            if self.task.status not in CoreStatus.DONE_STATUS:
                self.task.stop()
            self.status = CoreStatus.FAIL
        else:
            logger.info(
                '[Job] %s have finished already status %s',
                self.name, self.status)

        logger.info('[Job] %s stop successfully', self.name)

    @block_method
    def get(self, attr):
        """Get Job's attribute

        If it calls this function while Job is still undone, will appears
        message for remind.

        Args:
            attr (str): String that contains the attribute's name.
        Returns:
            Value of the named attribute of the given object.
        """
        return getattr(self, attr)
