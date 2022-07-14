"""Initialization for running SDK."""
import asyncio
import logging

import pandas as pd

from decanter.core.core_api import CoreAPI, worker
from decanter.core.extra import CoreStatus

logger = logging.getLogger(__name__)


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


class Context:
    """Init the connection to decanter core server and functionality for running SDK.

    Example:
        .. code-block:: python

            from decanter import core
            context = core.Context.create(
                username='usr', password='pwd', host='decantercoreserver')
            context.run()

    """

    # 'str: User name to login in Decanter Core server'
    USERNAME = None
    # 'str: Password to login in Decanter Core server'
    PASSWORD = None
    # Decanter Core server\'s URL.'
    HOST = None
    # .. _EventLoop: https://docs.python.org/3/library/asyncio-eventloop.html#event-loop
    LOOP = None
    # List of Tasks of Asynchronous I/O.
    CORO_TASKS = []
    # list of finished and waited Jobs.
    JOBS = []
    # CoreX API endpoint
    api = None

    def __init__(self):
        pass

    @classmethod
    def create(cls, username, password, host):
        """Create context instance and init necessary variable and objects.

        Setting the user, password, and host for the funture connection when
        calling APIs, and create an event loop if it isn't exist. Check if the
        connection is healthy after args be set.

        Args:
            username (str): User name for login Decanter Core server
            password (str): Password name for login Decanter Core server
            host (str): Decanter Core server URL.

        Returns:
            :class:`~decanter.core.context.Context>`

        """
        context = cls()
        context.close()
        Context.USERNAME = username
        Context.PASSWORD = password
        Context.HOST = host

        # get the current event loop
        # it will create a new event loop if it does not exist
        Context.LOOP = get_or_create_eventloop()

        # if the current loop is closed create a new one
        if Context.LOOP.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            Context.LOOP = get_or_create_eventloop()
            logger.debug("[Context] create and set new event loop")
        context.healthy()
        Context.api = CoreAPI()
        return context

    @staticmethod
    def run():
        """Start execute the tasks in CORO_TASKs.

        Gather all tasks and execute.  It will block on all tasks until all
        have been finished.

        """
        logger.info("Run %s coroutines", len(Context.CORO_TASKS))

        if Context.LOOP is None:
            logger.error("[Context] create context before run")
            raise Exception()

        loop_running = Context.LOOP.is_running()
        logger.info("[Context] Context.LOOP.is_running(): {})".format(loop_running))
        if loop_running is False:
            groups = asyncio.gather(*Context.CORO_TASKS)
            Context.LOOP.run_until_complete(groups)
            Context.CORO_TASKS = []

    @staticmethod
    def close():
        """Close the event loop and reset JOBS and CORO_TASKS.

        Close the event loop if it's not running (will not close in
        Jupyter Notebook).

        """
        logger.debug("[Context] try to close context")
        if Context.LOOP is not None:
            Context.LOOP = get_or_create_eventloop()
            if Context.LOOP.is_running() is False:
                Context.LOOP.close()
                logger.info("[Context] close event loop successfully")
        else:
            logger.info("[Context] no event loop to close")
        logger.debug("[Context] remain CORO TASKS %s", len(Context.CORO_TASKS))
        Context.JOBS = []
        Context.CORO_TASKS = []
        Context.USERNAME = Context.PASSWORD = Context.HOST = None

    @staticmethod
    def healthy():
        """Check the connection between Decanter Core server.

        Send a fake request to determine if there's connection or
        authorization errors.

        """
        try:
            res = worker.Worker().get_status()
            if res.status_code // 100 != 2:
                raise Exception()
        except Exception as err:
            logger.error("[Context] connect not healthy :(")
            raise SystemExit(err)
        else:
            logger.info("[Context] connect healthy :)")

    @staticmethod
    def get_all_jobs():
        """Get a list of Jobs that have been or waiting to be executed.

        Returns:
            list(:class:`~decanter.core.jobs.job.Job`)

        """
        return Context.JOBS

    @staticmethod
    def get_jobs_status(sort_by_status=False, status=None):
        """Get a dataframe of jobs and its corresponding status. Return
        all jobs and its status if no arguments passed.

        Args:
            sort_by_status (:obj:`bool`, optional): DataFrame will sort by
                status, group the job with same status. Defaults to Faise.
            status (:obj:`list`(:obj:`str`), optional): Only select the job
                with the status in status list.
        Returns:
            :class:`pandas.DataFrame`: DataFrame with Job name and its status.

        Raises:
            Exception: If any status in status list is invalid.

        """
        jobs_status = {"Job": [], "status": []}
        for job in Context.JOBS:
            jobs_status["Job"].append(job.name)
            jobs_status["status"].append(job.status)

        jobs_df = pd.DataFrame(jobs_status)
        if status:
            if any(stat not in CoreStatus.ALL_STATUS for stat in status):
                raise Exception("Invalid status.")

            return jobs_df.loc[jobs_df["status"].isin(status)]

        if sort_by_status:
            jobs_df = jobs_df.sort_values(by=["status"])

        return jobs_df

    @staticmethod
    def get_jobs_by_name(names):
        """Get the Job instances by its name.

        Args:
            names (list(str)): Names of wish to select.

        Returns:
            list(:class:`~decanter.core.jobs.job.Job`): Jobs with name in names list.

        """
        res = []
        for job in Context.JOBS:
            if job.name in names:
                res.append(job)

        return res

    @staticmethod
    def stop_jobs(jobs_list):
        """Stop Jobs in jobs_list.

        Args:
            jobs_list (list(:class:`~decanter.core.jobs.job.Job`)):
                List of jobs instance wished to be stopped.
        """
        for job in jobs_list:
            job.stop()

    @staticmethod
    def stop_all_jobs():
        """Stop all Jobs which status is still in pending or running"""
        for job in Context.JOBS:
            if job.status not in ["done", "fail", "invalid"]:
                job.stop()
