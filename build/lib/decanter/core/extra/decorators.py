# pylint: disable=pointless-statement
"""
Decorators used by method.
"""
import logging

from functools import wraps

from decanter.core.extra import CoreStatus

logger = logging.getLogger(__name__)


def block_method(func):
    """Block when getting attribute while the Job isn't finished."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_done() is False:
            logger.info('%s attribute \'%s\' is not done.', self.name, args)
            return None
        try:
            ans = func(self, *args, **kwargs)
            if ans is None:
                raise AttributeError
            return ans
        except AttributeError:
            logger.error('%s no such attribute %s.', self.name, args)

    return wrapper


def update(func):
    """Update the key val to object attributes."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        class_type = self.__class__.__name__
        if not args[0]:
            return
        try:
            self.result = args[0]
            for attr, val in self.result.items():
                self.__dict__.update({attr if attr != '_id' else 'id': val})

        except AttributeError as err:
            logger.debug('[%s] \'%s\' %s', class_type, self.name, err)
            self.status = CoreStatus.FAIL

        else:
            func(self, *args, **kwargs)

    return wrapper


def corex_obj(required):
    """Decorator for create function of class:`<CoreXOBJ>`.

    Returns:
        None: if all required argument missing else
        class:`func <CoreOBJ>`

    Raises:
        ValueError: If required key missing in arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):

            if required is None:
                return func(cls, *args, **kwargs)

            if all(kwargs[x] is None for x in required):
                return None

            for req in required:
                if kwargs[req] is None:
                    raise ValueError(
                        '%s missing required value %s' % (cls.__name__, req))

            return func(cls, *args, **kwargs)

        return wrapper

    return decorator
