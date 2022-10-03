from enum import Enum


class Status(Enum):
    pending = "pending"
    running = "running"
    fail = "fail"
    done = "done"
    prepare = "prepare"
