from datetime import datetime
from .constants import DATETIME_FORMAT


def date_time_formatter(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FORMAT) if timestamp else ""