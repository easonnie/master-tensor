"""
Logging module:

loggers, : emmit a LogRecord instance and then pass below
handlers,
filters,
formatter.
"""
import logging

logger_a = logging.getLogger("A")
"""
Default logger is 'ROOT'
"""
logger_a.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(formatter)

fh = logging.FileHandler(filename='log.txt')
fh.setLevel(logging.INFO)

logger_a.addHandler(fh)
logger_a.addHandler(ch)
logger_a.info(msg='hello world')
