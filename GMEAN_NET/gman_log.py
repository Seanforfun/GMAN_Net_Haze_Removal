#  ====================================================
#   Filename: gman_log.py
#   Author: Seanforfun
#   Function: Dehazenet project's log module.
#  ====================================================
import logging

LOGGING_LEVEL = logging.INFO

logging.basicConfig(level=LOGGING_LEVEL,format='%(asctime)s  [ %(levelname)s ]: %(message)s')
logger = logging.getLogger(__name__)

handler = logging.FileHandler("./log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def info(message):
    logger.info(message)


def warn(message):
    logger.warning(message)


def error(message):
    logger.error(message)


def debug(message):
    logger.debug(message)


if __name__ == '__main__':
    pass
