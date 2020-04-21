# author:hjt
# contact: hanguantianxia@sina.com
# datetime:2020/4/17 13:07
# software: PyCharm

"""
test logging function
"""

import logging
from utils import log
#
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
#
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# logging.debug("This is a debug log.")
# logging.info("This is a info log.")
# logging.warning("This is a warning log.")
# logging.error("This is a error log.")
# logging.critical("This is a critical log.")

logger = log()

logger.info('ew')
logger = log()
logger.warning("泰拳警告")
logger.info("提示")
logger.error("错误")
logger.debug("查错")