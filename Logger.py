#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging.handlers
from logging.handlers import RotatingFileHandler
import traceback

class Logger:
    def __init__(self, logName, logPath='/log/', to_msecs=True):
        try:
            if not os.path.exists(logPath):
                os.makedirs(logPath)
                
            ### Setting log
            self.__logger = logging.getLogger(logName)
            self.close()
            self.__logger.setLevel(logging.DEBUG)
            fileName = os.path.join(logPath, logName + '.log')
            self.__fHandler = RotatingFileHandler(fileName, maxBytes=4194304, backupCount=5)
            self.__fHandler.setLevel(logging.DEBUG)

            fmt = '%(asctime)s %(levelname)-8s | %(message)s'

            if to_msecs:
                fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s | %(message)s'

            self.__formatter= logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
            self.__fHandler.setFormatter(self.__formatter)
            
            self.__logger.addHandler(self.__fHandler)

            ### console handler
            self.__consoleHandler = logging.StreamHandler()
            self.__consoleHandler.setLevel(logging.DEBUG)
            self.__consoleHandler.setFormatter(self.__formatter)
            self.__logger.addHandler(self.__consoleHandler)
            
        except:
            print('AUOLog init fail.')
            traceback.print_exc()
    
    def debug(self, msg):
        self.__logger.debug(msg)
    
    def info(self, msg):
        self.__logger.info(msg)

    def warning(self, msg):
        self.__logger.warning(msg)

    def error(self, msg):
        self.__logger.error(msg)

    def critical(self, msg):
        self.__logger.critical(msg)

    def exception(self, msg):
        self.__logger.exception(msg)

    def log(self, level, msg):
        self.__logger.log(level, msg)

    def setLevel(self, level):
        self.__logger.setLevel(level)

    def disable(self):
        logging.disable(50)

    def close(self):
        handlers = self.__logger.handlers[:]
        for handler in handlers:
            handler.flush()
            if not (handler is None):
                handler.close()
            self.__logger.removeHandler(handler)
            
if __name__ == "__main__":
    logName = "log_test"
    logPath = f"/cccchiang/TabNet/log"
    logger = Logger(logName, logPath) ### Declare Log file
    logger.info(f"Save log file at: {logPath}/log")

