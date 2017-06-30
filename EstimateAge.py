#!/usr/bin/env python

import ControllerThread
import ConfigParser
import sys

if __name__ == '__main__':

    help_message = '''
    USAGE: EstimateAge.py [params file]
    '''

    if len(sys.argv) > 1:
        paramFile = sys.argv[1]
    else:
        paramFile = "config.ini"

    params = ConfigParser.ConfigParser()
    params.read(paramFile)
    
    # Initialize controller thread

    controllerThread = ControllerThread.ControllerThread(params)
    controllerThread.start()
